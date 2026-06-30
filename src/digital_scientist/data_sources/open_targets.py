import os
import re
from difflib import get_close_matches

import requests

OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"
GROK_URL = "https://api.x.ai/v1/chat/completions"

TYPO_ALIASES = {
  "heart": "heart disease",
    "alzhimers": "alzheimer disease",
    "alzheimers": "alzheimer disease",
    "alzhiemers": "alzheimer disease",
    "diabtes": "diabetes",
    "diabetis": "diabetes",
    "asthmaa": "asthma",
    "astmaa": "asthma",
    "hert disease": "heart disease",
    "lung canser": "lung cancer",
}

CANONICAL_DISEASE_QUERIES = [
    "alzheimer disease",
    "diabetes",
    "asthma",
    "heart disease",
    "lung cancer",
    "breast cancer",
    "prostate cancer",
    "colorectal cancer",
    "parkinson disease",
    "chronic kidney disease",
    "hypertension",
]


def _normalize_text(text):
  text = text.lower().replace("'s", "s")
  text = text.replace("'", "")
  return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _search_hits(query_text):
  query = """
  query($q: String!) {
    search(queryString: $q) {
      hits { id name }
    }
  }
  """
  data = _post_graphql(query, {"q": query_text})
  return data["data"]["search"]["hits"]


def _grok_suggest_query(disease_name):
  """
  Optional fallback to normalize heavily misspelled disease names.
  Uses Grok only when local alias/fuzzy matching cannot find Open Targets hits.
  """
  api_key = os.getenv("GROK_API_KEY")
  if not api_key:
    return None

  model = os.getenv("GROK_MODEL", "grok-3-mini")
  prompt = (
      "You normalize user disease text for biomedical search. "
      "Return only one short canonical disease phrase in plain text, no punctuation, no explanation. "
      f"Input: {disease_name}"
  )

  try:
    response = requests.post(
        GROK_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=12,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"].strip()
    return content.splitlines()[0].strip(" \t\"'`.,")[:80]
  except Exception:
    # Keep pipeline resilient even if Grok is not available.
    return None


def _post_graphql(query, variables=None):
  response = requests.post(OT_URL, json={"query": query, "variables": variables or {}}, timeout=20)
  response.raise_for_status()

  data = response.json()
  if "errors" in data:
    raise RuntimeError(f"Open Targets API error: {data['errors']}")

  return data


def get_disease_id(disease_name, return_meta=False):
    query_norm = _normalize_text(disease_name)
    query_text = TYPO_ALIASES.get(query_norm, disease_name)
    resolver = "alias" if query_text != disease_name else "direct"
    hits = _search_hits(query_text)

    # Fallback 1: explicit typo aliases for common misspellings.
    if not hits and query_text != disease_name:
      hits = _search_hits(disease_name)
      if hits:
        resolver = "direct"

    # Fallback 2: fuzzy correction against common disease queries.
    if not hits:
      close = get_close_matches(query_norm, CANONICAL_DISEASE_QUERIES, n=1, cutoff=0.72)
      if close:
        query_text = close[0]
        hits = _search_hits(query_text)
        if hits:
          resolver = "fuzzy"

    # Fallback 3: optional Grok normalization only if local methods find nothing.
    if not hits:
      grok_query = _grok_suggest_query(disease_name)
      if grok_query and _normalize_text(grok_query) != query_norm:
        query_text = grok_query
        hits = _search_hits(grok_query)
        if hits:
          resolver = "grok"

    def hit_score(hit):
      hit_id = hit["id"]
      name = hit.get("name", "")
      name_norm = _normalize_text(name)

      score = 0
      if hit_id.startswith(("EFO_", "MONDO_")):
        score += 10
      if name_norm == query_norm:
        score += 100
      elif query_norm in name_norm:
        score += 80
      elif name_norm in query_norm:
        score += 60
      if "biomarker measurement" in name_norm:
        score -= 200
      if "disease" in name_norm:
        score += 5
      return score

    disease_hits = [hit for hit in hits if hit["id"].startswith(("EFO_", "MONDO_"))]
    if disease_hits:
      best_hit = max(disease_hits, key=hit_score)
      if return_meta:
        return best_hit["id"], best_hit["name"], {
            "resolver": resolver,
            "query_used": query_text,
        }
      return best_hit["id"], best_hit["name"]

    if hits:
      best_hit = max(hits, key=hit_score)
      if return_meta:
        return best_hit["id"], best_hit["name"], {
            "resolver": resolver,
            "query_used": query_text,
        }
      return best_hit["id"], best_hit["name"]

    raise ValueError(f"No disease ID found for: {disease_name}")


def get_top_targets(disease_id, min_score=0.6, min_genetic=0.2, top_n=7):
    query = f"""
    query {{
      disease(efoId: "{disease_id}") {{
        associatedTargets(page: {{size: 50, index: 0}}) {{
          rows {{
            score
            datatypeScores {{ id score }}
            target {{ id approvedSymbol approvedName }}
          }}
        }}
      }}
    }}
    """
    data = _post_graphql(query)

    rows = data["data"]["disease"]["associatedTargets"]["rows"]

    filtered = [
        t for t in rows
        if t["score"] >= min_score and any(
            d["id"] == "genetic_association" and d["score"] > min_genetic
            for d in t["datatypeScores"]
        )
    ]

    top = sorted(filtered, key=lambda x: x["score"], reverse=True)[:top_n]

    return [
        {
            "ensembl_id": t["target"]["id"],
            "symbol": t["target"]["approvedSymbol"],
            "name": t["target"]["approvedName"],
            "score": round(t["score"], 4),
        }
        for t in top
    ]
