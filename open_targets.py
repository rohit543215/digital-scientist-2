import requests

OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"


def get_disease_id(disease_name):
    query = """
    query($q: String!) {
      search(queryString: $q) {
        hits { id name }
      }
    }
    """
    r = requests.post(OT_URL, json={"query": query, "variables": {"q": disease_name}})
    hits = r.json()["data"]["search"]["hits"]

    for hit in hits:
        if hit["id"].startswith("MONDO"):
            return hit["id"], hit["name"]
    for hit in hits:
        if hit["id"].startswith("EFO"):
            return hit["id"], hit["name"]

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
    r = requests.post(OT_URL, json={"query": query})
    data = r.json()

    if "errors" in data:
        raise RuntimeError(f"Open Targets API error: {data['errors']}")

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
