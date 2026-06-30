"""
ChemistryAgent
Takes raw compound data and writes a clear, structured summary of the chemistry findings.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model=os.getenv("LLM_MODEL"),
    temperature=0.3,
)


def summarize_candidates(disease: str, candidates: list) -> str:
    """Takes candidate list and returns a plain-language chemistry summary."""
    if not candidates:
        return "No drug-like candidates were found for this disease."

    lines = "\n".join(
        f"- {c.get('name', c.get('chembl_id'))} | Target: {c.get('target_symbol')} | "
        f"MW: {c.get('mw')} Da | LogP: {c.get('logp')} | "
        f"Potency (pChEMBL): {c.get('pchembl')} | AI Score: {c.get('activity_score', 'N/A')}"
        for c in candidates[:10]
    )

    prompt = f"""You are a medicinal chemist. Explain these drug candidates for {disease} in simple language.

Candidates (all passed drug-likeness filter):
{lines}

For the top candidates explain:
- Why they look promising (potency, drug-likeness)
- What the AI score means
- Which ones stand out and why

Keep it concise, structured, and easy to understand."""

    return llm.invoke(prompt).content
