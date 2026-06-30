"""
BiologyAgent
Takes raw target data and writes a clear, structured summary of the biological findings.
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


def summarize_targets(disease: str, targets: list) -> str:
    """Takes target list and returns a plain-language biological summary."""
    lines = "\n".join(
        f"- {t['symbol']} ({t['name']}) | confidence score: {t['score']}"
        for t in targets
    )
    prompt = f"""You are a biology expert. Explain these drug targets for {disease} in simple, clear language.

Targets found:
{lines}

For each target briefly explain:
- What it does in the body
- Why it's relevant to {disease}
- How confident we are (based on the score)

Keep it concise and easy to understand. No jargon."""

    return llm.invoke(prompt).content
