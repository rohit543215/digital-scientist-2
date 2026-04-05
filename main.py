import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from open_targets import get_disease_id, get_top_targets
from chembl import fetch_compounds
from drug_likeness import filter_drug_like

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "neural_network"))
try:
    from predict import score_compounds
    NN_AVAILABLE = True
except Exception:
    NN_AVAILABLE = False

load_dotenv()

client = OpenAI(api_key=os.getenv("API_KEY"), base_url="https://openrouter.ai/api/v1")
MODEL = os.getenv("LLM_MODEL")

W = 60  # output width

def divider(char="─"): print(char * W)
def header(text): print(f"\n{'═' * W}\n  {text}\n{'═' * W}")
def section(n, text): print(f"\n  Step {n}  {text}")
def row(label, value): print(f"    {label:<18} {value}")


def ai_summary(disease, targets, candidates):
    target_lines = "\n".join(
        f"  - {t['symbol']} ({t['name']}) | score: {t['score']}" for t in targets
    )
    candidate_lines = "\n".join(
        f"  - {c['name']} | Target: {c['target_symbol']} | MW: {c['mw']} | LogP: {c['logp']} | pChEMBL: {c['pchembl']}"
        for c in candidates[:10]
    )
    prompt = f"""You are a drug discovery AI. Summarize findings for {disease} concisely.

Top targets:
{target_lines}

Drug-like candidates (passed Lipinski Ro5):
{candidate_lines}

Give:
1. Most promising targets and why
2. Best molecule candidates and why
3. Recommended next steps

Keep under 200 words. Be direct."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def run(disease):
    header(f"Drug Discovery  ·  {disease}")

    # ── Step 1: Disease → Targets ──────────────────────────────
    section(1, "Disease → Targets  (Open Targets)")
    disease_id, disease_label = get_disease_id(disease)
    row("Disease", f"{disease_label}")
    row("ID", disease_id)

    targets = get_top_targets(disease_id)
    print(f"\n    {'Symbol':<10} {'Score':>6}  Name")
    divider()
    for t in targets:
        print(f"    {t['symbol']:<10} {t['score']:>6}  {t['name'][:38]}")

    # ── Step 2: Targets → Molecules ────────────────────────────
    section(2, "Targets → Molecules  (ChEMBL)")
    compounds = fetch_compounds(targets)
    row("Compounds found", str(len(compounds)))

    # ── Step 3: Drug-Likeness Filter ───────────────────────────
    section(3, "Drug-Likeness Filter  (Lipinski Ro5)")
    passed, failed = filter_drug_like(compounds)
    row("Passed", str(len(passed)))
    row("Failed", str(len(failed)))

    if passed:
        print(f"\n    {'Molecule':<28} {'Target':<8} {'MW':>6} {'LogP':>6} {'pChEMBL':>8}")
        divider()
        for c in passed:
            print(f"    {c['name']:<28} {c['target_symbol']:<8} {c['mw']:>6} {c['logp']:>6} {str(c['pchembl']):>8}")

    # ── Step 3.5: Neural Network Scoring ───────────────────────
    if NN_AVAILABLE and passed:
        section("3.5", "Bioactivity Scoring  (Neural Network)")
        passed = score_compounds(passed)
        print(f"\n    {'Molecule':<28} {'Target':<8} {'AI Score':>9}")
        divider()
        for c in passed[:10]:
            bar = "█" * int(c['activity_score'] * 10)
            print(f"    {c['name']:<28} {c['target_symbol']:<8} {c['activity_score']:>6.4f}  {bar}")

    # ── Step 4: AI Summary ─────────────────────────────────────
    section(4, "AI Analysis  (Gemini)")
    if not os.getenv("API_KEY"):
        print("    ⚠  API_KEY not set — skipping.")
    elif not passed:
        print("    ⚠  No candidates to summarize.")
    else:
        print()
        summary = ai_summary(disease, targets, passed)
        divider()
        print(summary)
        divider()

    print(f"\n  ✓ Done  ·  {len(passed)} drug-like candidates for '{disease}'\n")
    return passed


if __name__ == "__main__":
    disease_input = input("Enter disease name: ").strip()
    run(disease_input)
