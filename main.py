import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from digital_scientist.pipeline import run as run_pipeline

load_dotenv()

W = 60

def divider(): print("─" * W)
def header(text): print(f"\n{'═' * W}\n  {text}\n{'═' * W}")
def section(n, text): print(f"\n  Step {n}  {text}")


def display(disease: str):
    header(f"Drug Discovery  ·  {disease}")
    print("\n  Running agents...\n")

    result = run_pipeline(disease)

    # ── Targets ────────────────────────────────────────────────
    section(1, "Biology Agent  →  Top Targets")
    print(f"\n    {'Symbol':<10} {'Score':>6}  Name")
    divider()
    for t in result["targets"]:
        print(f"    {t['symbol']:<10} {t['score']:>6}  {t['name'][:38]}")

    # ── Candidates ─────────────────────────────────────────────
    section(2, "Chemistry Agent  →  Drug Candidates")
    candidates = result["candidates"]
    if candidates:
        print(f"\n    {'Molecule':<28} {'Target':<8} {'MW':>6} {'LogP':>6} {'Score':>7}")
        divider()
        for c in candidates[:10]:
            score = f"{c['activity_score']:.4f}" if c.get("activity_score") else "  N/A"
            print(f"    {str(c.get('name', c.get('chembl_id','?'))):<28} "
                  f"{str(c.get('target_symbol','')):<8} "
                  f"{str(c.get('mw','?')):>6} "
                  f"{str(c.get('logp','?')):>6} "
                  f"{score:>7}")
    else:
        print("    No drug-like candidates found.")

    # ── Summary ────────────────────────────────────────────────
    section(3, "Coordinator  →  AI Summary")
    print()
    divider()
    print("Biology Agent:\n")
    print(result.get("biology_summary", "No biology summary available."))
    print("\nChemistry Agent:\n")
    print(result.get("chemistry_summary", "No chemistry summary available."))
    divider()

    print(f"\n  ✓ Done  ·  {len(candidates)} candidates found for '{disease}'\n")


if __name__ == "__main__":
    disease_input = input("Enter disease name: ").strip()
    display(disease_input)
