"""
pipeline.py
Single entry point — runs all steps in order and returns results.
"""

import os
from dotenv import load_dotenv
from digital_scientist.data_sources.open_targets import get_disease_id, get_top_targets
from digital_scientist.data_sources.chembl import fetch_compounds
from digital_scientist.filters.drug_likeness import filter_drug_like
from digital_scientist.agents.biology_agent import summarize_targets
from digital_scientist.agents.chemistry_agent import summarize_candidates

NN_IMPORT_ERROR = ""
try:
    from neural_network.predict import score_compounds, model_healthcheck
    NN_AVAILABLE = True
except Exception as exc:
    NN_AVAILABLE = False
    NN_IMPORT_ERROR = str(exc)

load_dotenv()


def run(disease_name: str) -> dict:
    # Step 1: Disease → Targets
    disease_id, label, disease_lookup = get_disease_id(disease_name, return_meta=True)
    targets = get_top_targets(disease_id)

    # Step 2: Targets → Compounds (parallel)
    compounds = fetch_compounds(targets)

    # Step 3: Drug-likeness filter
    passed, failed = filter_drug_like(compounds)

    # Step 3.5: Neural network scoring
    if NN_AVAILABLE and passed:
        passed = score_compounds(passed)

    # Step 4: Agent summaries
    biology_summary   = summarize_targets(label, targets)
    chemistry_summary = summarize_candidates(label, passed)

    return {
        "disease": label,
        "disease_id": disease_id,
        "disease_lookup": disease_lookup,
        "targets": targets,
        "candidates": passed,
        "failed": len(failed),
        "biology_summary": biology_summary,
        "chemistry_summary": chemistry_summary,
    }


def get_model_status(test_smiles: str = "CCO") -> dict:
    """Return neural network readiness diagnostics for best_model.pt."""
    if not NN_AVAILABLE:
        return {
            "ok": False,
            "message": f"Neural network module not available: {NN_IMPORT_ERROR}",
        }
    return model_healthcheck(test_smiles=test_smiles)
