"""
predict.py
Loads the trained BioactivityNet and scores new compounds from ChEMBL.
Returns compounds ranked by predicted bioactivity probability.
"""

import os
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from model import BioactivityNet

RDLogger.DisableLog("rdApp.*")

FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS   = 2048
MODEL_PATH         = os.path.join(os.path.dirname(__file__), "best_model.pt")
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

morgan_gen = GetMorganGenerator(radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_BITS)

# Load model once at import time
_model = None

def get_model():
    global _model
    if _model is None:
        _model = BioactivityNet().to(DEVICE)
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        _model.eval()
    return _model


def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return morgan_gen.GetFingerprintAsNumPy(mol).astype(np.float32)


def score_compounds(compounds):
    """
    Takes a list of compound dicts (must have 'smiles' key).
    Adds 'activity_score' (0-1) and 'predicted_active' (bool) to each.
    Returns list sorted by activity_score descending.
    """
    model = get_model()

    valid, fps, indices = [], [], []
    for i, c in enumerate(compounds):
        fp = smiles_to_fp(c.get("smiles", ""))
        if fp is not None:
            fps.append(fp)
            indices.append(i)

    if not fps:
        return compounds

    X = torch.tensor(np.array(fps), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        scores = model(X).cpu().numpy()

    for idx, score in zip(indices, scores):
        compounds[idx]["activity_score"] = round(float(score), 4)
        compounds[idx]["predicted_active"] = bool(score >= 0.5)

    # Mark compounds with no SMILES
    for c in compounds:
        if "activity_score" not in c:
            c["activity_score"] = 0.0
            c["predicted_active"] = False

    return sorted(compounds, key=lambda x: x["activity_score"], reverse=True)
