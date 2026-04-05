from rdkit import Chem
from rdkit.Chem import Descriptors


def check_lipinski(smiles):
    """
    Lipinski's Rule of Five:
      - MW < 500 Da
      - LogP < 5
      - H-bond donors < 5
      - H-bond acceptors < 10
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = Descriptors.NumHDonors(mol)
    hba  = Descriptors.NumHAcceptors(mol)

    violations = sum([mw >= 500, logp >= 5, hbd >= 5, hba >= 10])

    return {
        "mw": round(mw, 2),
        "logp": round(logp, 2),
        "hbd": hbd,
        "hba": hba,
        "violations": violations,
        "drug_like": violations == 0,
    }


def filter_drug_like(compounds):
    passed, failed = [], []

    for c in compounds:
        smiles = c.get("smiles")
        if not smiles:
            continue

        props = check_lipinski(smiles)
        if props is None:
            continue

        entry = {**c, **props}
        (passed if props["drug_like"] else failed).append(entry)

    return passed, failed
