import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
SESSION = requests.Session()  # reuse TCP connections


def _get(url, params=None):
    response = SESSION.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def _fetch_molecule(mol_id, symbol, pchembl, activity_type):
    """Fetch a single molecule's details — runs in parallel."""
    mol_data = _get(f"{CHEMBL_API}/molecule/{mol_id}.json")
    pref_name = mol_data.get("pref_name")
    if not pref_name:
        synonyms = mol_data.get("molecule_synonyms", [])
        pref_name = synonyms[0].get("synonyms", mol_id) if synonyms else mol_id
    structures = mol_data.get("molecule_structures") or {}
    smiles = structures.get("canonical_smiles")
    return {
        "chembl_id": mol_id,
        "name": pref_name,
        "smiles": smiles,
        "pchembl": pchembl,
        "activity_type": activity_type,
        "target_symbol": symbol,
    }


def _fetch_target(target, limit):
    """Fetch activities for one target — runs in parallel."""
    symbol = target["symbol"]
    results = _get(f"{CHEMBL_API}/target/search.json", {"q": symbol, "limit": 1}).get("targets", [])
    if not results:
        return []
    chembl_id = results[0]["target_chembl_id"]
    activities = _get(f"{CHEMBL_API}/activity.json", {
        "target_chembl_id": chembl_id,
        "pchembl_value__gte": 6,
        "limit": limit,
    }).get("activities", [])
    return [(a.get("molecule_chembl_id"), symbol, a.get("pchembl_value"), a.get("standard_type"))
            for a in activities if a.get("molecule_chembl_id")]


def fetch_compounds(top_targets, limit=5):
    # Step 1: fetch all target activity lists in parallel
    mol_tasks = []
    with ThreadPoolExecutor(max_workers=7) as ex:
        futures = {ex.submit(_fetch_target, t, limit): t for t in top_targets}
        for f in as_completed(futures):
            mol_tasks.extend(f.result())

    # Step 2: fetch all molecule details in parallel
    compounds = []
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = [ex.submit(_fetch_molecule, *task) for task in mol_tasks]
        for f in as_completed(futures):
            result = f.result()
            if result:
                compounds.append(result)

    return compounds
