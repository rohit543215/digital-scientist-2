import requests

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"


def fetch_compounds(top_targets, limit=5):
    compounds = []

    for target in top_targets:
        symbol = target["symbol"]

        r = requests.get(f"{CHEMBL_API}/target/search.json", params={"q": symbol, "limit": 1})
        results = r.json().get("targets", [])
        if not results:
            continue

        chembl_id = results[0]["target_chembl_id"]

        activities = requests.get(
            f"{CHEMBL_API}/activity.json",
            params={"target_chembl_id": chembl_id, "pchembl_value__gte": 6, "limit": limit}
        ).json().get("activities", [])

        for c in activities:
            mol_id = c.get("molecule_chembl_id")
            if not mol_id:
                continue

            mol_data = requests.get(f"{CHEMBL_API}/molecule/{mol_id}.json").json()
            pref_name = mol_data.get("pref_name")
            if not pref_name:
                synonyms = mol_data.get("molecule_synonyms", [])
                pref_name = synonyms[0].get("synonyms", mol_id) if synonyms else mol_id

            structures = mol_data.get("molecule_structures") or {}
            smiles = structures.get("canonical_smiles")

            compounds.append({
                "chembl_id": mol_id,
                "name": pref_name,
                "smiles": smiles,
                "pchembl": c.get("pchembl_value"),
                "activity_type": c.get("standard_type"),
                "target_symbol": symbol,
            })

    return compounds
