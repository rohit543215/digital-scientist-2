"""
dataset.py
Pulls 500k bioactivity records from the local ChEMBL SQLite dump,
generates Morgan fingerprints, and saves .npy files for training.
"""

import logging
import os
import sqlite3

import numpy as np
import pandas as pd
import chembl_downloader
from rdkit import Chem, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from tqdm import tqdm

logging.disable(logging.WARNING)
RDLogger.DisableLog("rdApp.*")

FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS   = 2048
TARGET_RECORDS     = 500_000


def get_db_path():
    """Download ChEMBL SQLite (latest version) if not already cached."""
    print("Fetching ChEMBL SQLite database (downloads once, ~3GB)...")
    path = chembl_downloader.download_extract_sqlite()
    print(f"Database ready at: {path}")
    return path


def query_activities(db_path, limit=TARGET_RECORDS):
    """
    Query binding assay records that have:
      - a valid SMILES
      - a measured pChEMBL value (proxy for potency)
    """
    sql = f"""
    SELECT
        md.chembl_id        AS chembl_id,
        cs.canonical_smiles AS smiles,
        act.pchembl_value   AS pchembl
    FROM
        activities          act
        JOIN molecule_dictionary  md  ON act.molregno   = md.molregno
        JOIN compound_structures  cs  ON act.molregno   = cs.molregno
        JOIN assays               a   ON act.assay_id   = a.assay_id
    WHERE
        act.pchembl_value IS NOT NULL
        AND cs.canonical_smiles IS NOT NULL
        AND a.assay_type = 'B'
    LIMIT {limit}
    """
    print(f"Querying {limit:,} records from SQLite...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    print(f"Retrieved {len(df):,} records.")
    return df


morgan_gen = GetMorganGenerator(radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_BITS)

def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = morgan_gen.GetFingerprintAsNumPy(mol)
    return fp.astype(np.float32)


def build_fingerprints(df, out_X="fingerprints_X.npy", out_y="labels_y.npy"):
    print("Generating Morgan fingerprints (chunked to save memory)...")
    n = len(df)

    tmp_X = out_X + ".tmp"
    tmp_y = out_y + ".tmp"

    # Write directly to disk via memmap
    X = np.lib.format.open_memmap(tmp_X, mode="w+", dtype=np.float32, shape=(n, FINGERPRINT_BITS))
    y = np.lib.format.open_memmap(tmp_y, mode="w+", dtype=np.float32, shape=(n,))

    valid = 0
    for _, row in tqdm(df.iterrows(), total=n):
        fp = smiles_to_fingerprint(row["smiles"])
        if fp is not None:
            X[valid] = fp
            y[valid] = 1 if float(row["pchembl"]) >= 6.0 else 0
            valid += 1

    X.flush()
    y.flush()
    del X, y  # close memmap handles before reading

    # Load trimmed slices and save as final .npy
    X_tmp = np.load(tmp_X, mmap_mode="r")
    y_tmp = np.load(tmp_y, mmap_mode="r")
    X_final = np.array(X_tmp[:valid], dtype=np.float32)
    y_final = np.array(y_tmp[:valid], dtype=np.float32)
    del X_tmp, y_tmp  # release memmap handles before deleting files

    np.save(out_X, X_final)
    np.save(out_y, y_final)

    # Clean up temp files
    try:
        os.remove(tmp_X)
        os.remove(tmp_y)
    except OSError:
        pass  # non-critical, just temp files

    print(f"\nFinal dataset : {valid:,} compounds")
    print(f"  Active      : {y_final.sum():,.0f} ({y_final.mean()*100:.1f}%)")
    print(f"  Inactive    : {(1-y_final).sum():,.0f} ({(1-y_final.mean())*100:.1f}%)")
    return valid


def main():
    db_path = get_db_path()
    df = query_activities(db_path)
    df.to_csv("raw_activities.csv", index=False)
    print("Saved raw_activities.csv")

    build_fingerprints(df)
    print("Saved fingerprints_X.npy and labels_y.npy")


if __name__ == "__main__":
    main()
