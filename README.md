# AI Drug Discovery Pipeline

An end-to-end drug discovery system that takes a disease name as input and returns ranked drug candidates using biomedical databases, cheminformatics, and a trained neural network.

## How It Works

```
Disease Name
    │
    ▼
Step 1 — Open Targets
    Finds the top 7 high-confidence biological targets
    associated with the disease using genetic evidence scores
    │
    ▼
Step 2 — ChEMBL
    Fetches potent compounds (pChEMBL ≥ 6) for each target
    from the ChEMBL bioactivity database
    │
    ▼
Step 3 — RDKit (Lipinski's Rule of Five)
    Filters molecules by drug-likeness:
    MW < 500 · LogP < 5 · HBD < 5 · HBA < 10
    │
    ▼
Step 3.5 — Neural Network (BioactivityNet)
    Scores each candidate using a PyTorch model trained on
    500k ChEMBL compounds. Ranks by predicted bioactivity.
    │
    ▼
Step 4 — Gemini (via OpenRouter)
    AI summary: best targets, best molecules, next steps
```

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up environment
Create a `.env` file in the project root:
```
API_KEY=your-openrouter-api-key
LLM_MODEL=google/gemini-flash-1.5
```
Get a free API key at [openrouter.ai](https://openrouter.ai)

### 3. Run
```bash
python main.py
```
Enter a disease when prompted:
```
Enter disease name: lung cancer
```

## Example Output

```
════════════════════════════════════════════════════════════
  Drug Discovery  ·  heart disease
════════════════════════════════════════════════════════════

  Step 1  Disease → Targets  (Open Targets)
    Disease            Heart failure
    ID                 MONDO_0005267

    Symbol      Score  Name
────────────────────────────────────────────────────────────
    SCN5A        0.894  Sodium channel protein type 5 subunit
    RYR2         0.865  Ryanodine receptor 2
    KCNQ1        0.789  Potassium voltage-gated channel

  Step 3.5  Bioactivity Scoring  (Neural Network)
    Molecule                     Target   AI Score
────────────────────────────────────────────────────────────
    CHEMBL338973                 KCNQ1    0.8821  ████████
    CHEMBL5639861                RYR2     0.8102  ████████
```

## Project Structure

```
├── main.py               # Entry point — orchestrates all steps
├── open_targets.py       # Step 1: disease → targets via Open Targets GraphQL API
├── chembl.py             # Step 2: targets → compounds via ChEMBL REST API
├── drug_likeness.py      # Step 3: Lipinski Ro5 filter using RDKit
├── requirements.txt      # Python dependencies
├── .env                  # API keys (not committed)
│
└── neural_network/
    ├── dataset.py        # Downloads ChEMBL SQLite, builds Morgan fingerprints
    ├── model.py          # PyTorch BioactivityNet — train on Google Colab
    ├── predict.py        # Loads best_model.pt and scores new compounds
    ├── best_model.pt     # Trained model weights (500k compounds, 30 epochs)
    └── colab_train.ipynb # Colab notebook for training
```

## Neural Network

The model is a feedforward network trained on 500,000 ChEMBL binding assay records.

- Input: 2048-bit Morgan fingerprints (radius 2)
- Architecture: `2048 → 1024 → 512 → 256 → 1`
- Batch normalization + dropout (0.3) at each layer
- Trained with weighted sampling to handle class imbalance
- Label: active if pChEMBL ≥ 6.0 (IC50 ≤ 1 µM)

To retrain the model:
1. Run `python neural_network/dataset.py` to rebuild the dataset
2. Upload `fingerprints_X.npy` and `labels_y.npy` to Google Drive
3. Open `neural_network/colab_train.ipynb` in Google Colab (T4 GPU)
4. Run all cells — training takes ~15 min
5. Download `best_model.pt` and place it in `neural_network/`

## Data Sources

| Source | What it provides | Access |
|---|---|---|
| [Open Targets](https://platform.opentargets.org) | Disease-target associations | Free, no key |
| [ChEMBL](https://www.ebi.ac.uk/chembl) | Bioactivity data, SMILES | Free, no key |
| [OpenRouter](https://openrouter.ai) | LLM API (Gemini) | Free tier available |

## Requirements

```
requests
openai
python-dotenv
rdkit
chembl-downloader
torch
scikit-learn
pandas
tqdm
matplotlib
```
