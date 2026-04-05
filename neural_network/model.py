"""
model.py
PyTorch neural network for molecular bioactivity prediction.
Input: 2048-bit Morgan fingerprints
Output: probability of being active (pChEMBL >= 6)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
FINGERPRINT_BITS = 2048
BATCH_SIZE       = 512
EPOCHS           = 30
LEARNING_RATE    = 1e-3
DROPOUT          = 0.3
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"


# ── Dataset ───────────────────────────────────────────────────────────────────
class MoleculeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Model ─────────────────────────────────────────────────────────────────────
class BioactivityNet(nn.Module):
    """
    3-layer feedforward network with batch norm and dropout.
    2048 → 1024 → 512 → 256 → 1
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FINGERPRINT_BITS, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# ── Training ──────────────────────────────────────────────────────────────────
def make_loader(X, y, shuffle=True, balance=False):
    dataset = MoleculeDataset(X, y)
    sampler = None

    if balance:
        # Oversample minority class to handle imbalance
        counts = np.bincount(y.astype(int))
        weights = 1.0 / counts[y.astype(int)]
        sampler = WeightedRandomSampler(weights, len(weights))
        shuffle = False

    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, sampler=sampler)


def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            probs = model(X_batch).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy().astype(int))

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return acc, auc, all_labels, all_preds


def plot_history(train_losses, val_accs, val_aucs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train Loss")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(val_accs, label="Accuracy")
    ax2.plot(val_aucs, label="ROC-AUC")
    ax2.axhline(0.85, color="red", linestyle="--", label="85% target")
    ax2.set_title("Validation Metrics")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("Saved training_history.png")


def main():
    print(f"Device: {DEVICE}")

    # ── Load data ──────────────────────────────────────────────
    print("Loading dataset...")
    # Support loading from Google Drive path via environment variable
    data_dir = os.environ.get("DATA_DIR", ".")
    X = np.load(os.path.join(data_dir, "fingerprints_X.npy"), mmap_mode="r")
    y = np.load(os.path.join(data_dir, "labels_y.npy"), mmap_mode="r")
    print(f"Loaded {X.shape[0]:,} compounds | Features: {X.shape[1]}")
    print(f"Class balance — Active: {y.sum():,.0f} ({y.mean()*100:.1f}%) | Inactive: {(1-y).sum():,.0f}")

    # ── Split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.12, random_state=42, stratify=y_train
    )
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    train_loader = make_loader(X_train, y_train, balance=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  shuffle=False)

    # ── Model ──────────────────────────────────────────────────
    model = BioactivityNet().to(DEVICE)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.BCELoss()

    # ── Training loop ──────────────────────────────────────────
    train_losses, val_accs, val_aucs = [], [], []
    best_auc = 0

    print("\nTraining...\n")
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, train_loader, optimizer, criterion)
        acc, auc, _, _ = evaluate(model, val_loader)
        scheduler.step(1 - auc)

        train_losses.append(loss)
        val_accs.append(acc)
        val_aucs.append(auc)

        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {loss:.4f} | Val Acc: {acc*100:.2f}% | Val AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "best_model.pt")

    # ── Final test evaluation ──────────────────────────────────
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load("best_model.pt"))
    test_acc, test_auc, labels, preds = evaluate(model, test_loader)

    print(f"\n{'='*45}")
    print(f"  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Test ROC-AUC  : {test_auc:.4f}")
    print(f"{'='*45}\n")
    print(classification_report(labels, preds, target_names=["Inactive", "Active"]))

    plot_history(train_losses, val_accs, val_aucs)
    print("\nDone. Model saved to best_model.pt")


if __name__ == "__main__":
    main()
