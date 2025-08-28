# src/train.py
from __future__ import annotations
import argparse, json
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from pathlib import Path
from model import MLP

def make_loaders(features_path: str, batch: int = 64) -> tuple[DataLoader, DataLoader]:
    """Arma DataLoaders de train/valid desde features.pt."""
    obj = torch.load(features_path)
    Xtr = torch.from_numpy(obj["X_train"]); ytr = torch.from_numpy(obj["y_train"])
    Xva = torch.from_numpy(obj["X_valid"]); yva = torch.from_numpy(obj["y_valid"])
    tr = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True, num_workers=0)
    va = DataLoader(TensorDataset(Xva, yva), batch_size=batch, shuffle=False, num_workers=0)
    return tr, va

def train(args: argparse.Namespace) -> None:
    """Entrena el MLP y guarda el mejor modelo."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = make_loaders(args.features, args.batch_size)
    input_dim = tr.dataset.tensors[0].shape[1]
    num_classes = int(torch.max(tr.dataset.tensors[1]).item() + 1)

    model = MLP(input_dim, args.hidden, num_classes).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit  = nn.CrossEntropyLoss()

    best_acc, patience, bad = 0.0, args.patience, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            optim.step()

        model.eval(); correct = total = 0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        acc = correct / total
        print(f"Epoch {epoch}: val acc = {acc:.4f}")

        if acc > best_acc:
            best_acc, bad = acc, 0
            Path("results").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "results/best_model.pt")
            with open("results/val_best.json", "w") as f:
                json.dump({"val_acc": best_acc, "epoch": epoch}, f, indent=2)
        else:
            bad += 1
            if bad >= patience:
                print("⏹ early stopping")
                break

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default="data/features.pt")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()
    train(args)
