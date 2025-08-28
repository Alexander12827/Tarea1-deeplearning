# src/evaluate.py
from __future__ import annotations
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score
from model import MLP

def main() -> None:
    """Carga modelo y evalúa en test con matriz de confusión."""
    obj = torch.load("data/features.pt")
    Xte = torch.from_numpy(obj["X_test"]); yte = torch.from_numpy(obj["y_test"])
    label_map = obj["label_map"]
    id2lab = {v: k for k, v in label_map.items()}
    num_classes = len(id2lab)
    input_dim = Xte.shape[1]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim, hidden=256, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load("results/best_model.pt", map_location=DEVICE))
    model.eval()

    te = DataLoader(TensorDataset(Xte, yte), batch_size=128, shuffle=False)
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in te:
            xb = xb.to(DEVICE)
            pred = model(xb).argmax(dim=1).cpu()
            y_pred.extend(pred.tolist())
            y_true.extend(yb.tolist())

    acc = accuracy_score(y_true, y_pred)
    print(f"Test accuracy: {acc:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fig = plt.figure()
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=[id2lab[i] for i in range(num_classes)],
                yticklabels=[id2lab[i] for i in range(num_classes)])
    plt.title(f"Confusion Matrix (acc={acc:.3f})")
    plt.xlabel("Pred"); plt.ylabel("True")
    Path("results").mkdir(exist_ok=True)
    out = "results/confusion_matrix.png"
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"Matriz de confusión guardada en {out}")

if __name__ == "__main__":
    main()
