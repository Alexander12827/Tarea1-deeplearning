# src/preprocessing.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torchaudio as ta
import torchaudio.functional as F
import numpy as np

def list_wavs(root: Path) -> List[Path]:
    """Lista todos los .wav recursivamente."""
    return list(root.rglob("*.wav"))

def label_from_path(p: Path) -> str:
    """Deriva la etiqueta desde el nombre de la carpeta padre del wav."""
    return p.parent.name.lower()

def make_label_map(paths: List[Path]) -> Dict[str, int]:
    """Crea un mapeo etiqueta→id a partir de las carpetas presentes."""
    labels = sorted({label_from_path(p) for p in paths})
    return {lab: i for i, lab in enumerate(labels)}

def extract_logmel(x: torch.Tensor, sr: int, n_mels: int = 64) -> torch.Tensor:
    """Extrae log-Mel a partir de x: [T]."""
    n_fft = int(0.025 * sr)  # ~25 ms
    hop = int(0.010 * sr)    # ~10 ms
    mel = ta.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)(x)
    logmel = torch.log(mel + 1e-6)
    return logmel  # [n_mels, frames]

def time_pool_stats(feats: torch.Tensor) -> torch.Tensor:
    """Concatena mean y std a lo largo del tiempo → [2 * n_mels]."""
    mu = feats.mean(dim=-1)
    sd = feats.std(dim=-1, unbiased=False)
    return torch.cat([mu, sd], dim=0)

def featurize_file(wav_path: Path) -> Tuple[np.ndarray, str]:
    """Devuelve (feature_vector, label_str) para un wav."""
    x, sr = ta.load(str(wav_path))  # x: [C, T]
    x = x.mean(dim=0)               # mono
    logmel = extract_logmel(x, sr)  # [M, F]
    delta = F.compute_deltas(logmel)
    delta2 = F.compute_deltas(delta)
    feat = torch.cat([time_pool_stats(logmel),
                      time_pool_stats(delta),
                      time_pool_stats(delta2)], dim=0)  # [3 * 2 * M]
    return feat.numpy().astype(np.float32), label_from_path(wav_path)

def run_partition(part_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Procesa una partición (train/valid/test)."""
    wavs = list_wavs(part_dir)
    feats, labs = [], []
    for p in wavs:
        f, lab = featurize_file(p)
        feats.append(f)
        labs.append(lab)
    label_map = make_label_map(wavs)
    y = np.array([label_map[l] for l in labs], dtype=np.int64)
    X = np.vstack(feats)
    return X, y, label_map

def main() -> None:
    """Procesa data/{train,valid,test} y guarda features en data/features.pt."""
    root = Path("data")
    out = Path("data/features.pt")
    Xtr, ytr, map_tr = run_partition(root / "train")
    Xva, yva, _ = run_partition(root / "valid")
    Xte, yte, _ = run_partition(root / "test")
    torch.save({
        "X_train": Xtr, "y_train": ytr,
        "X_valid": Xva, "y_valid": yva,
        "X_test":  Xte, "y_test":  yte,
        "label_map": map_tr
    }, out)
    print(f"✔ Features guardadas en {out} | dims: {Xtr.shape} (train)")

if __name__ == "__main__":
    main()
