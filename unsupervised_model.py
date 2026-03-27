"""
unsupervised_model.py
=====================

Unsupervised anomaly detection — improved version.

Models:  Isolation Forest  ·  Autoencoder (wider)  ·  VAE
Feature: MI feature selection  ·  PCA analysis
Eval:    F1-optimal threshold  ·  per-attack-type recall  ·  ensemble scoring

Inputs  (from csv_preprocessing.py → processed/):
    unsupervised_benign_train.csv, train_binary.csv (for MI),
    val_binary.csv, test_binary.csv, feature_names.txt

Outputs:
    models/  — iforest, ae, vae, scaler, thresholds, manifest
    processed/ — predictions, metrics, per-attack analysis, IP summary
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _sanitize(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _json(obj: Any, **kw: Any) -> str:
    return json.dumps(_sanitize(obj), **kw)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Cfg:
    processed_dir: Path = PROJECT_ROOT / "processed"
    model_dir: Path = PROJECT_ROOT / "models"
    binary_label_col: str = "binary_label"
    label_col: str = "Label"
    sample_id_col: str = "sample_id"
    random_state: int = 42

    # Feature selection
    mi_top_k: int = 50
    mi_subsample: int = 100_000
    pca_variance_ratio: float = 0.95

    # Isolation Forest
    if_n_estimators: int = 300
    if_max_samples: int | str = "auto"
    if_contamination: float | str = "auto"

    # Autoencoder (wider)
    ae_bottleneck_dim: int = 20
    ae_n_layers: int = 3
    ae_epochs: int = 80
    ae_batch_size: int = 2048
    ae_lr: float = 1e-3
    ae_weight_decay: float = 1e-5
    ae_patience: int = 10
    ae_dropout: float = 0.15

    # VAE
    vae_bottleneck_dim: int = 20
    vae_n_layers: int = 3
    vae_beta: float = 0.5
    vae_epochs: int = 80
    vae_lr: float = 1e-3
    vae_patience: int = 10

    # Thresholds
    precision_target: float = 0.995
    recall_target: float = 0.995

    ip_attack_ratio_threshold: float = 0.5


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _must(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")


def read_feature_names(path: Path) -> list[str]:
    _must(path)
    return [l.strip() for l in path.read_text("utf-8").splitlines() if l.strip()]


def load_csv(processed_dir: Path, name: str) -> pd.DataFrame:
    p = processed_dir / name
    _must(p)
    return pd.read_csv(p)


def extract_features(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"缺少特征列: {missing[:5]}")
    return df[cols].to_numpy(dtype=np.float64)


def clean_array(arr: np.ndarray, medians: np.ndarray) -> np.ndarray:
    out = arr.copy()
    bad = ~np.isfinite(out)
    if bad.any():
        r, c = np.where(bad)
        out[r, c] = medians[c]
    return out


# ---------------------------------------------------------------------------
# Feature selection: Mutual Information
# ---------------------------------------------------------------------------

def compute_mi_ranking(
    X: np.ndarray, y: np.ndarray, feature_names: list[str],
    subsample: int = 100_000, seed: int = 42,
) -> List[Tuple[str, float]]:
    if len(X) > subsample:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), subsample, replace=False)
        X, y = X[idx], y[idx]
    mi = mutual_info_classif(X, y, random_state=seed)
    return sorted(zip(feature_names, mi.tolist()), key=lambda t: -t[1])


def select_top_mi(
    ranking: List[Tuple[str, float]], all_features: list[str], k: int,
) -> Tuple[list[str], list[int]]:
    selected = [f for f, _ in ranking[:k]]
    indices = [all_features.index(f) for f in selected]
    return selected, indices


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    r: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        r["auc"] = float(roc_auc_score(y_true, y_score))
    else:
        r["auc"] = float("nan")
    return r


def print_metrics(tag: str, y_true: np.ndarray, y_pred: np.ndarray,
                  y_score: np.ndarray | None) -> Dict[str, Any]:
    r = _metrics(y_true, y_pred, y_score)
    print(
        f"  [{tag}] acc={r['accuracy']:.4f} prec={r['precision']:.4f} "
        f"rec={r['recall']:.4f} f1={r['f1']:.4f} auc={r['auc']:.4f} fpr={r['fpr']:.4f}"
    )
    return r


# ---------------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------------

def find_hp_threshold(y: np.ndarray, s: np.ndarray, target: float) -> float:
    pr, rc, th = precision_recall_curve(y, s)
    v = np.where(pr[:-1] >= target)[0]
    if len(v) == 0:
        return float(th[int(np.nanargmax(pr[:-1]))])
    return float(th[v[int(np.nanargmax(rc[v]))]])


def find_hr_threshold(y: np.ndarray, s: np.ndarray, target: float) -> float:
    pr, rc, th = precision_recall_curve(y, s)
    v = np.where(rc[:-1] >= target)[0]
    if len(v) == 0:
        return float(th[int(np.nanargmax(rc[:-1]))])
    return float(th[v[int(np.nanargmax(pr[v]))]])


def find_best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    pr, rc, th = precision_recall_curve(y, s)
    f1_arr = np.where((pr[:-1] + rc[:-1]) > 0,
                       2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1]), 0.0)
    return float(th[int(np.nanargmax(f1_arr))])


def find_default_threshold(benign_scores: np.ndarray, pct: float = 99.0) -> float:
    return float(np.percentile(benign_scores, pct))


def build_thresholds(y_val: np.ndarray, val_scores: np.ndarray,
                     cfg: Cfg) -> Dict[str, float]:
    benign = val_scores[y_val == 0]
    return {
        "default": find_default_threshold(benign),
        "high_precision": find_hp_threshold(y_val, val_scores, cfg.precision_target),
        "high_recall": find_hr_threshold(y_val, val_scores, cfg.recall_target),
        "best_f1": find_best_f1_threshold(y_val, val_scores),
    }


def evaluate_all_thresholds(
    tag: str, y: np.ndarray, scores: np.ndarray, thresholds: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key, th in thresholds.items():
        pred = (scores >= th).astype(int)
        out[key] = print_metrics(f"{tag}/{key}(th={th:.6f})", y, pred, scores)
    return out


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------

def train_iforest(X: np.ndarray, cfg: Cfg) -> IsolationForest:
    m = IsolationForest(
        n_estimators=cfg.if_n_estimators,
        max_samples=cfg.if_max_samples,
        contamination=cfg.if_contamination,
        random_state=cfg.random_state, n_jobs=-1,
    )
    m.fit(X)
    return m


def score_iforest(m: IsolationForest, X: np.ndarray) -> np.ndarray:
    return -m.decision_function(X)


# ---------------------------------------------------------------------------
# Autoencoder (flexible depth / width)
# ---------------------------------------------------------------------------

def _make_dims(input_dim: int, bottleneck: int, n_layers: int) -> list[int]:
    return np.linspace(input_dim, bottleneck, n_layers + 1).astype(int).tolist()


def _build_layers(dims: list[int], dropout: float, skip_last_act: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = i == len(dims) - 2
        if not (skip_last_act and is_last):
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, bottleneck: int, n_layers: int = 3,
                 dropout: float = 0.15):
        super().__init__()
        dims = _make_dims(input_dim, bottleneck, n_layers)
        self.encoder = _build_layers(dims, dropout, skip_last_act=True)
        self.decoder = _build_layers(list(reversed(dims)), dropout, skip_last_act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

class VAE(nn.Module):
    def __init__(self, input_dim: int, bottleneck: int, n_layers: int = 3,
                 dropout: float = 0.15):
        super().__init__()
        dims = _make_dims(input_dim, bottleneck, n_layers)
        self.encoder_shared = _build_layers(dims[:-1], dropout, skip_last_act=False)
        pre = dims[-2]
        self.fc_mu = nn.Linear(pre, bottleneck)
        self.fc_logvar = nn.Linear(pre, bottleneck)
        self.decoder = _build_layers(list(reversed(dims)), dropout, skip_last_act=True)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_shared(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return mu

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, lv = self.encode(x)
        z = self.reparameterize(mu, lv)
        return self.decoder(z), mu, lv


# ---------------------------------------------------------------------------
# Unified training & scoring
# ---------------------------------------------------------------------------

def _score_batched(model: nn.Module, X: torch.Tensor, device: torch.device,
                   is_vae: bool, bs: int = 4096) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            b = X[i:i + bs].to(device)
            if is_vae:
                recon, _, _ = model(b)
            else:
                recon = model(b)
            parts.append(((recon - b) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(parts)


def train_reconstruction_model(
    model: nn.Module,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Cfg,
    device: torch.device,
    is_vae: bool = False,
    label: str = "AE",
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [{label}] parameters: {n_params:,}")

    epochs = cfg.vae_epochs if is_vae else cfg.ae_epochs
    lr = cfg.vae_lr if is_vae else cfg.ae_lr
    patience = cfg.vae_patience if is_vae else cfg.ae_patience

    train_t = torch.tensor(X_train, dtype=torch.float32)
    val_t = torch.tensor(X_val, dtype=torch.float32)
    loader = DataLoader(TensorDataset(train_t), batch_size=cfg.ae_batch_size,
                        shuffle=True, drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.ae_weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=5, factor=0.5, min_lr=1e-6)

    best_auc, wait, best_state = -1.0, 0, None
    history: List[Dict[str, float]] = []

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device)
            if is_vae:
                recon, mu, lv = model(batch)
                recon_loss = nn.functional.mse_loss(recon, batch)
                kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
                loss = recon_loss + cfg.vae_beta * kl
            else:
                recon = model(batch)
                loss = nn.functional.mse_loss(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(batch)
            n += len(batch)
        avg_loss = total_loss / max(n, 1)

        val_sc = _score_batched(model, val_t, device, is_vae)
        vauc = float(roc_auc_score(y_val, val_sc)) if len(np.unique(y_val)) > 1 else 0.0
        sched.step(vauc)
        cur_lr = opt.param_groups[0]["lr"]
        history.append({"epoch": ep, "loss": avg_loss, "val_auc": vauc, "lr": cur_lr})
        print(f"  [{label}] ep {ep:3d}/{epochs}  loss={avg_loss:.6f}  "
              f"val_auc={vauc:.4f}  lr={cur_lr:.2e}")

        if vauc > best_auc:
            best_auc, wait = vauc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if wait >= patience:
            print(f"  [{label}] Early stop ep {ep}, best_auc={best_auc:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ---------------------------------------------------------------------------
# Ensemble scoring
# ---------------------------------------------------------------------------

def _minmax(scores: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    ref = ref if ref is not None else scores
    lo, hi = float(ref.min()), float(ref.max())
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def search_alpha_2(s1_val: np.ndarray, s2_val: np.ndarray,
                   y: np.ndarray, steps: int = 21) -> Tuple[float, float]:
    n1, n2 = _minmax(s1_val), _minmax(s2_val)
    best_a, best_auc = 0.5, -1.0
    for a in np.linspace(0, 1, steps):
        auc = roc_auc_score(y, a * n1 + (1 - a) * n2)
        if auc > best_auc:
            best_a, best_auc = float(a), auc
    return best_a, best_auc


def search_weights_3(
    s1: np.ndarray, s2: np.ndarray, s3: np.ndarray,
    y: np.ndarray, steps: int = 11,
) -> Tuple[Tuple[float, float, float], float]:
    n1, n2, n3 = _minmax(s1), _minmax(s2), _minmax(s3)
    best_w, best_auc = (1 / 3, 1 / 3, 1 / 3), -1.0
    grid = np.linspace(0, 1, steps)
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9:
                continue
            w3 = max(w3, 0.0)
            auc = roc_auc_score(y, w1 * n1 + w2 * n2 + w3 * n3)
            if auc > best_auc:
                best_w, best_auc = (float(w1), float(w2), float(w3)), auc
    return best_w, best_auc


def apply_ensemble_2(s1: np.ndarray, s2: np.ndarray, alpha: float,
                     ref1: np.ndarray, ref2: np.ndarray) -> np.ndarray:
    return alpha * _minmax(s1, ref1) + (1 - alpha) * _minmax(s2, ref2)


def apply_ensemble_3(s1: np.ndarray, s2: np.ndarray, s3: np.ndarray,
                     w: Tuple[float, float, float],
                     r1: np.ndarray, r2: np.ndarray, r3: np.ndarray) -> np.ndarray:
    return w[0] * _minmax(s1, r1) + w[1] * _minmax(s2, r2) + w[2] * _minmax(s3, r3)


# ---------------------------------------------------------------------------
# Per-attack-type analysis
# ---------------------------------------------------------------------------

def per_attack_analysis(
    scores: np.ndarray, threshold: float, labels: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    preds = (scores >= threshold).astype(int)
    result: Dict[str, Dict[str, Any]] = {}
    for lab in sorted(set(labels)):
        mask = labels == lab
        total = int(mask.sum())
        detected = int(preds[mask].sum())
        result[lab] = {
            "total": total,
            "detected": detected,
            "recall": round(detected / total, 4) if total > 0 else 0.0,
        }
    return result


# ---------------------------------------------------------------------------
# IP-level summary
# ---------------------------------------------------------------------------

def save_ip_summary(cfg: Cfg, pred_df: pd.DataFrame) -> None:
    tp = cfg.processed_dir / "trace_metadata.csv"
    if not tp.exists():
        print("[IP Summary] trace_metadata.csv 不存在，跳过。")
        return
    trace = pd.read_csv(tp)
    if not {cfg.sample_id_col, "Source IP"}.issubset(trace.columns):
        print("[IP Summary] 缺少必要列，跳过。")
        return
    if "split_binary" in trace.columns:
        trace = trace[trace["split_binary"] == "test"]
    merged = trace.merge(pred_df, on=cfg.sample_id_col, how="inner")
    if merged.empty:
        print("[IP Summary] 无匹配记录，跳过。")
        return

    score_col = "best_ensemble_score" if "best_ensemble_score" in merged.columns else "ae_score"
    pred_col = "best_ensemble_pred_best_f1" if "best_ensemble_pred_best_f1" in merged.columns else "ae_pred_best_f1"

    agg = (merged.groupby("Source IP", dropna=False)
           .agg(total_flows=(pred_col, "size"),
                attack_flows=(pred_col, "sum"),
                attack_ratio=(pred_col, "mean"),
                avg_score=(score_col, "mean"),
                true_attack_ratio=("y_true", "mean"))
           .reset_index()
           .sort_values("attack_ratio", ascending=False))
    out = cfg.processed_dir / "ip_attack_summary_test_unsupervised.csv"
    agg.to_csv(out, index=False)
    print(f"[IP Summary] 已保存: {out}")
    for _, row in agg.head(10).iterrows():
        print(f"  {row['Source IP']}: ratio={row['attack_ratio']:.4f}, flows={int(row['total_flows'])}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    pa = argparse.ArgumentParser(description="Unsupervised anomaly detection (improved)")
    pa.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "processed")
    pa.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models")
    pa.add_argument("--mi-top-k", type=int, default=50)
    pa.add_argument("--ae-bottleneck", type=int, default=20)
    pa.add_argument("--vae-bottleneck", type=int, default=20)
    pa.add_argument("--vae-beta", type=float, default=0.5)
    pa.add_argument("--ae-epochs", type=int, default=80)
    pa.add_argument("--vae-epochs", type=int, default=80)
    pa.add_argument("--if-n-estimators", type=int, default=300)
    pa.add_argument("--device", type=str, default=None)
    args = pa.parse_args()

    cfg = Cfg(
        processed_dir=args.processed_dir, model_dir=args.model_dir,
        mi_top_k=args.mi_top_k,
        ae_bottleneck_dim=args.ae_bottleneck, vae_bottleneck_dim=args.vae_bottleneck,
        vae_beta=args.vae_beta, ae_epochs=args.ae_epochs, vae_epochs=args.vae_epochs,
        if_n_estimators=args.if_n_estimators,
    )
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[config] device={device}, mi_top_k={cfg.mi_top_k}, "
          f"ae_bottleneck={cfg.ae_bottleneck_dim}, vae_bottleneck={cfg.vae_bottleneck_dim}")

    # ==================================================================
    # 1  LOAD DATA
    # ==================================================================
    feature_cols = read_feature_names(cfg.processed_dir / "feature_names.txt")

    print("\n[data] 加载 benign 训练集 …")
    benign_df = load_csv(cfg.processed_dir, "unsupervised_benign_train.csv")

    print("[data] 加载验证集 …")
    val_df = load_csv(cfg.processed_dir, "val_binary.csv")

    print("[data] 加载测试集 …")
    test_df = load_csv(cfg.processed_dir, "test_binary.csv")

    X_benign_raw = extract_features(benign_df, feature_cols)
    X_val_raw = extract_features(val_df, feature_cols)
    X_test_raw = extract_features(test_df, feature_cols)
    y_val = val_df[cfg.binary_label_col].to_numpy(dtype=int)
    y_test = test_df[cfg.binary_label_col].to_numpy(dtype=int)
    sample_ids = (test_df[cfg.sample_id_col].to_numpy()
                  if cfg.sample_id_col in test_df.columns
                  else np.arange(len(test_df)))
    attack_labels = (test_df[cfg.label_col].astype(str).str.strip().to_numpy()
                     if cfg.label_col in test_df.columns else None)
    del benign_df, val_df, test_df

    print(f"[data] benign={len(X_benign_raw)}, val={len(X_val_raw)}, "
          f"test={len(X_test_raw)}, features={len(feature_cols)}")

    # ==================================================================
    # 2  MI FEATURE SELECTION  (Step 3)
    # ==================================================================
    active_features = feature_cols
    mi_ranking: list | None = None

    if 0 < cfg.mi_top_k < len(feature_cols):
        print(f"\n[MI] 加载训练集计算互信息 (subsample={cfg.mi_subsample}) …")
        train_df = load_csv(cfg.processed_dir, "train_binary.csv")
        X_train_mi = extract_features(train_df, feature_cols)
        y_train_mi = train_df[cfg.binary_label_col].to_numpy(dtype=int)
        del train_df

        mi_ranking = compute_mi_ranking(
            X_train_mi, y_train_mi, feature_cols,
            subsample=cfg.mi_subsample, seed=cfg.random_state)
        del X_train_mi, y_train_mi

        selected, sel_idx = select_top_mi(mi_ranking, feature_cols, cfg.mi_top_k)
        print(f"[MI] 已选择 top-{cfg.mi_top_k} 特征。前 10:")
        for f, s in mi_ranking[:10]:
            print(f"    {f}: {s:.4f}")
        print(f"  被排除的特征 ({len(feature_cols) - cfg.mi_top_k}):")
        for f, s in mi_ranking[cfg.mi_top_k:]:
            print(f"    {f}: {s:.4f}")

        X_benign_raw = X_benign_raw[:, sel_idx]
        X_val_raw = X_val_raw[:, sel_idx]
        X_test_raw = X_test_raw[:, sel_idx]
        active_features = selected

    # ==================================================================
    # 3  CLEAN → SCALE → PCA ANALYSIS  (Step 3)
    # ==================================================================
    medians = np.nanmedian(np.where(np.isfinite(X_benign_raw), X_benign_raw, np.nan), axis=0)
    X_benign = clean_array(X_benign_raw, medians)
    X_val = clean_array(X_val_raw, medians)
    X_test = clean_array(X_test_raw, medians)
    del X_benign_raw, X_val_raw, X_test_raw

    scaler = StandardScaler()
    X_benign = scaler.fit_transform(X_benign)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, cfg.model_dir / "unsupervised_scaler.joblib")

    print(f"\n[PCA] 分析 (variance={cfg.pca_variance_ratio}) …")
    pca = PCA(n_components=cfg.pca_variance_ratio, svd_solver="full")
    pca.fit(X_benign)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    pca_info = {
        "n_components_95": int(pca.n_components_),
        "input_features": len(active_features),
        "top_10_explained_variance": pca.explained_variance_ratio_[:10].tolist(),
        "cumulative_variance": cum_var.tolist(),
    }
    print(f"[PCA] {pca.n_components_} 个主成分解释 {cum_var[-1]*100:.1f}% 方差 "
          f"(输入 {len(active_features)} 特征)")

    input_dim = X_benign.shape[1]

    # ==================================================================
    # 4  ISOLATION FOREST
    # ==================================================================
    print("\n" + "=" * 60)
    print("[IF] 训练 Isolation Forest …")
    iforest = train_iforest(X_benign, cfg)
    if_val = score_iforest(iforest, X_val)
    if_test = score_iforest(iforest, X_test)
    joblib.dump(iforest, cfg.model_dir / "unsupervised_iforest.joblib")

    if_th = build_thresholds(y_val, if_val, cfg)
    print("[IF] 阈值:", {k: f"{v:.6f}" for k, v in if_th.items()})
    print("[IF Val]")
    if_val_m = evaluate_all_thresholds("IF-Val", y_val, if_val, if_th)
    print("[IF Test]")
    if_test_m = evaluate_all_thresholds("IF-Test", y_test, if_test, if_th)

    # ==================================================================
    # 5  AUTOENCODER (wider)   (Step 4)
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"[AE] 训练 Autoencoder (bottleneck={cfg.ae_bottleneck_dim}) …")
    ae = Autoencoder(input_dim, cfg.ae_bottleneck_dim, cfg.ae_n_layers, cfg.ae_dropout)
    ae, ae_hist = train_reconstruction_model(
        ae, X_benign, X_val, y_val, cfg, device, is_vae=False, label="AE")
    ae_val = _score_batched(ae, torch.tensor(X_val, dtype=torch.float32), device, False)
    ae_test = _score_batched(ae, torch.tensor(X_test, dtype=torch.float32), device, False)
    torch.save({"state": ae.state_dict(), "input_dim": input_dim,
                "bottleneck": cfg.ae_bottleneck_dim, "n_layers": cfg.ae_n_layers,
                "dropout": cfg.ae_dropout}, cfg.model_dir / "unsupervised_autoencoder.pth")

    ae_th = build_thresholds(y_val, ae_val, cfg)
    print("[AE] 阈值:", {k: f"{v:.6f}" for k, v in ae_th.items()})
    print("[AE Val]")
    ae_val_m = evaluate_all_thresholds("AE-Val", y_val, ae_val, ae_th)
    print("[AE Test]")
    ae_test_m = evaluate_all_thresholds("AE-Test", y_test, ae_test, ae_th)

    # ==================================================================
    # 6  VAE  (Step 4)
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"[VAE] 训练 VAE (bottleneck={cfg.vae_bottleneck_dim}, β={cfg.vae_beta}) …")
    vae = VAE(input_dim, cfg.vae_bottleneck_dim, cfg.vae_n_layers, cfg.ae_dropout)
    vae, vae_hist = train_reconstruction_model(
        vae, X_benign, X_val, y_val, cfg, device, is_vae=True, label="VAE")
    vae_val = _score_batched(vae, torch.tensor(X_val, dtype=torch.float32), device, True)
    vae_test = _score_batched(vae, torch.tensor(X_test, dtype=torch.float32), device, True)
    torch.save({"state": vae.state_dict(), "input_dim": input_dim,
                "bottleneck": cfg.vae_bottleneck_dim, "n_layers": cfg.vae_n_layers,
                "dropout": cfg.ae_dropout, "beta": cfg.vae_beta},
               cfg.model_dir / "unsupervised_vae.pth")

    vae_th = build_thresholds(y_val, vae_val, cfg)
    print("[VAE] 阈值:", {k: f"{v:.6f}" for k, v in vae_th.items()})
    print("[VAE Val]")
    vae_val_m = evaluate_all_thresholds("VAE-Val", y_val, vae_val, vae_th)
    print("[VAE Test]")
    vae_test_m = evaluate_all_thresholds("VAE-Test", y_test, vae_test, vae_th)

    # ==================================================================
    # 7  ENSEMBLE  (Step 2)
    # ==================================================================
    print("\n" + "=" * 60)
    print("[Ensemble] 搜索最优融合权重 …")

    alpha_if_ae, auc_if_ae = search_alpha_2(if_val, ae_val, y_val)
    print(f"  IF+AE:  α={alpha_if_ae:.2f}, val_auc={auc_if_ae:.4f}")

    alpha_if_vae, auc_if_vae = search_alpha_2(if_val, vae_val, y_val)
    print(f"  IF+VAE: α={alpha_if_vae:.2f}, val_auc={auc_if_vae:.4f}")

    alpha_ae_vae, auc_ae_vae = search_alpha_2(ae_val, vae_val, y_val)
    print(f"  AE+VAE: α={alpha_ae_vae:.2f}, val_auc={auc_ae_vae:.4f}")

    w3, auc_3 = search_weights_3(if_val, ae_val, vae_val, y_val)
    print(f"  IF+AE+VAE: w=({w3[0]:.2f},{w3[1]:.2f},{w3[2]:.2f}), val_auc={auc_3:.4f}")

    ensemble_configs = [
        ("IF+AE", auc_if_ae, lambda v, t: (
            apply_ensemble_2(v[0], v[1], alpha_if_ae, if_val, ae_val),
            apply_ensemble_2(t[0], t[1], alpha_if_ae, if_val, ae_val))),
        ("IF+VAE", auc_if_vae, lambda v, t: (
            apply_ensemble_2(v[0], v[2], alpha_if_vae, if_val, vae_val),
            apply_ensemble_2(t[0], t[2], alpha_if_vae, if_val, vae_val))),
        ("AE+VAE", auc_ae_vae, lambda v, t: (
            apply_ensemble_2(v[1], v[2], alpha_ae_vae, ae_val, vae_val),
            apply_ensemble_2(t[1], t[2], alpha_ae_vae, ae_val, vae_val))),
        ("IF+AE+VAE", auc_3, lambda v, t: (
            apply_ensemble_3(v[0], v[1], v[2], w3, if_val, ae_val, vae_val),
            apply_ensemble_3(t[0], t[1], t[2], w3, if_val, ae_val, vae_val))),
    ]

    val_tuple = (if_val, ae_val, vae_val)
    test_tuple = (if_test, ae_test, vae_test)

    ens_results: Dict[str, Any] = {}
    best_ens_name, best_ens_auc = "", -1.0
    best_ens_test_scores: np.ndarray | None = None
    best_ens_th: Dict[str, float] = {}
    best_ens_test_m: Dict[str, Any] = {}

    for ens_name, val_auc, fn in ensemble_configs:
        ens_v, ens_t = fn(val_tuple, test_tuple)
        eth = build_thresholds(y_val, ens_v, cfg)
        print(f"\n[{ens_name}] val_auc={val_auc:.4f}, 阈值: "
              + ", ".join(f"{k}={v:.4f}" for k, v in eth.items()))
        print(f"[{ens_name} Val]")
        evm = evaluate_all_thresholds(f"{ens_name}-Val", y_val, ens_v, eth)
        print(f"[{ens_name} Test]")
        etm = evaluate_all_thresholds(f"{ens_name}-Test", y_test, ens_t, eth)
        ens_results[ens_name] = {"val_auc": val_auc, "thresholds": eth,
                                  "val": evm, "test": etm}
        if val_auc > best_ens_auc:
            best_ens_name, best_ens_auc = ens_name, val_auc
            best_ens_test_scores = ens_t
            best_ens_th = eth
            best_ens_test_m = etm

    print(f"\n[Ensemble] 最佳融合: {best_ens_name} (val_auc={best_ens_auc:.4f})")

    # ==================================================================
    # 8  PER-ATTACK-TYPE ANALYSIS  (Step 1)
    # ==================================================================
    attack_analysis: Dict[str, Any] = {}
    if attack_labels is not None:
        print("\n" + "=" * 60)
        print("[Per-Attack] 按攻击类型检出率 (使用 best_f1 阈值)")

        models_for_analysis = [
            ("IF", if_test, if_th),
            ("AE", ae_test, ae_th),
            ("VAE", vae_test, vae_th),
        ]
        if best_ens_test_scores is not None:
            models_for_analysis.append((f"Ens({best_ens_name})", best_ens_test_scores, best_ens_th))

        for mname, mscores, mth in models_for_analysis:
            pa_result = per_attack_analysis(mscores, mth["best_f1"], attack_labels)
            attack_analysis[mname] = pa_result
            print(f"\n  [{mname}] (th={mth['best_f1']:.6f})")
            for lab, info in pa_result.items():
                tag = "[Y]" if info["recall"] >= 0.8 else "[N]"
                print(f"    {tag} {lab}: {info['detected']}/{info['total']} "
                      f"(recall={info['recall']:.4f})")

    # ==================================================================
    # 9  SAVE ALL OUTPUTS
    # ==================================================================
    print("\n" + "=" * 60)
    print("[Save] 保存结果 …")

    # Predictions
    pred_data: Dict[str, Any] = {
        cfg.sample_id_col: sample_ids,
        "y_true": y_test,
        "iforest_score": if_test,
        "ae_score": ae_test,
        "vae_score": vae_test,
    }
    for mname, mscores, mth in [("iforest", if_test, if_th),
                                  ("ae", ae_test, ae_th),
                                  ("vae", vae_test, vae_th)]:
        for tkey in ("default", "high_precision", "high_recall", "best_f1"):
            pred_data[f"{mname}_pred_{tkey}"] = (mscores >= mth[tkey]).astype(int)

    if best_ens_test_scores is not None:
        pred_data["best_ensemble_score"] = best_ens_test_scores
        for tkey in ("default", "high_precision", "high_recall", "best_f1"):
            pred_data[f"best_ensemble_pred_{tkey}"] = (
                best_ens_test_scores >= best_ens_th[tkey]).astype(int)

    pred_df = pd.DataFrame(pred_data)
    pred_path = cfg.processed_dir / "test_predictions_unsupervised.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"  预测: {pred_path}")

    # Metrics JSON
    all_metrics: Dict[str, Any] = {
        "feature_selection": {
            "method": "mutual_information" if mi_ranking else "all_features",
            "n_selected": len(active_features),
            "n_total": len(feature_cols),
            "mi_ranking": [(f, round(s, 6)) for f, s in mi_ranking] if mi_ranking else None,
        },
        "pca_analysis": pca_info,
        "isolation_forest": {"thresholds": if_th, "val": if_val_m, "test": if_test_m},
        "autoencoder": {"thresholds": ae_th, "val": ae_val_m, "test": ae_test_m,
                        "history": ae_hist},
        "vae": {"thresholds": vae_th, "val": vae_val_m, "test": vae_test_m,
                "history": vae_hist},
        "ensembles": ens_results,
        "best_ensemble": best_ens_name,
        "per_attack_analysis": attack_analysis,
    }
    mp = cfg.processed_dir / "metrics_unsupervised.json"
    mp.write_text(_json(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  指标: {mp}")

    # Thresholds
    tp = cfg.model_dir / "unsupervised_thresholds.json"
    tp.write_text(_json({
        "isolation_forest": if_th, "autoencoder": ae_th, "vae": vae_th,
        "best_ensemble": {"name": best_ens_name, **best_ens_th},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  阈值: {tp}")

    # Selected features
    sf = cfg.model_dir / "unsupervised_selected_features.txt"
    sf.write_text("\n".join(active_features), encoding="utf-8")
    print(f"  特征: {sf}")

    # Manifest
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "active_features": active_features,
        "n_features": len(active_features),
        "n_benign_train": int(X_benign.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "isolation_forest": {"n_estimators": cfg.if_n_estimators},
        "autoencoder": {
            "bottleneck": cfg.ae_bottleneck_dim, "n_layers": cfg.ae_n_layers,
            "epochs_run": len(ae_hist), "dropout": cfg.ae_dropout,
            "best_val_auc": max(h["val_auc"] for h in ae_hist) if ae_hist else None,
        },
        "vae": {
            "bottleneck": cfg.vae_bottleneck_dim, "n_layers": cfg.vae_n_layers,
            "beta": cfg.vae_beta, "epochs_run": len(vae_hist),
            "best_val_auc": max(h["val_auc"] for h in vae_hist) if vae_hist else None,
        },
        "best_ensemble": best_ens_name,
        "best_ensemble_val_auc": best_ens_auc,
    }
    mnp = cfg.model_dir / "unsupervised_manifest.json"
    mnp.write_text(_json(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  元信息: {mnp}")

    # IP summary
    save_ip_summary(cfg, pred_df)

    # ==================================================================
    # 10  SUMMARY
    # ==================================================================
    print("\n" + "=" * 60)
    print("[Summary] 测试集 best_f1 阈值下各模型 F1:")
    for mname, mm in [("IF", if_test_m), ("AE", ae_test_m),
                       ("VAE", vae_test_m), (f"Ens({best_ens_name})", best_ens_test_m)]:
        bf = mm.get("best_f1", {})
        print(f"  {mname:20s}  F1={bf.get('f1',0):.4f}  "
              f"Recall={bf.get('recall',0):.4f}  Prec={bf.get('precision',0):.4f}  "
              f"AUC={bf.get('auc',0):.4f}")

    print("\n[unsupervised_model] 全部完成。")


if __name__ == "__main__":
    print("[unsupervised_model] 开始训练（改进版）…")
    try:
        main()
    except Exception as e:
        print(f"[unsupervised_model] 错误: {e}")
        raise
