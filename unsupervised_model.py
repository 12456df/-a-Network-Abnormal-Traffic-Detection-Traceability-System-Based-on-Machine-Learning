"""
unsupervised_model.py
=====================

Unsupervised anomaly detection: Isolation Forest + Autoencoder.

Training paradigm: learn "normal" traffic patterns from BENIGN-only samples;
flag deviations as anomalies.  Outputs anomaly scores and threshold-based
predictions compatible with the supervised pipeline and traceback module.

Inputs  (from csv_preprocessing.py):
    processed/unsupervised_benign_train.csv   – BENIGN-only features for training
    processed/val_binary.csv                  – labelled val set for threshold tuning
    processed/test_binary.csv                 – labelled test set for evaluation
    processed/feature_names.txt               – ordered feature column names

Outputs:
    models/unsupervised_iforest.joblib
    models/unsupervised_autoencoder.pth
    models/unsupervised_scaler.joblib
    models/unsupervised_thresholds.json
    models/unsupervised_manifest.json
    processed/test_predictions_unsupervised.csv
    processed/metrics_unsupervised.json
    processed/ip_attack_summary_test_unsupervised.csv
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
from sklearn.ensemble import IsolationForest
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
# JSON helpers (shared pattern with supervised_model.py)
# ---------------------------------------------------------------------------


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    return json.dumps(_sanitize_for_json(obj), **kwargs)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class UnsupervisedConfig:
    processed_dir: Path = PROJECT_ROOT / "processed"
    model_dir: Path = PROJECT_ROOT / "models"
    binary_label_col: str = "binary_label"
    sample_id_col: str = "sample_id"
    random_state: int = 42

    # Isolation Forest
    if_n_estimators: int = 300
    if_max_samples: int | str = "auto"
    if_contamination: float | str = "auto"

    # Autoencoder
    ae_epochs: int = 80
    ae_batch_size: int = 2048
    ae_lr: float = 1e-3
    ae_weight_decay: float = 1e-5
    ae_patience: int = 10
    ae_dropout: float = 0.1
    ae_bottleneck_dim: int = 12

    # Threshold targets (mirror supervised_model.py)
    precision_target: float = 0.995
    recall_target: float = 0.995

    ip_attack_ratio_threshold: float = 0.5

    def __post_init__(self) -> None:
        for name in ("precision_target", "recall_target", "ip_attack_ratio_threshold"):
            val = getattr(self, name)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} 必须在 [0, 1] 范围内，当前值: {val}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _must_exist(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")


def read_feature_names(path: Path) -> list[str]:
    _must_exist(path)
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise RuntimeError(f"特征列表为空: {path}")
    return names


def load_benign_train(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "unsupervised_benign_train.csv"
    _must_exist(path)
    return pd.read_csv(path)


def load_split(processed_dir: Path, filename: str) -> pd.DataFrame:
    path = processed_dir / filename
    _must_exist(path)
    return pd.read_csv(path)


def extract_features(
    df: pd.DataFrame, feature_cols: list[str]
) -> np.ndarray:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"数据缺少特征列，示例: {missing[:5]}")
    return df[feature_cols].to_numpy(dtype=np.float64)


def clean_array(arr: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Replace inf / NaN with pre-computed column medians."""
    result = arr.copy()
    bad = ~np.isfinite(result)
    if bad.any():
        row_idx, col_idx = np.where(bad)
        result[row_idx, col_idx] = medians[col_idx]
    return result


# ---------------------------------------------------------------------------
# Metrics (shared pattern with supervised_model.py)
# ---------------------------------------------------------------------------


def metrics_dict(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None
) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        result["auc"] = float(roc_auc_score(y_true, y_score))
    else:
        result["auc"] = float("nan")
    return result


def print_metrics(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
) -> Dict[str, float]:
    result = metrics_dict(y_true, y_pred, y_score)
    print(f"\n[{name}]")
    print(
        "accuracy={accuracy:.4f} precision={precision:.4f} recall={recall:.4f} "
        "f1={f1:.4f} auc={auc:.4f} fpr={fpr:.4f}".format(**result)
    )
    print(
        "confusion_matrix [[tn, fp], [fn, tp]] =",
        [
            [int(result["tn"]), int(result["fp"])],
            [int(result["fn"]), int(result["tp"])],
        ],
    )
    print("classification_report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    return result


# ---------------------------------------------------------------------------
# Threshold helpers (same logic as supervised_model.py)
# ---------------------------------------------------------------------------


def find_high_precision_threshold(
    y_true: np.ndarray, y_score: np.ndarray, target: float
) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    valid = np.where(precisions[:-1] >= target)[0]
    if len(valid) == 0:
        best = int(np.nanargmax(precisions[:-1]))
        return float(thresholds[best])
    best = valid[int(np.nanargmax(recalls[valid]))]
    return float(thresholds[best])


def find_high_recall_threshold(
    y_true: np.ndarray, y_score: np.ndarray, target: float
) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    valid = np.where(recalls[:-1] >= target)[0]
    if len(valid) == 0:
        best = int(np.nanargmax(recalls[:-1]))
        return float(thresholds[best])
    best = valid[int(np.nanargmax(precisions[valid]))]
    return float(thresholds[best])


def find_default_threshold(
    y_score_benign: np.ndarray, percentile: float = 99.0
) -> float:
    """99th-percentile of benign scores → ~1 % FPR on normal traffic."""
    return float(np.percentile(y_score_benign, percentile))


def evaluate_threshold_bundle(
    split_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    all_metrics: Dict[str, Dict[str, float]] = {}
    for key, th in thresholds.items():
        y_pred = (y_score >= th).astype(int)
        m = print_metrics(
            f"{split_name}-{key}(th={th:.6f})", y_true, y_pred, y_score
        )
        all_metrics[key] = m
    return all_metrics


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------


def train_isolation_forest(
    X_train: np.ndarray, cfg: UnsupervisedConfig
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=cfg.if_n_estimators,
        max_samples=cfg.if_max_samples,
        contamination=cfg.if_contamination,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    print(
        f"  n_estimators={cfg.if_n_estimators}, "
        f"max_samples={cfg.if_max_samples}, "
        f"contamination={cfg.if_contamination}"
    )
    model.fit(X_train)
    return model


def score_isolation_forest(
    model: IsolationForest, X: np.ndarray
) -> np.ndarray:
    """Higher value → more anomalous (negate sklearn convention)."""
    return -model.decision_function(X)


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------


class Autoencoder(nn.Module):
    def __init__(
        self, input_dim: int, bottleneck_dim: int = 12, dropout: float = 0.1
    ):
        super().__init__()
        h1 = max(bottleneck_dim * 4, input_dim * 2 // 3)
        h2 = max(bottleneck_dim * 2, input_dim // 3)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _score_ae_batched(
    model: Autoencoder,
    X: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Per-sample MSE reconstruction error (higher → more anomalous)."""
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size].to(device)
            recon = model(batch)
            mse = ((recon - batch) ** 2).mean(dim=1)
            parts.append(mse.cpu().numpy())
    return np.concatenate(parts)


def train_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: UnsupervisedConfig,
    device: torch.device,
) -> Tuple[Autoencoder, List[Dict[str, float]]]:
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim, cfg.ae_bottleneck_dim, cfg.ae_dropout).to(device)
    print(f"  architecture: {input_dim} → encoder → {cfg.ae_bottleneck_dim} → decoder → {input_dim}")
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=cfg.ae_batch_size,
        shuffle=True,
        drop_last=False,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.ae_lr, weight_decay=cfg.ae_weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5, min_lr=1e-6
    )

    best_auc = -1.0
    patience_counter = 0
    best_state: dict | None = None
    history: List[Dict[str, float]] = []

    for epoch in range(1, cfg.ae_epochs + 1):
        # ---- train ----
        model.train()
        epoch_loss = 0.0
        n_samples = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = nn.functional.mse_loss(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
            n_samples += len(batch)
        epoch_loss /= max(n_samples, 1)

        # ---- validate (AUC as early-stopping criterion) ----
        val_scores = _score_ae_batched(model, val_tensor, device)
        val_auc = (
            float(roc_auc_score(y_val, val_scores))
            if len(np.unique(y_val)) > 1
            else 0.0
        )
        scheduler.step(val_auc)

        lr_now = optimizer.param_groups[0]["lr"]
        history.append(
            {"epoch": epoch, "train_loss": epoch_loss, "val_auc": val_auc, "lr": lr_now}
        )
        print(
            f"  [AE] epoch {epoch:3d}/{cfg.ae_epochs}  "
            f"train_loss={epoch_loss:.6f}  val_auc={val_auc:.4f}  lr={lr_now:.2e}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= cfg.ae_patience:
            print(
                f"  [AE] Early stopping at epoch {epoch}, best val_auc={best_auc:.4f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ---------------------------------------------------------------------------
# IP-level summary (mirrors supervised_model.save_ip_level_summary)
# ---------------------------------------------------------------------------


def save_ip_level_summary(
    cfg: UnsupervisedConfig, pred_df: pd.DataFrame
) -> None:
    trace_path = cfg.processed_dir / "trace_metadata.csv"
    if not trace_path.exists():
        print("[IP Summary] 未找到 trace_metadata.csv，跳过 IP 级汇总。")
        return

    trace_df = pd.read_csv(trace_path)
    required = {cfg.sample_id_col, "Source IP"}
    if not required.issubset(set(trace_df.columns)):
        print("[IP Summary] trace_metadata 缺少 sample_id / Source IP，跳过。")
        return

    test_trace = trace_df.copy()
    if "split_binary" in test_trace.columns:
        test_trace = test_trace[test_trace["split_binary"] == "test"]

    merged = test_trace.merge(pred_df, on=cfg.sample_id_col, how="inner")
    if merged.empty:
        print("[IP Summary] 测试集预测与 trace_metadata 未匹配到记录，跳过。")
        return

    ip_summary = (
        merged.groupby("Source IP", dropna=False)
        .agg(
            total_flows=("iforest_pred_default", "size"),
            iforest_attack_flows=("iforest_pred_default", "sum"),
            iforest_attack_ratio=("iforest_pred_default", "mean"),
            ae_attack_flows=("ae_pred_default", "sum"),
            ae_attack_ratio=("ae_pred_default", "mean"),
            avg_iforest_score=("iforest_score", "mean"),
            avg_ae_score=("ae_score", "mean"),
            true_attack_ratio=("y_true", "mean"),
        )
        .reset_index()
        .sort_values("iforest_attack_ratio", ascending=False)
    )

    out_path = cfg.processed_dir / "ip_attack_summary_test_unsupervised.csv"
    ip_summary.to_csv(out_path, index=False)
    print(f"\n[IP Summary] 已输出: {out_path}")
    print("[IP Summary] Top-10 可疑 IP (按 Isolation Forest 攻击流比例):")
    for _, row in ip_summary.head(10).iterrows():
        print(
            f"  - {row['Source IP']}: "
            f"if_ratio={float(row['iforest_attack_ratio']):.4f}, "
            f"ae_ratio={float(row['ae_attack_ratio']):.4f}, "
            f"flows={int(row['total_flows'])}"
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unsupervised anomaly detection: Isolation Forest + Autoencoder."
    )
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "processed")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--if-n-estimators", type=int, default=300)
    parser.add_argument("--ae-epochs", type=int, default=80)
    parser.add_argument("--ae-batch-size", type=int, default=2048)
    parser.add_argument("--ae-lr", type=float, default=1e-3)
    parser.add_argument("--ae-patience", type=int, default=10)
    parser.add_argument("--precision-target", type=float, default=0.995)
    parser.add_argument("--recall-target", type=float, default=0.995)
    parser.add_argument(
        "--device", type=str, default=None, help="cpu / cuda / cuda:0 …"
    )
    args = parser.parse_args()

    cfg = UnsupervisedConfig(
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
        if_n_estimators=args.if_n_estimators,
        ae_epochs=args.ae_epochs,
        ae_batch_size=args.ae_batch_size,
        ae_lr=args.ae_lr,
        ae_patience=args.ae_patience,
        precision_target=args.precision_target,
        recall_target=args.recall_target,
    )

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[unsupervised_model] PyTorch device: {device}")

    # ================================================================
    # 1. Load data
    # ================================================================
    feature_cols = read_feature_names(cfg.processed_dir / "feature_names.txt")

    print("[unsupervised_model] 加载 BENIGN 训练集 …")
    benign_df = load_benign_train(cfg.processed_dir)
    X_benign_raw = extract_features(benign_df, feature_cols)
    del benign_df

    print("[unsupervised_model] 加载验证集 …")
    val_df = load_split(cfg.processed_dir, "val_binary.csv")
    X_val_raw = extract_features(val_df, feature_cols)
    y_val = val_df[cfg.binary_label_col].to_numpy(dtype=int)

    print("[unsupervised_model] 加载测试集 …")
    test_df = load_split(cfg.processed_dir, "test_binary.csv")
    X_test_raw = extract_features(test_df, feature_cols)
    y_test = test_df[cfg.binary_label_col].to_numpy(dtype=int)
    sample_ids_test = (
        test_df[cfg.sample_id_col].to_numpy()
        if cfg.sample_id_col in test_df.columns
        else np.arange(len(test_df))
    )
    del val_df, test_df

    print(
        f"[unsupervised_model] 数据规模: benign_train={len(X_benign_raw)}, "
        f"val={len(X_val_raw)}, test={len(X_test_raw)}, features={len(feature_cols)}"
    )

    # ================================================================
    # 2. Clean inf / NaN → column medians from benign training set
    # ================================================================
    benign_medians = np.nanmedian(
        np.where(np.isfinite(X_benign_raw), X_benign_raw, np.nan), axis=0
    )
    X_benign = clean_array(X_benign_raw, benign_medians)
    X_val = clean_array(X_val_raw, benign_medians)
    X_test = clean_array(X_test_raw, benign_medians)
    del X_benign_raw, X_val_raw, X_test_raw

    # ================================================================
    # 3. StandardScaler (fit on benign train only)
    # ================================================================
    scaler = StandardScaler()
    X_benign = scaler.fit_transform(X_benign)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = cfg.model_dir / "unsupervised_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"[unsupervised_model] 已保存 scaler: {scaler_path}")

    # ================================================================
    # 4. Isolation Forest
    # ================================================================
    print("\n" + "=" * 60)
    print("[Isolation Forest] 开始训练 …")
    iforest = train_isolation_forest(X_benign, cfg)

    if_val_scores = score_isolation_forest(iforest, X_val)
    if_test_scores = score_isolation_forest(iforest, X_test)

    if_benign_val_scores = if_val_scores[y_val == 0]
    if_thresholds: Dict[str, float] = {
        "default": find_default_threshold(if_benign_val_scores),
        "high_precision": find_high_precision_threshold(
            y_val, if_val_scores, cfg.precision_target
        ),
        "high_recall": find_high_recall_threshold(
            y_val, if_val_scores, cfg.recall_target
        ),
    }
    print(
        f"\n[IF Thresholds] default={if_thresholds['default']:.6f}, "
        f"high_precision={if_thresholds['high_precision']:.6f}, "
        f"high_recall={if_thresholds['high_recall']:.6f}"
    )

    print("\n[IF Validation Metrics]")
    if_val_metrics = evaluate_threshold_bundle(
        "IF-Val", y_val, if_val_scores, if_thresholds
    )
    print("\n[IF Test Metrics]")
    if_test_metrics = evaluate_threshold_bundle(
        "IF-Test", y_test, if_test_scores, if_thresholds
    )

    iforest_path = cfg.model_dir / "unsupervised_iforest.joblib"
    joblib.dump(iforest, iforest_path)
    print(f"\n[Isolation Forest] 已保存模型: {iforest_path}")

    # ================================================================
    # 5. Autoencoder
    # ================================================================
    print("\n" + "=" * 60)
    print("[Autoencoder] 开始训练 …")
    ae_model, ae_history = train_autoencoder(X_benign, X_val, y_val, cfg, device)

    ae_val_scores = _score_ae_batched(
        ae_model, torch.tensor(X_val, dtype=torch.float32), device
    )
    ae_test_scores = _score_ae_batched(
        ae_model, torch.tensor(X_test, dtype=torch.float32), device
    )

    ae_benign_val_scores = ae_val_scores[y_val == 0]
    ae_thresholds: Dict[str, float] = {
        "default": find_default_threshold(ae_benign_val_scores),
        "high_precision": find_high_precision_threshold(
            y_val, ae_val_scores, cfg.precision_target
        ),
        "high_recall": find_high_recall_threshold(
            y_val, ae_val_scores, cfg.recall_target
        ),
    }
    print(
        f"\n[AE Thresholds] default={ae_thresholds['default']:.6f}, "
        f"high_precision={ae_thresholds['high_precision']:.6f}, "
        f"high_recall={ae_thresholds['high_recall']:.6f}"
    )

    print("\n[AE Validation Metrics]")
    ae_val_metrics = evaluate_threshold_bundle(
        "AE-Val", y_val, ae_val_scores, ae_thresholds
    )
    print("\n[AE Test Metrics]")
    ae_test_metrics = evaluate_threshold_bundle(
        "AE-Test", y_test, ae_test_scores, ae_thresholds
    )

    ae_model_path = cfg.model_dir / "unsupervised_autoencoder.pth"
    torch.save(
        {
            "model_state_dict": ae_model.state_dict(),
            "input_dim": len(feature_cols),
            "bottleneck_dim": cfg.ae_bottleneck_dim,
            "dropout": cfg.ae_dropout,
        },
        ae_model_path,
    )
    print(f"\n[Autoencoder] 已保存模型: {ae_model_path}")

    # ================================================================
    # 6. Save predictions & metrics
    # ================================================================
    pred_df = pd.DataFrame(
        {
            cfg.sample_id_col: sample_ids_test,
            "y_true": y_test,
            "iforest_score": if_test_scores,
            "ae_score": ae_test_scores,
            "iforest_pred_default": (if_test_scores >= if_thresholds["default"]).astype(int),
            "iforest_pred_hp": (if_test_scores >= if_thresholds["high_precision"]).astype(int),
            "iforest_pred_hr": (if_test_scores >= if_thresholds["high_recall"]).astype(int),
            "ae_pred_default": (ae_test_scores >= ae_thresholds["default"]).astype(int),
            "ae_pred_hp": (ae_test_scores >= ae_thresholds["high_precision"]).astype(int),
            "ae_pred_hr": (ae_test_scores >= ae_thresholds["high_recall"]).astype(int),
        }
    )
    pred_path = cfg.processed_dir / "test_predictions_unsupervised.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\n[unsupervised_model] 已保存测试集预测: {pred_path}")

    all_metrics = {
        "isolation_forest": {
            "thresholds": if_thresholds,
            "val": if_val_metrics,
            "test": if_test_metrics,
        },
        "autoencoder": {
            "thresholds": ae_thresholds,
            "val": ae_val_metrics,
            "test": ae_test_metrics,
            "training_history": ae_history,
        },
    }
    metrics_path = cfg.processed_dir / "metrics_unsupervised.json"
    metrics_path.write_text(
        safe_json_dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[unsupervised_model] 已保存指标报告: {metrics_path}")

    thresholds_combined = {
        "isolation_forest": if_thresholds,
        "autoencoder": ae_thresholds,
    }
    thresholds_path = cfg.model_dir / "unsupervised_thresholds.json"
    thresholds_path.write_text(
        safe_json_dumps(thresholds_combined, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[unsupervised_model] 已保存阈值配置: {thresholds_path}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "n_benign_train": int(X_benign.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "isolation_forest": {
            "n_estimators": cfg.if_n_estimators,
            "max_samples": str(cfg.if_max_samples),
            "contamination": str(cfg.if_contamination),
        },
        "autoencoder": {
            "epochs_run": len(ae_history),
            "bottleneck_dim": cfg.ae_bottleneck_dim,
            "dropout": cfg.ae_dropout,
            "lr": cfg.ae_lr,
            "weight_decay": cfg.ae_weight_decay,
            "batch_size": cfg.ae_batch_size,
            "patience": cfg.ae_patience,
            "best_val_auc": (
                max(h["val_auc"] for h in ae_history) if ae_history else None
            ),
        },
    }
    manifest_path = cfg.model_dir / "unsupervised_manifest.json"
    manifest_path.write_text(
        safe_json_dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[unsupervised_model] 已保存模型元信息: {manifest_path}")

    # ================================================================
    # 7. IP-level summary for traceback
    # ================================================================
    save_ip_level_summary(cfg, pred_df)

    print("\n[unsupervised_model] 训练与评估完成。")


if __name__ == "__main__":
    print("[unsupervised_model] 开始无监督异常检测训练 …")
    try:
        main()
    except Exception as exc:
        print(f"[unsupervised_model] 发生错误: {exc}")
        raise
    else:
        print("[unsupervised_model] 全部完成。")
