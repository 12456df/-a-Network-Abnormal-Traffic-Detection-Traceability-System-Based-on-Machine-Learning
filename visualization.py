"""
visualization.py
================

从 ensemble_detector 产物生成论文 / 开题用检测侧插图（Matplotlib + Seaborn）。

依赖文件（缺失时默认跳过并警告，可用 --strict 遇错退出）:
    processed/metrics_ensemble.json     — 指标柱状图、告警分布、混淆矩阵（由 tn/fp/fn/tp）
    processed/test_predictions_ensemble.csv — ROC/PR、校准、阈值曲线、分数直方图、混淆（CSV）
    processed/train_binary.csv / val_binary.csv / test_binary.csv — 各 split 标签分布
    processed/feature_names.txt + models/supervised_rf_binary.joblib — 特征重要性

未来契约（只读，文件存在才绘图）:
    processed/trace_eval.json — 推荐顶层为标量数值，如
        {"top1_accuracy": 0.85, "top3_accuracy": 0.72, "path_f1": 0.61}
        或 {"metrics": {"name": value, ...}}
    processed/latency.csv — 列 latency_ms（float），或单列数值 CSV

溯源路径图见 attack_traceback.py 与 traceback/attack_graph.html，本脚本不重复。
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent

# --all 包含的任务（不含 pca，避免大 CSV 上默认过慢）
CORE_TASKS: Tuple[str, ...] = (
    "confusion",
    "roc",
    "pr",
    "metrics",
    "calibration",
    "threshold",
    "alerts",
    "hist",
    "label_dist",
    "importance",
    "trace_eval",
    "latency",
)

EPILOG = """
各任务与输入依赖:
  confusion      metrics_ensemble.json 和/或 test_predictions_ensemble.csv
  roc, pr        test_predictions_ensemble.csv
  metrics        metrics_ensemble.json（test 节）
  calibration    test_predictions_ensemble.csv
  threshold      test_predictions_ensemble.csv
  alerts         metrics_ensemble.json
  hist           test_predictions_ensemble.csv
  label_dist     train/val/test_binary.csv
  importance     feature_names.txt + supervised_rf_binary.joblib
  trace_eval     trace_eval.json（可选）
  latency        latency.csv（可选）
  pca            test_binary.csv + feature_names.txt（显式 --only pca，会抽样）

与溯源衔接: 检测图由本脚本输出 → 告警分级见 alerts/hist → 路径见 traceback/attack_graph.html 与 attack_timeline.png。
"""


def _apply_style(dpi: int, figsize: Tuple[float, float]) -> None:
    """Use default Latin fonts only so figures render without CJK font issues."""
    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.05)
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi
    plt.rcParams["figure.figsize"] = figsize


def _warn(msg: str) -> None:
    warnings.warn(msg, UserWarning, stacklevel=2)


def _need(path: Path, task: str, strict: bool) -> bool:
    if path.exists():
        return True
    msg = f"[{task}] missing file, skip: {path}"
    if strict:
        raise FileNotFoundError(msg)
    print(f"WARNING: {msg}")
    return False


def _load_metrics(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _theta_rf(metrics: Dict[str, Any]) -> float:
    cfg = metrics.get("config") or {}
    return float(cfg.get("theta_rf", 0.5))


def plot_confusion(
    out_dir: Path,
    pred_df: Optional[pd.DataFrame],
    metrics: Optional[Dict[str, Any]],
    theta_rf: float,
    dpi: int,
    figsize: Tuple[float, float],
) -> Optional[Path]:
    titles_mat: List[Tuple[str, np.ndarray]] = []

    if pred_df is not None and {"y_true", "rf_prob", "final_binary"}.issubset(
        pred_df.columns
    ):
        y = pred_df["y_true"].to_numpy(dtype=int)
        rf_pred = (pred_df["rf_prob"].to_numpy(dtype=float) >= theta_rf).astype(int)
        ens_pred = pred_df["final_binary"].to_numpy(dtype=int)
        cm_rf = confusion_matrix(y, rf_pred, labels=[0, 1])
        cm_en = confusion_matrix(y, ens_pred, labels=[0, 1])
        titles_mat = [
            ("RF (prob >= theta_rf)", cm_rf),
            ("Ensemble (HIGH/CRITICAL -> attack)", cm_en),
        ]
    elif metrics is not None:
        test = metrics.get("test") or {}
        rf_m = test.get("rf_alone_metrics") or {}
        en_m = test.get("binary_metrics") or {}
        for name, m in (("RF alone", rf_m), ("Ensemble", en_m)):
            tn, fp, fn, tp = m.get("tn"), m.get("fp"), m.get("fn"), m.get("tp")
            if None in (tn, fp, fn, tp):
                continue
            cm = np.array([[tn, fp], [fn, tp]], dtype=float)
            titles_mat.append((name, cm))
    else:
        return None

    if not titles_mat:
        return None

    n = len(titles_mat)
    fig, axes = plt.subplots(
        1, n, figsize=(figsize[0] * 0.65 * n, figsize[1] * 0.55)
    )
    if n == 1:
        axes = np.array([axes])
    for ax, (title, cm) in zip(axes.flat, titles_mat):
        sns.heatmap(
            cm,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            ax=ax,
            cbar=True,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        ax.set_title(title)
    plt.suptitle("Confusion matrices (test set)", fontsize=14, y=1.02)
    plt.tight_layout()
    outp = out_dir / "confusion_rf_vs_ensemble.png"
    fig.savefig(outp, bbox_inches="tight")
    plt.close(fig)
    print(f"[confusion] -> {outp}")
    return outp


def _roc_one(
    y: np.ndarray, s: np.ndarray, name: str
) -> Tuple[np.ndarray, np.ndarray, float]:
    if len(np.unique(y)) < 2:
        return np.array([0, 1]), np.array([0, 1]), float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    a = roc_auc_score(y, s)
    return fpr, tpr, a


def plot_roc(
    out_dir: Path, pred_df: pd.DataFrame, dpi: int, figsize: Tuple[float, float]
) -> Optional[Path]:
    y = pred_df["y_true"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=figsize)
    for col, label in (
        ("rf_prob", "RF probability"),
        ("risk_score", "Ensemble risk (0.85*RF + 0.15*unsup)"),
    ):
        if col not in pred_df.columns:
            continue
        s = pred_df[col].to_numpy(dtype=float)
        fpr, tpr, ar = _roc_one(y, s, label)
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={ar:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    outp = out_dir / "roc_curves.png"
    fig.savefig(outp, bbox_inches="tight")
    plt.close(fig)
    print(f"[roc] -> {outp}")
    return outp


def plot_pr(
    out_dir: Path, pred_df: pd.DataFrame, dpi: int, figsize: Tuple[float, float]
) -> Optional[Path]:
    y = pred_df["y_true"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=figsize)
    for col, label in (
        ("rf_prob", "RF probability"),
        ("risk_score", "Ensemble risk"),
    ):
        if col not in pred_df.columns:
            continue
        s = pred_df[col].to_numpy(dtype=float)
        if len(np.unique(y)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y, s)
        ap = auc(rec, prec)
        ax.plot(rec, prec, lw=2, label=f"{label} (AP={ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve (test set)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    outp = out_dir / "pr_curves.png"
    fig.savefig(outp, bbox_inches="tight")
    plt.close(fig)
    print(f"[pr] -> {outp}")
    return outp


def plot_metrics_bars(
    out_dir: Path, metrics: Dict[str, Any], dpi: int, figsize: Tuple[float, float]
) -> Optional[Path]:
    test = metrics.get("test") or {}
    rf_m = test.get("rf_alone_metrics")
    en_m = test.get("binary_metrics")
    if not rf_m or not en_m:
        return None
    keys = ("accuracy", "precision", "recall", "f1", "auc", "fpr")
    labels_en = ("Accuracy", "Precision", "Recall", "F1", "AUC", "FPR")
    x = np.arange(len(keys))
    w = 0.35
    rf_vals = [float(rf_m.get(k) or 0.0) for k in keys]
    en_vals = [float(en_m.get(k) or 0.0) for k in keys]
    fig, ax = plt.subplots(figsize=(figsize[0] * 1.1, figsize[1]))
    ax.bar(x - w / 2, rf_vals, w, label="RF alone")
    ax.bar(x + w / 2, en_vals, w, label="Ensemble")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_en)
    ax.set_ylabel("Score")
    ax.set_title("Test set: RF alone vs. ensemble (metrics_ensemble.json)")
    ax.legend()
    ax.set_ylim(0, max(max(rf_vals + en_vals) * 1.08, 0.05))
    plt.tight_layout()
    outp = out_dir / "metrics_bars.png"
    fig.savefig(outp, bbox_inches="tight")
    plt.close(fig)
    print(f"[metrics] -> {outp}")
    return outp


def plot_calibration(
    out_dir: Path, pred_df: pd.DataFrame, dpi: int, figsize: Tuple[float, float]
) -> Optional[Path]:
    y = pred_df["y_true"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        return None
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for col, label in (
        ("rf_prob", "RF prob"),
        ("risk_score", "Risk score"),
    ):
        if col not in pred_df.columns:
            continue
        s = np.clip(pred_df[col].to_numpy(dtype=float), 0, 1)
        prob_true, prob_pred = calibration_curve(y, s, n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, "s-", lw=2, label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve (binned, test set)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    outp = out_dir / "calibration.png"
    fig.savefig(outp, bbox_inches="tight")
    plt.close(fig)
    print(f"[calibration] -> {outp}")
    return outp


def plot_threshold(
    out_dir: Path, pred_df: pd.DataFrame, dpi: int, figsize: Tuple[float, float]
) -> Optional[Path]:
    y = pred_df["y_true"].to_numpy(dtype=int)
    if "risk_score" not in pred_df.columns:
        return None
    s = pred_df["risk_score"].to_numpy(dtype=float)
    thresholds = np.linspace(0, 1, 101)
    fprs, recs, precs = [], [], []
    for t in thresholds:
        pred = (s >= t).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        fprs.append(fpr)
        recs.append(rec)
        precs.append(prec)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(thresholds, fprs, label="FPR", lw=2)
    ax.plot(thresholds, recs, label="Recall", lw=2)
    ax.plot(thresholds, precs, label="Precision", lw=2)
    ax.set_xlabel("Threshold (on risk_score)")
    ax.set_ylabel("Rate")
    ax.set_title("FPR, recall, and precision vs. threshold (risk_score)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    outp = out_dir / "threshold_tradeoff.png"
    fig.savefig(outp, bbox_inches="tight")
    plt.close(fig)
    print(f"[threshold] -> {outp}")
    return outp


def plot_alerts(
    out_dir: Path, metrics: Dict[str, Any], dpi: int, figsize: Tuple[float, float]
) -> Optional[Path]:
    test = metrics.get("test") or {}
    dist = test.get("alert_distribution")
    if not dist:
        return None
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    counts = [int(dist[k]["count"]) for k in order if k in dist]
    if not counts:
        return None
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("colorblind", n_colors=len(order))
    ax.bar(order[: len(counts)], counts, color=colors)
    ax.set_ylabel("Count")
    ax.set_title("Test set: alert-level sample counts")
    plt.xticks(rotation=15)
    plt.tight_layout()
    outp = out_dir / "alert_distribution.png"
    fig.savefig(outp, bbox_inches="tight")
    plt.close(fig)
    print(f"[alerts] -> {outp}")
    return outp


def plot_hist(
    out_dir: Path,
    pred_df: pd.DataFrame,
    dpi: int,
    figsize: Tuple[float, float],
    max_sample: int,
) -> Optional[Path]:
    y = pred_df["y_true"].to_numpy(dtype=int)
    benign = y == 0
    attack = y == 1
    cols = [c for c in ("rf_prob", "unsup_score", "risk_score") if c in pred_df.columns]
    if not cols:
        return None
    df = pred_df
    if len(df) > max_sample:
        df = df.sample(max_sample, random_state=42)
        y = df["y_true"].to_numpy(dtype=int)
        benign = y == 0
        attack = y == 1
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n * 0.45, figsize[1]))
    if n == 1:
        axes = np.array([axes])
    titles = {"rf_prob": "RF prob", "unsup_score": "Unsup score", "risk_score": "Risk"}
    for ax, col in zip(axes.flat, cols):
        v = df[col].to_numpy(dtype=float)
        ax.hist(
            v[benign],
            bins=40,
            alpha=0.55,
            label="BENIGN",
            density=True,
            range=(0, 1) if col != "unsup_score" else None,
        )
        ax.hist(
            v[attack],
            bins=40,
            alpha=0.55,
            label="ATTACK",
            density=True,
            range=(0, 1) if col != "unsup_score" else None,
        )
        ax.set_title(titles.get(col, col))
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    plt.suptitle("Score distributions by true label", y=1.02)
    plt.tight_layout()
    outp = out_dir / "score_distributions.png"
    fig.savefig(outp, bbox_inches="tight")
    plt.close(fig)
    print(f"[hist] -> {outp}")
    return outp


def plot_label_dist(
    processed_dir: Path,
    out_dir: Path,
    dpi: int,
    figsize: Tuple[float, float],
    strict: bool,
) -> Optional[Path]:
    splits = []
    for name in ("train", "val", "test"):
        p = processed_dir / f"{name}_binary.csv"
        if not _need(p, "label_dist", strict):
            continue
        try:
            bit = pd.read_csv(p, usecols=["binary_label"])
        except ValueError:
            bit = pd.read_csv(p)
            if "binary_label" not in bit.columns:
                _warn(f"{p}: missing binary_label column")
                continue
        vc = bit["binary_label"].value_counts().sort_index()
        splits.append((name, vc))
    if not splits:
        return None
    fig, ax = plt.subplots(figsize=figsize)
    names = [s[0] for s in splits]
    b0 = [float(s[1].get(0, 0)) for s in splits]
    b1 = [float(s[1].get(1, 0)) for s in splits]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, b0, w, label="BENIGN (0)")
    ax.bar(x + w / 2, b1, w, label="ATTACK (1)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Count")
    ax.set_title("binary_label counts by data split")
    ax.legend()
    plt.tight_layout()
    outp = out_dir / "label_distribution_splits.png"
    fig.savefig(outp, bbox_inches="tight")
    plt.close(fig)
    print(f"[label_dist] -> {outp}")
    return outp


def plot_importance(
    model_path: Path,
    feature_names_path: Path,
    out_dir: Path,
    top_k: int,
    dpi: int,
    figsize: Tuple[float, float],
) -> Optional[Path]:
    model = joblib.load(model_path)
    names = [
        line.strip()
        for line in feature_names_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return None
    imp = np.asarray(imp, dtype=float)
    n = min(len(names), len(imp), top_k)
    if n == 0:
        return None
    idx = np.argsort(imp)[::-1][:n]
    fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] * 1.1))
    y_pos = np.arange(len(idx))
    ax.barh(y_pos, imp[idx], color=sns.color_palette("colorblind", n_colors=1)[0])
    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[i] for i in idx], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Random forest feature importance (top {n})")
    plt.tight_layout()
    outp = out_dir / "rf_feature_importance_topk.png"
    fig.savefig(outp, bbox_inches="tight")
    plt.close(fig)
    print(f"[importance] -> {outp}")
    return outp


def _flatten_numeric_metrics(obj: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(obj, dict):
        if "metrics" in obj and isinstance(obj["metrics"], dict):
            obj = obj["metrics"]
        for k, v in obj.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[str(k)] = float(v)
            elif isinstance(v, dict) and "value" in v:
                try:
                    out[str(k)] = float(v["value"])
                except (TypeError, ValueError):
                    pass
    return out


def plot_trace_eval(
    path: Path, out_dir: Path, dpi: int, figsize: Tuple[float, float]
) -> Optional[Path]:
    data = json.loads(path.read_text(encoding="utf-8"))
    flat = _flatten_numeric_metrics(data)
    if not flat:
        return None
    names = list(flat.keys())
    vals = [flat[k] for k in names]
    fig, ax = plt.subplots(figsize=(figsize[0] * 1.1, figsize[1]))
    ax.barh(names, vals, color=sns.color_palette("colorblind", n_colors=len(names)))
    ax.set_xlabel("Score")
    ax.set_title("Traceback evaluation metrics (trace_eval.json)")
    plt.tight_layout()
    outp = out_dir / "trace_eval.png"
    fig.savefig(outp, bbox_inches="tight")
    plt.close(fig)
    print(f"[trace_eval] -> {outp}")
    return outp


def plot_latency(
    path: Path, out_dir: Path, dpi: int, figsize: Tuple[float, float]
) -> Optional[Path]:
    df = pd.read_csv(path)
    if "latency_ms" in df.columns:
        col = "latency_ms"
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            return None
        col = num_cols[0]
    v = df[col].dropna().to_numpy(dtype=float)
    if len(v) == 0:
        return None
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(v, bins=min(50, max(10, len(v) // 20)), color="steelblue", edgecolor="white")
    ax.set_xlabel(str(col))
    ax.set_ylabel("Count")
    ax.set_title("Detection latency distribution (latency.csv)")
    plt.tight_layout()
    outp = out_dir / "latency_distribution.png"
    fig.savefig(outp, bbox_inches="tight")
    plt.close(fig)
    print(f"[latency] -> {outp}")
    return outp


def plot_pca(
    processed_dir: Path,
    feature_names_path: Path,
    out_dir: Path,
    sample_size: int,
    random_state: int,
    dpi: int,
    figsize: Tuple[float, float],
    strict: bool,
) -> Optional[Path]:
    test_p = processed_dir / "test_binary.csv"
    if not _need(test_p, "pca", strict):
        return None
    names = [
        line.strip()
        for line in feature_names_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    usecols = ["binary_label"] + names
    try:
        df = pd.read_csv(test_p, usecols=usecols)
    except ValueError:
        df = pd.read_csv(test_p, nrows=min(sample_size * 4, 200_000))
        missing = [c for c in names if c not in df.columns]
        if missing:
            raise KeyError(f"test_binary missing feature columns: {missing[:5]}")
        df = df.dropna(subset=["binary_label"])
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=random_state)
    X = df[names].replace([np.inf, -np.inf], np.nan).fillna(df[names].median())
    y = df["binary_label"].to_numpy(dtype=int)
    X = X.to_numpy(dtype=np.float64)
    pca = PCA(n_components=2, random_state=random_state)
    z = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=figsize)
    m0 = y == 0
    m1 = y == 1
    ax.scatter(z[m0, 0], z[m0, 1], s=6, alpha=0.35, label="BENIGN")
    ax.scatter(z[m1, 0], z[m1, 1], s=6, alpha=0.35, label="ATTACK")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"PCA scatter (test_binary sample, n={len(y)})")
    ax.legend()
    plt.tight_layout()
    outp = out_dir / "pca_scatter_test.png"
    fig.savefig(outp, bbox_inches="tight")
    plt.close(fig)
    print(f"[pca] -> {outp}")
    return outp


def _parse_only(s: str) -> Set[str]:
    parts = {p.strip().lower() for p in s.split(",") if p.strip()}
    valid = set(CORE_TASKS) | {"pca", "all"}
    bad = parts - valid
    if bad:
        raise ValueError(f"Unknown tasks: {bad}; valid: {sorted(valid)}")
    tasks: Set[str] = set()
    if "all" in parts:
        tasks |= set(CORE_TASKS)
    tasks |= parts - {"all"}
    return tasks


def _figsize_preset(preset: str) -> Tuple[float, float]:
    if preset == "slides":
        return (10.0, 5.5)
    return (6.5, 4.2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export detection-side figures for papers (ensemble_detector outputs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "processed",
        help="Directory with metrics_ensemble.json, *_binary.csv, test_predictions_ensemble.csv",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory with supervised_rf_binary.joblib",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Figure output directory (default: processed-dir/figures)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--fig-preset",
        choices=("paper", "slides"),
        default="paper",
        help="paper: single-column size; slides: widescreen",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        help="Comma-separated task names, or 'all' (excludes pca unless listed)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if a required input file is missing",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Number of top features to plot")
    parser.add_argument(
        "--hist-max-rows",
        type=int,
        default=80_000,
        help="Max rows for hist task (subsample for speed)",
    )
    parser.add_argument("--pca-sample", type=int, default=8000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    processed = args.processed_dir.resolve()
    model_dir = args.model_dir.resolve()
    out_dir = (args.out_dir or (processed / "figures")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = _parse_only(args.only)

    figsize = _figsize_preset(args.fig_preset)
    _apply_style(args.dpi, figsize)

    metrics_path = processed / "metrics_ensemble.json"
    pred_path = processed / "test_predictions_ensemble.csv"
    feat_path = processed / "feature_names.txt"
    rf_model = model_dir / "supervised_rf_binary.joblib"
    trace_path = processed / "trace_eval.json"
    latency_path = processed / "latency.csv"

    metrics: Optional[Dict[str, Any]] = None
    pred_df: Optional[pd.DataFrame] = None

    if any(
        t in tasks
        for t in ("confusion", "metrics", "alerts")
    ):
        if _need(metrics_path, "metrics_json", args.strict):
            metrics = _load_metrics(metrics_path)

    if any(
        t in tasks
        for t in (
            "confusion",
            "roc",
            "pr",
            "calibration",
            "threshold",
            "hist",
        )
    ):
        if _need(pred_path, "predictions_csv", args.strict):
            pred_df = _load_predictions(pred_path)

    theta = _theta_rf(metrics) if metrics else 0.5

    if "confusion" in tasks:
        plot_confusion(out_dir, pred_df, metrics, theta, args.dpi, figsize)

    if "roc" in tasks and pred_df is not None:
        plot_roc(out_dir, pred_df, args.dpi, figsize)

    if "pr" in tasks and pred_df is not None:
        plot_pr(out_dir, pred_df, args.dpi, figsize)

    if "metrics" in tasks and metrics is not None:
        plot_metrics_bars(out_dir, metrics, args.dpi, figsize)

    if "calibration" in tasks and pred_df is not None:
        plot_calibration(out_dir, pred_df, args.dpi, figsize)

    if "threshold" in tasks and pred_df is not None:
        plot_threshold(out_dir, pred_df, args.dpi, figsize)

    if "alerts" in tasks and metrics is not None:
        plot_alerts(out_dir, metrics, args.dpi, figsize)

    if "hist" in tasks and pred_df is not None:
        plot_hist(out_dir, pred_df, args.dpi, figsize, args.hist_max_rows)

    if "label_dist" in tasks:
        plot_label_dist(processed, out_dir, args.dpi, figsize, args.strict)

    if "importance" in tasks:
        if _need(feat_path, "importance", args.strict) and _need(
            rf_model, "importance", args.strict
        ):
            plot_importance(
                rf_model, feat_path, out_dir, args.top_k, args.dpi, figsize
            )

    if "trace_eval" in tasks:
        if trace_path.exists():
            plot_trace_eval(trace_path, out_dir, args.dpi, figsize)
        elif args.strict:
            raise FileNotFoundError(f"[trace_eval] missing: {trace_path}")
        else:
            print(f"WARNING: [trace_eval] missing file, skip: {trace_path}")

    if "latency" in tasks:
        if latency_path.exists():
            plot_latency(latency_path, out_dir, args.dpi, figsize)
        elif args.strict:
            raise FileNotFoundError(f"[latency] missing: {latency_path}")
        else:
            print(f"WARNING: [latency] missing file, skip: {latency_path}")

    if "pca" in tasks:
        if _need(feat_path, "pca", args.strict):
            plot_pca(
                processed,
                feat_path,
                out_dir,
                args.pca_sample,
                args.random_state,
                args.dpi,
                figsize,
                args.strict,
            )

    print(f"\nDone. Output directory: {out_dir}")


if __name__ == "__main__":
    main()
