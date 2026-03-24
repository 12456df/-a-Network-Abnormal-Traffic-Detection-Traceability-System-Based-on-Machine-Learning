"""
supervised_model.py
===================

Supervised binary anomaly detection training script.

This script focuses on RandomForest only and implements:
1) Binary classification (BENIGN=0 / ATTACK=1)
2) Dual threshold strategy from validation set:
   - high_precision threshold: operations-friendly, lower false positives
   - high_recall threshold: security-first, lower false negatives
3) Test-set flow-level anomaly marking outputs for traceback pipeline
4) Optional IP-level aggregation for suspicious source ranking
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def _sanitize_for_json(obj: Any) -> Any:
    """将 float NaN/inf 转为 None，使 json.dumps 输出合法 JSON。"""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    return json.dumps(_sanitize_for_json(obj), **kwargs)


@dataclass
class TrainConfig:
    processed_dir: Path = PROJECT_ROOT / "processed"
    model_dir: Path = PROJECT_ROOT / "models"
    binary_label_col: str = "binary_label"
    sample_id_col: str = "sample_id"
    random_state: int = 42
    n_estimators: int = 400
    max_depth: int | None = None
    precision_target: float = 0.995
    recall_target: float = 0.995
    ip_attack_ratio_threshold: float = 0.5

    def __post_init__(self) -> None:
        for name in ("precision_target", "recall_target", "ip_attack_ratio_threshold"):
            val = getattr(self, name)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} 必须在 [0, 1] 范围内，当前值: {val}")


def _must_exist(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")


def read_feature_names(path: Path) -> list[str]:
    _must_exist(path)
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise RuntimeError(f"特征列表为空: {path}")
    return names


def load_binary_splits(processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = processed_dir / "train_binary.csv"
    val_path = processed_dir / "val_binary.csv"
    test_path = processed_dir / "test_binary.csv"
    for path in [train_path, val_path, test_path]:
        _must_exist(path)
    return pd.read_csv(train_path), pd.read_csv(val_path), pd.read_csv(test_path)


def ensure_feature_alignment(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    feature_cols = list(feature_cols)
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise KeyError(f"数据缺少特征列，示例: {missing[:5]}")
    return df[feature_cols].copy()


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> Dict[str, float]:
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


def print_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> Dict[str, float]:
    result = metrics_dict(y_true, y_pred, y_score)
    print(f"\n[{name}]")
    print(
        "accuracy={accuracy:.4f} precision={precision:.4f} recall={recall:.4f} "
        "f1={f1:.4f} auc={auc:.4f} fpr={fpr:.4f}".format(**result)
    )
    print("confusion_matrix [[tn, fp], [fn, tp]] =", [[int(result["tn"]), int(result["fp"])], [int(result["fn"]), int(result["tp"])]])
    print("classification_report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    return result


def build_rf_model(cfg: TrainConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        class_weight="balanced_subsample",
        random_state=cfg.random_state,
        n_jobs=-1,
    )


def find_high_precision_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    precision_target: float,
) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns len(thresholds) == len(precisions) - 1
    valid = np.where(precisions[:-1] >= precision_target)[0]
    if len(valid) == 0:
        # fallback: use the threshold with max precision
        best_idx = int(np.nanargmax(precisions[:-1]))
        return float(thresholds[best_idx])
    # among candidates meeting precision, prefer highest recall
    best_idx = valid[int(np.nanargmax(recalls[valid]))]
    return float(thresholds[best_idx])


def find_high_recall_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    recall_target: float,
) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    valid = np.where(recalls[:-1] >= recall_target)[0]
    if len(valid) == 0:
        # fallback: use threshold with max recall
        best_idx = int(np.nanargmax(recalls[:-1]))
        return float(thresholds[best_idx])
    # among candidates meeting recall, prefer highest precision
    best_idx = valid[int(np.nanargmax(precisions[valid]))]
    return float(thresholds[best_idx])


def evaluate_threshold_bundle(
    split_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    all_metrics: Dict[str, Dict[str, float]] = {}
    for key, th in thresholds.items():
        y_pred = (y_score >= th).astype(int)
        metrics = print_metrics(f"{split_name}-{key}(th={th:.6f})", y_true, y_pred, y_score)
        all_metrics[key] = metrics
    return all_metrics


def train_supervised_rf(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: TrainConfig,
) -> Tuple[RandomForestClassifier, Dict[str, float], pd.DataFrame, Dict[str, Dict[str, Dict[str, float]]]]:
    x_train = ensure_feature_alignment(train_df, feature_cols)
    y_train = train_df[cfg.binary_label_col].to_numpy(dtype=int)
    x_val = ensure_feature_alignment(val_df, feature_cols)
    y_val = val_df[cfg.binary_label_col].to_numpy(dtype=int)
    x_test = ensure_feature_alignment(test_df, feature_cols)
    y_test = test_df[cfg.binary_label_col].to_numpy(dtype=int)

    for name, xdf in [("train", x_train), ("val", x_val), ("test", x_test)]:
        n_inf = np.isinf(xdf.values).sum()
        n_nan = np.isnan(xdf.values).sum()
        if n_inf > 0 or n_nan > 0:
            print(f"[WARNING] {name} 集特征含 {n_nan} 个 NaN、{n_inf} 个 inf，将替换为列中位数")
    x_train = x_train.replace([np.inf, -np.inf], np.nan)
    train_medians = x_train.median()
    x_train = x_train.fillna(train_medians)
    x_val = x_val.replace([np.inf, -np.inf], np.nan).fillna(train_medians)
    x_test = x_test.replace([np.inf, -np.inf], np.nan).fillna(train_medians)

    model = build_rf_model(cfg)
    model.fit(x_train, y_train)

    train_prob = model.predict_proba(x_train)[:, 1]
    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    thresholds = {
        "default": 0.5,
        "high_precision": find_high_precision_threshold(y_val, val_prob, cfg.precision_target),
        "high_recall": find_high_recall_threshold(y_val, val_prob, cfg.recall_target),
    }
    print(
        "\n[Thresholds] "
        f"default={thresholds['default']:.6f}, "
        f"high_precision={thresholds['high_precision']:.6f}, "
        f"high_recall={thresholds['high_recall']:.6f}"
    )

    print("\n[Train Metrics]")
    train_metrics = evaluate_threshold_bundle("Train", y_train, train_prob, thresholds)
    print("\n[Validation Metrics]")
    val_metrics = evaluate_threshold_bundle("Val", y_val, val_prob, thresholds)
    print("\n[Test Metrics]")
    test_metrics = evaluate_threshold_bundle("Test", y_test, test_prob, thresholds)

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n[Supervised-RF] Top-10 全局重要特征:")
    for name, value in importances.head(10).items():
        print(f"  - {name}: {value:.6f}")

    sample_ids = (
        test_df[cfg.sample_id_col].values
        if cfg.sample_id_col in test_df.columns
        else np.arange(len(test_df))
    )
    pred_df = pd.DataFrame(
        {
            cfg.sample_id_col: sample_ids,
            "y_true": y_test,
            "attack_prob": test_prob,
            "pred_default": (test_prob >= thresholds["default"]).astype(int),
            "pred_high_precision": (test_prob >= thresholds["high_precision"]).astype(int),
            "pred_high_recall": (test_prob >= thresholds["high_recall"]).astype(int),
        }
    )
    all_split_metrics = {"train": train_metrics, "val": val_metrics, "test": test_metrics}
    return model, thresholds, pred_df, all_split_metrics


def save_ip_level_summary(
    cfg: TrainConfig,
    supervised_pred_df: pd.DataFrame,
) -> None:
    trace_path = cfg.processed_dir / "trace_metadata.csv"
    if not trace_path.exists():
        print("[IP Summary] 未找到 trace_metadata.csv，跳过 IP 级汇总。")
        return

    trace_df = pd.read_csv(trace_path)
    required_cols = {cfg.sample_id_col, "Source IP"}
    if not required_cols.issubset(set(trace_df.columns)):
        print("[IP Summary] trace_metadata 缺少 sample_id/Source IP，跳过 IP 级汇总。")
        return

    test_trace = trace_df.copy()
    if "split_binary" in test_trace.columns:
        test_trace = test_trace[test_trace["split_binary"] == "test"]

    merged = test_trace.merge(supervised_pred_df, on=cfg.sample_id_col, how="inner")
    if merged.empty:
        print("[IP Summary] 测试集预测与 trace_metadata 未匹配到记录，跳过。")
        return

    ip_summary = (
        merged.groupby("Source IP", dropna=False)
        .agg(
            total_flows=("pred_default", "size"),
            pred_attack_flows_default=("pred_default", "sum"),
            pred_attack_ratio_default=("pred_default", "mean"),
            pred_attack_ratio_high_precision=("pred_high_precision", "mean"),
            pred_attack_ratio_high_recall=("pred_high_recall", "mean"),
            avg_attack_prob=("attack_prob", "mean"),
            true_attack_ratio=("y_true", "mean"),
        )
        .reset_index()
        .sort_values("pred_attack_ratio_default", ascending=False)
    )
    ip_summary["ip_is_attack_default"] = (
        ip_summary["pred_attack_ratio_default"] >= cfg.ip_attack_ratio_threshold
    ).astype(int)
    ip_summary["ip_is_attack_high_precision"] = (
        ip_summary["pred_attack_ratio_high_precision"] >= cfg.ip_attack_ratio_threshold
    ).astype(int)
    ip_summary["ip_is_attack_high_recall"] = (
        ip_summary["pred_attack_ratio_high_recall"] >= cfg.ip_attack_ratio_threshold
    ).astype(int)

    out_path = cfg.processed_dir / "ip_attack_summary_test.csv"
    ip_summary.to_csv(out_path, index=False)
    print(f"\n[IP Summary] 已输出: {out_path}")
    print("[IP Summary] Top-10 可疑 IP:")
    for _, row in ip_summary.head(10).iterrows():
        print(
            f"  - {row['Source IP']}: ip_is_attack_default={int(row['ip_is_attack_default'])}, "
            f"pred_attack_ratio_default={float(row['pred_attack_ratio_default']):.4f}, total_flows={int(row['total_flows'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train supervised RandomForest binary attack detector.")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "processed")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--precision-target", type=float, default=0.995, help="Target precision for high_precision threshold.")
    parser.add_argument("--recall-target", type=float, default=0.995, help="Target recall for high_recall threshold.")
    args = parser.parse_args()

    cfg = TrainConfig(
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
        precision_target=args.precision_target,
        recall_target=args.recall_target,
    )

    print("[supervised_model] 所需监督训练文件:")
    print("  - train_binary.csv, val_binary.csv, test_binary.csv")
    print("  - feature_names.txt")
    print("[supervised_model] 输出阈值策略: default / high_precision / high_recall")

    feature_cols = read_feature_names(cfg.processed_dir / "feature_names.txt")
    train_df, val_df, test_df = load_binary_splits(cfg.processed_dir)
    for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if cfg.binary_label_col not in df.columns:
            raise KeyError(f"{df_name} 集缺少标签列 {cfg.binary_label_col}")

    print(
        "[supervised_model] 数据规模: "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, features={len(feature_cols)}"
    )

    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    rf_model, thresholds, supervised_pred_df, split_metrics = train_supervised_rf(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        cfg=cfg,
    )
    sup_model_path = cfg.model_dir / "supervised_rf_binary.joblib"
    joblib.dump(rf_model, sup_model_path)
    print(f"\n[supervised_model] 已保存监督模型: {sup_model_path}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "hyperparams": {
            "n_estimators": cfg.n_estimators,
            "max_depth": cfg.max_depth,
            "class_weight": "balanced_subsample",
            "random_state": cfg.random_state,
        },
        "threshold_targets": {
            "precision_target": cfg.precision_target,
            "recall_target": cfg.recall_target,
        },
        "thresholds": thresholds,
    }
    manifest_path = cfg.model_dir / "model_manifest.json"
    manifest_path.write_text(safe_json_dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[supervised_model] 已保存模型元信息: {manifest_path}")

    test_pred_path = cfg.processed_dir / "test_predictions_supervised.csv"
    supervised_pred_df.to_csv(test_pred_path, index=False)
    print(f"[supervised_model] 已保存测试集预测: {test_pred_path}")

    threshold_path = cfg.model_dir / "supervised_rf_thresholds.json"
    threshold_path.write_text(safe_json_dumps(thresholds, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[supervised_model] 已保存阈值配置: {threshold_path}")

    metrics_path = cfg.processed_dir / "metrics_supervised_rf.json"
    metrics_path.write_text(safe_json_dumps(split_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[supervised_model] 已保存指标报告: {metrics_path}")

    save_ip_level_summary(cfg, supervised_pred_df)
    print("\n[supervised_model] 训练与评估完成。")


if __name__ == "__main__":
    main()
