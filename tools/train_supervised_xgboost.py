"""
tools/train_supervised_xgboost.py
=================================
与 supervised_model.RandomForest 使用相同数据与验证集阈值策略，训练 XGBoost 二分类对照模型。
输出：models/supervised_xgb_binary.joblib、models/supervised_xgb_manifest.json、
      processed/supervised_xgb_test_predictions.csv
不修改 ensemble_detector（默认仍使用 RF）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from supervised_model import (  # noqa: E402
    ensure_feature_alignment,
    evaluate_threshold_bundle,
    find_high_precision_threshold,
    find_high_recall_threshold,
    load_binary_splits,
    read_feature_names,
    safe_json_dumps,
)


def _sanitize_nested(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_nested(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_nested(v) for v in obj]
    return obj


@dataclass
class XgbTrainConfig:
    processed_dir: Path = PROJECT_ROOT / "processed"
    model_dir: Path = PROJECT_ROOT / "models"
    binary_label_col: str = "binary_label"
    sample_id_col: str = "sample_id"
    random_state: int = 42
    n_estimators: int = 400
    max_depth: int = 8
    learning_rate: float = 0.08
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    precision_target: float = 0.995
    recall_target: float = 0.995


def main() -> None:
    import xgboost as xgb

    pa = argparse.ArgumentParser(description="Train XGBoost binary detector (RF baseline comparison).")
    pa.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "processed")
    pa.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models")
    pa.add_argument("--n-estimators", type=int, default=400)
    pa.add_argument("--max-depth", type=int, default=8)
    pa.add_argument("--learning-rate", type=float, default=0.08)
    pa.add_argument("--precision-target", type=float, default=0.995)
    pa.add_argument("--recall-target", type=float, default=0.995)
    args = pa.parse_args()

    cfg = XgbTrainConfig(
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        precision_target=args.precision_target,
        recall_target=args.recall_target,
    )
    feature_cols = read_feature_names(cfg.processed_dir / "feature_names.txt")
    train_df, val_df, test_df = load_binary_splits(cfg.processed_dir)

    x_train = ensure_feature_alignment(train_df, feature_cols)
    y_train = train_df[cfg.binary_label_col].to_numpy(dtype=int)
    x_val = ensure_feature_alignment(val_df, feature_cols)
    y_val = val_df[cfg.binary_label_col].to_numpy(dtype=int)
    x_test = ensure_feature_alignment(test_df, feature_cols)
    y_test = test_df[cfg.binary_label_col].to_numpy(dtype=int)

    x_train = x_train.replace([np.inf, -np.inf], np.nan)
    train_medians = x_train.median()
    x_train = x_train.fillna(train_medians)
    x_val = x_val.replace([np.inf, -np.inf], np.nan).fillna(train_medians)
    x_test = x_test.replace([np.inf, -np.inf], np.nan).fillna(train_medians)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=1.0,
        min_child_weight=1,
        random_state=cfg.random_state,
        n_jobs=-1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
    )
    print(
        f"[XGB] train={len(y_train)} val={len(y_val)} test={len(y_test)} "
        f"features={len(feature_cols)} scale_pos_weight={scale_pos_weight:.4f}"
    )
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

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )
    print("\n[XGB] Top-10 feature_importances:")
    for name, value in importances.head(10).items():
        print(f"  - {name}: {value:.6f}")

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.model_dir / "supervised_xgb_binary.joblib"
    joblib.dump(model, model_path)
    print(f"\n[XGB] 已保存: {model_path}")

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
    pred_path = cfg.processed_dir / "supervised_xgb_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"[XGB] 测试集预测: {pred_path}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimator": "xgboost.XGBClassifier",
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "hyperparams": {
            "n_estimators": cfg.n_estimators,
            "max_depth": cfg.max_depth,
            "learning_rate": cfg.learning_rate,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
            "random_state": cfg.random_state,
            "tree_method": "hist",
        },
        "threshold_targets": {
            "precision_target": cfg.precision_target,
            "recall_target": cfg.recall_target,
        },
        "thresholds": thresholds,
        "metrics": {
            "train": _sanitize_nested(train_metrics),
            "val": _sanitize_nested(val_metrics),
            "test": _sanitize_nested(test_metrics),
        },
    }
    manifest_path = cfg.model_dir / "supervised_xgb_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"[XGB] manifest: {manifest_path}")

    print("\n[XGB] 完成。对比 RandomForest 请并排查看 test 指标与 supervised_rf 。")


if __name__ == "__main__":
    main()
