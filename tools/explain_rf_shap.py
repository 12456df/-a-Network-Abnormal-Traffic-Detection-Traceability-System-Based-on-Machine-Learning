"""
tools/explain_rf_shap.py
========================
对已训练的 RandomForest 二分类模型做 SHAP TreeExplainer 解释（全局摘要 + Top 特征表）。
依赖：pip install shap

输出目录默认 reports/shap/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shap  # noqa: E402

from supervised_model import ensure_feature_alignment, read_feature_names  # noqa: E402


def main() -> None:
    pa = argparse.ArgumentParser(description="SHAP explanations for supervised RF.")
    pa.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "processed")
    pa.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models")
    pa.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "shap")
    pa.add_argument("--split", choices=["test", "val"], default="test")
    pa.add_argument("--max-samples", type=int, default=2000,
                    help="抽样行数（过大很慢）；<=0 表示全量")
    pa.add_argument("--random-state", type=int, default=42)
    args = pa.parse_args()

    model_path = args.model_dir / "supervised_rf_binary.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"请先训练 RF: {model_path}")

    rf = joblib.load(model_path)
    feature_cols = read_feature_names(args.processed_dir / "feature_names.txt")
    csv_name = f"{args.split}_binary.csv"
    df = pd.read_csv(args.processed_dir / csv_name)

    x_df = ensure_feature_alignment(df, feature_cols)
    x_df = x_df.replace([np.inf, -np.inf], np.nan)
    med = x_df.median()
    x_df = x_df.fillna(med)
    X = x_df.to_numpy(dtype=np.float64)

    rng = np.random.default_rng(args.random_state)
    n = len(X)
    if args.max_samples > 0 and n > args.max_samples:
        idx = rng.choice(n, size=args.max_samples, replace=False)
        Xs = X[idx]
        print(f"[SHAP] 从 {csv_name} 抽样 {args.max_samples} / {n} 行")
    else:
        Xs = X
        print(f"[SHAP] 使用 {csv_name} 全部 {n} 行")

    print("[SHAP] TreeExplainer（可能需要一两分钟）…")
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list):
        shap_attack = sv[1]
    else:
        shap_attack = sv

    mean_abs = np.mean(np.abs(shap_attack), axis=0)
    order = np.argsort(-mean_abs)
    top_features = [
        {"feature": feature_cols[i], "mean_abs_shap": float(mean_abs[i])}
        for i in order[:30]
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    top_path = args.out_dir / "top_features.json"
    top_path.write_text(json.dumps(top_features, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SHAP] 已写: {top_path}")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_attack,
        Xs,
        feature_names=feature_cols,
        show=False,
        max_display=20,
    )
    bar_path = args.out_dir / "summary_beeswarm.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] 已写: {bar_path}")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_attack,
        Xs,
        feature_names=feature_cols,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    bar_only = args.out_dir / "summary_bar.png"
    plt.savefig(bar_only, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] 已写: {bar_only}")

    meta = {
        "split": args.split,
        "n_rows_used": int(Xs.shape[0]),
        "n_features": len(feature_cols),
        "model": str(model_path),
    }
    (args.out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("[SHAP] 完成。")


if __name__ == "__main__":
    main()
