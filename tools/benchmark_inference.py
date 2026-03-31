"""
tools/benchmark_inference.py
============================
离线测时：RandomForest predict_proba；可选测试「集成路径」中单次前向（RF + IF + AE + VAE 批量）。

默认仅依赖 numpy/pandas/joblib/sklearn，不需要 PyTorch。加 --ensemble 时需已安装 torch 且无监督模型文件。

输出：reports/benchmark_inference.json（含 flows/s 粗估）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from supervised_model import read_feature_names  # noqa: E402


def extract_and_clean(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """与 ensemble_detector.extract_and_clean 一致，避免默认依赖 torch。"""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"缺少特征列: {missing[:5]}")
    X = df[cols].to_numpy(dtype=np.float64)
    bad = ~np.isfinite(X)
    if bad.any():
        med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        r, c = np.where(bad)
        X[r, c] = med[c]
    return X


def main() -> None:
    pa = argparse.ArgumentParser(description="Benchmark inference throughput (offline).")
    pa.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "processed")
    pa.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models")
    pa.add_argument("--n-rows", type=int, default=5000, help="参与计时的行数（从 test 截取）")
    pa.add_argument("--repeat", type=int, default=5, help="重复完整 forward 的次数")
    pa.add_argument("--warmup", type=int, default=2)
    pa.add_argument(
        "--ensemble",
        action="store_true",
        help="额外测 RF+无监督融合前向（需 torch 与无监督模型文件）",
    )
    pa.add_argument("--device", type=str, default=None)
    args = pa.parse_args()

    test_path = args.processed_dir / "test_binary.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"缺少 {test_path}")
    test_df = pd.read_csv(test_path)
    n = min(args.n_rows, len(test_df))
    test_df = test_df.iloc[:n].copy()

    rf_path = args.model_dir / "supervised_rf_binary.joblib"
    if not rf_path.exists():
        raise FileNotFoundError(f"缺少 {rf_path}")
    rf = joblib.load(rf_path)
    feat_path = args.processed_dir / "feature_names.txt"
    rf_feats = read_feature_names(feat_path)
    X_rf = extract_and_clean(test_df, rf_feats)

    def run_rf() -> None:
        rf.predict_proba(X_rf)

    for _ in range(args.warmup):
        run_rf()

    t0 = time.perf_counter()
    for _ in range(args.repeat):
        run_rf()
    rf_elapsed = time.perf_counter() - t0
    rf_total = n * args.repeat
    rf_fps = rf_total / rf_elapsed if rf_elapsed > 0 else 0.0

    out = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_rows_per_batch": n,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "rf_seconds_total": round(rf_elapsed, 6),
        "rf_flows_per_second": round(rf_fps, 2),
        "note": "单进程、批量 predict_proba；非真实在线链路延迟。",
    }

    if args.ensemble:
        import torch  # noqa: WPS433

        from ensemble_detector import (  # noqa: E402
            EnsembleConfig,
            compute_rf_probs,
            compute_unsup_raw,
            load_models,
        )

        cfg = EnsembleConfig(processed_dir=args.processed_dir, model_dir=args.model_dir)
        device = (
            torch.device(args.device)
            if args.device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        try:
            models = load_models(cfg, device)
        except FileNotFoundError as e:
            out["ensemble_error"] = str(e)
        else:
            X_us = models["scaler"].transform(
                extract_and_clean(test_df, models["unsup_feats"])
            )

            def run_ens() -> None:
                compute_rf_probs(models["rf"], X_rf)
                compute_unsup_raw(models, X_us, device)

            for _ in range(args.warmup):
                run_ens()
            t1 = time.perf_counter()
            for _ in range(args.repeat):
                run_ens()
            ens_elapsed = time.perf_counter() - t1
            ens_fps = (n * args.repeat) / ens_elapsed if ens_elapsed > 0 else 0.0
            out["ensemble_device"] = str(device)
            out["ensemble_seconds_total"] = round(ens_elapsed, 6)
            out["ensemble_flows_per_second"] = round(ens_fps, 2)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "benchmark_inference.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[Benchmark] 已写: {out_path}")


if __name__ == "__main__":
    main()
