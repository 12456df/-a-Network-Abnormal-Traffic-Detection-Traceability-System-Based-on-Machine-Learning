"""
ensemble_detector.py
====================

Supervised (RF) + Unsupervised (IF / AE / VAE) cascade ensemble detector
with four-level alert system.

Alert Levels
    CRITICAL  ─  RF ≥ θ_rf  AND  unsup ≥ θ_unsup_high   (both models agree)
    HIGH      ─  RF ≥ θ_rf  AND  unsup <  θ_unsup_high   (RF-only, known attack)
    MEDIUM    ─  RF ∈ [θ_gray, θ_rf)  OR  unsup ≥ θ_unsup_extreme  (suspicious)
    LOW       ─  otherwise                                  (normal traffic)

Binary mapping:  CRITICAL + HIGH  →  ATTACK (1)
                 MEDIUM   + LOW   →  BENIGN (0)

MEDIUM-level flows are exported to a separate suspicious-flows table
for the traceback / manual-review pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from unsupervised_model import Autoencoder, VAE, _score_batched

PROJECT_ROOT = Path(__file__).resolve().parent

ALERT_NAMES = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}


class AlertLevel(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


# ═══════════════════════════════════════════════════════════════
#  JSON helpers
# ═══════════════════════════════════════════════════════════════

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
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    return obj


def _json(obj: Any, **kw: Any) -> str:
    return json.dumps(_sanitize(obj), **kw)


# ═══════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class EnsembleConfig:
    processed_dir: Path = PROJECT_ROOT / "processed"
    model_dir: Path = PROJECT_ROOT / "models"

    binary_label_col: str = "binary_label"
    label_col: str = "Label"
    sample_id_col: str = "sample_id"
    random_state: int = 42

    theta_rf: float = 0.5
    theta_gray: float = 0.15
    theta_unsup_high: float = 0.5
    theta_unsup_extreme: float = 0.85

    unsup_weights: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)

    medium_min_attack_ratio: float = 0.30


# ═══════════════════════════════════════════════════════════════
#  Data utilities
# ═══════════════════════════════════════════════════════════════

def _must(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"缺少文件: {p}")


def read_feature_names(p: Path) -> list[str]:
    _must(p)
    return [line.strip() for line in p.read_text("utf-8").splitlines()
            if line.strip()]


def load_csv(p: Path) -> pd.DataFrame:
    _must(p)
    return pd.read_csv(p)


def extract_and_clean(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Extract columns by name; replace inf / NaN with column medians."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"缺少特征列: {missing[:5]}")
    X = df[cols].to_numpy(dtype=np.float64)
    bad = ~np.isfinite(X)
    if bad.any():
        med = np.nanmedian(
            np.where(np.isfinite(X), X, np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        r, c = np.where(bad)
        X[r, c] = med[c]
    return X


# ═══════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════

def load_models(cfg: EnsembleConfig,
                device: torch.device) -> Dict[str, Any]:
    print("[Load] 加载有监督 RF 模型 …")
    rf = joblib.load(cfg.model_dir / "supervised_rf_binary.joblib")

    print("[Load] 加载无监督模型 (IF / AE / VAE / Scaler) …")
    iforest = joblib.load(cfg.model_dir / "unsupervised_iforest.joblib")
    scaler = joblib.load(cfg.model_dir / "unsupervised_scaler.joblib")

    ae_ck = torch.load(cfg.model_dir / "unsupervised_autoencoder.pth",
                       map_location=device, weights_only=False)
    ae = Autoencoder(ae_ck["input_dim"], ae_ck["bottleneck"],
                     ae_ck.get("n_layers", 3), ae_ck.get("dropout", 0.15))
    ae.load_state_dict(ae_ck["state"])
    ae.to(device).eval()

    vae_ck = torch.load(cfg.model_dir / "unsupervised_vae.pth",
                        map_location=device, weights_only=False)
    vae = VAE(vae_ck["input_dim"], vae_ck["bottleneck"],
              vae_ck.get("n_layers", 3), vae_ck.get("dropout", 0.15))
    vae.load_state_dict(vae_ck["state"])
    vae.to(device).eval()

    rf_feats = read_feature_names(cfg.processed_dir / "feature_names.txt")
    unsup_feats = read_feature_names(
        cfg.model_dir / "unsupervised_selected_features.txt")

    print(f"[Load] RF 特征数: {len(rf_feats)}, 无监督特征数: {len(unsup_feats)}")
    return dict(rf=rf, iforest=iforest, ae=ae, vae=vae, scaler=scaler,
                rf_feats=rf_feats, unsup_feats=unsup_feats)


# ═══════════════════════════════════════════════════════════════
#  Score computation
# ═══════════════════════════════════════════════════════════════

def compute_rf_probs(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def compute_unsup_raw(models: dict, X_scaled: np.ndarray,
                      device: torch.device):
    """Return raw IF / AE / VAE anomaly scores."""
    if_s = -models["iforest"].decision_function(X_scaled)
    t = torch.tensor(X_scaled, dtype=torch.float32)
    ae_s = _score_batched(models["ae"], t, device, is_vae=False)
    vae_s = _score_batched(models["vae"], t, device, is_vae=True)
    return if_s, ae_s, vae_s


def _minmax(s: np.ndarray,
            ref: np.ndarray | None = None) -> np.ndarray:
    ref = ref if ref is not None else s
    lo, hi = float(ref.min()), float(ref.max())
    if hi - lo < 1e-12:
        return np.zeros_like(s)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0)


def fuse_scores(if_s, ae_s, vae_s, w,
                ref_if=None, ref_ae=None,
                ref_vae=None) -> np.ndarray:
    """Weighted min-max normalized fusion → [0, 1]."""
    return (w[0] * _minmax(if_s, ref_if)
            + w[1] * _minmax(ae_s, ref_ae)
            + w[2] * _minmax(vae_s, ref_vae))


def search_weights(if_s, ae_s, vae_s, y,
                   steps: int = 11):
    """Grid-search best (w_if, w_ae, w_vae) by val AUC."""
    n1, n2, n3 = _minmax(if_s), _minmax(ae_s), _minmax(vae_s)
    best_w: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    best_auc = -1.0
    for w1 in np.linspace(0, 1, steps):
        for w2 in np.linspace(0, 1 - w1, max(steps, 2)):
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9:
                continue
            w3 = max(w3, 0.0)
            auc = roc_auc_score(y, w1 * n1 + w2 * n2 + w3 * n3)
            if auc > best_auc:
                best_w = (float(w1), float(w2), float(w3))
                best_auc = auc
    return best_w, best_auc


# ═══════════════════════════════════════════════════════════════
#  Four-level alert classification
# ═══════════════════════════════════════════════════════════════

def classify_alerts(rf_p: np.ndarray, unsup_s: np.ndarray,
                    cfg: EnsembleConfig) -> np.ndarray:
    """Vectorised four-level alert assignment."""
    n = len(rf_p)
    levels = np.full(n, AlertLevel.LOW, dtype=np.int32)

    rf_hi = rf_p >= cfg.theta_rf
    unsup_hi = unsup_s >= cfg.theta_unsup_high

    levels[rf_hi & unsup_hi] = AlertLevel.CRITICAL
    levels[rf_hi & ~unsup_hi] = AlertLevel.HIGH

    medium_mask = (~rf_hi) & (
        (rf_p >= cfg.theta_gray) | (unsup_s >= cfg.theta_unsup_extreme))
    levels[medium_mask] = AlertLevel.MEDIUM

    return levels


def to_binary(levels: np.ndarray) -> np.ndarray:
    """CRITICAL + HIGH → 1  (attack) ;  MEDIUM + LOW → 0  (benign)."""
    return (levels >= AlertLevel.HIGH).astype(int)


# ═══════════════════════════════════════════════════════════════
#  Threshold calibration  (on validation set)
# ═══════════════════════════════════════════════════════════════

def calibrate(rf_p: np.ndarray, unsup_s: np.ndarray,
              y: np.ndarray, cfg: EnsembleConfig) -> EnsembleConfig:
    print("\n[Calibrate] 在验证集上自动搜索阈值 …")

    # ── θ_gray ───────────────────────────────────────────────
    #    灰区 [θ_gray, θ_rf) 应包含有意义的攻击密度
    best_g, best_score = 0.15, -1.0
    for g in np.arange(0.02, 0.50, 0.01):
        mask = (rf_p >= g) & (rf_p < cfg.theta_rf)
        if mask.sum() < 10:
            continue
        score = float(y[mask].mean()) * np.log1p(mask.sum())
        if score > best_score:
            best_g, best_score = float(g), score
    cfg.theta_gray = best_g

    gray = (rf_p >= cfg.theta_gray) & (rf_p < cfg.theta_rf)
    gray_ar = float(y[gray].mean()) if gray.sum() > 0 else 0.0
    print(f"  θ_gray = {cfg.theta_gray:.2f}  "
          f"(灰区 {int(gray.sum())} 样本, 攻击率 {gray_ar:.4f})")

    # ── θ_unsup_high ─────────────────────────────────────────
    #    在 RF-TP 中按 25th 百分位切分 → CRITICAL 含 ~75 % RF-TP
    rf_tp = (rf_p >= cfg.theta_rf) & (y == 1)
    if rf_tp.sum() > 0:
        cfg.theta_unsup_high = float(np.percentile(unsup_s[rf_tp], 25))
    crit_frac = (unsup_s[rf_tp] >= cfg.theta_unsup_high).mean() \
        if rf_tp.sum() > 0 else 0
    print(f"  θ_unsup_high = {cfg.theta_unsup_high:.6f}  "
          f"(CRITICAL 占 RF-TP ≈{crit_frac * 100:.0f}%)")

    # ── θ_unsup_extreme ──────────────────────────────────────
    #    在 RF 低置信区 (< θ_gray) 中找保守阈值
    rf_lo = rf_p < cfg.theta_gray
    if rf_lo.sum() > 0:
        best_te = float(np.percentile(unsup_s[rf_lo], 99.5))
        best_rescue = 0
        for th in np.arange(0.50, 0.999, 0.005):
            flagged = rf_lo & (unsup_s >= th)
            if flagged.sum() < 3:
                continue
            ar = float(y[flagged].mean())
            n_atk = int(y[flagged].sum())
            if ar >= cfg.medium_min_attack_ratio and n_atk > best_rescue:
                best_te, best_rescue = float(th), n_atk
        cfg.theta_unsup_extreme = best_te
    print(f"  θ_unsup_extreme = {cfg.theta_unsup_extreme:.6f}")

    return cfg


# ═══════════════════════════════════════════════════════════════
#  Metrics helper
# ═══════════════════════════════════════════════════════════════

def _metrics(y_true: np.ndarray, y_pred: np.ndarray,
             y_score: np.ndarray | None = None) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    r: Dict[str, Any] = dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        fpr=float(fp / (fp + tn)) if (fp + tn) else 0.0,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )
    if y_score is not None and len(np.unique(y_true)) > 1:
        r["auc"] = float(roc_auc_score(y_true, y_score))
    else:
        r["auc"] = None
    return r


# ═══════════════════════════════════════════════════════════════
#  Full evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate(tag: str,
             y: np.ndarray,
             rf_p: np.ndarray,
             unsup_s: np.ndarray,
             levels: np.ndarray,
             cfg: EnsembleConfig,
             attack_labels: np.ndarray | None = None) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    # ── 1. 告警级别分布 ──────────────────────────────────────
    dist: Dict[str, Any] = {}
    for lv in AlertLevel:
        m = levels == lv
        n = int(m.sum())
        a = int(y[m].sum()) if n else 0
        dist[lv.name] = dict(
            count=n, attacks=a,
            attack_ratio=round(a / n, 6) if n else 0.0,
            pct=round(n / len(y) * 100, 2))
    results["alert_distribution"] = dist

    print(f"\n[{tag}] 四级告警分布")
    for name, d in dist.items():
        print(f"  {name:10s} {d['count']:>8d} 条 "
              f"({d['pct']:>6.2f}%)  "
              f"真实攻击率={d['attack_ratio']:.4f}")

    # ── 2. 二分类指标 (融合 vs RF 单独) ───────────────────────
    ens_pred = to_binary(levels)
    risk = 0.85 * rf_p + 0.15 * unsup_s
    ens_m = _metrics(y, ens_pred, risk)
    results["binary_metrics"] = ens_m

    rf_pred = (rf_p >= cfg.theta_rf).astype(int)
    rf_m = _metrics(y, rf_pred, rf_p)
    results["rf_alone_metrics"] = rf_m

    print(f"\n[{tag}] 二分类指标对比")
    print(f"  {'指标':<12s} {'RF 单独':>12s} {'融合系统':>12s} {'Δ':>10s}")
    print(f"  {'─' * 50}")
    for k in ("accuracy", "precision", "recall", "f1", "auc", "fpr"):
        rv = rf_m.get(k) or 0.0
        ev = ens_m.get(k) or 0.0
        d = ev - rv
        print(f"  {k:<12s} {rv:>12.6f} {ev:>12.6f} "
              f"{'+' if d >= 0 else ''}{d:>.6f}")

    print(f"\n  RF 单独:  TP={rf_m['tp']:>6d}  FP={rf_m['fp']:>5d}  "
          f"FN={rf_m['fn']:>5d}  TN={rf_m['tn']:>6d}")
    print(f"  融合系统:  TP={ens_m['tp']:>6d}  FP={ens_m['fp']:>5d}  "
          f"FN={ens_m['fn']:>5d}  TN={ens_m['tn']:>6d}")

    # ── 3. MEDIUM 级别专项分析 ────────────────────────────────
    med = levels == AlertLevel.MEDIUM
    rf_fn = (rf_pred == 0) & (y == 1)
    rescued = rf_fn & med
    total_atk = int(y.sum())

    binary_tp = ens_m["tp"]
    medium_tp = int(y[med].sum())
    extended_recall = ((binary_tp + medium_tp) / total_atk
                       if total_atk else 0.0)
    ma = dict(
        medium_total=int(med.sum()),
        medium_attacks=medium_tp,
        medium_attack_ratio=round(float(y[med].mean()), 4) if med.sum() else 0,
        rf_fn_total=int(rf_fn.sum()),
        rescued_by_medium=int(rescued.sum()),
        rescue_rate=round(int(rescued.sum()) / max(int(rf_fn.sum()), 1), 4),
        extended_recall=round(extended_recall, 6),
    )
    results["medium_analysis"] = ma

    print(f"\n[{tag}] MEDIUM 级别分析")
    print(f"  MEDIUM 总数:        {ma['medium_total']}")
    print(f"  其中真实攻击:       {ma['medium_attacks']}  "
          f"(攻击率 {ma['medium_attack_ratio']:.4f})")
    print(f"  RF 漏判 (FN):       {ma['rf_fn_total']}")
    print(f"  被 MEDIUM 捕获的 FN: {ma['rescued_by_medium']}  "
          f"({ma['rescue_rate'] * 100:.1f}%)")
    print(f"  扩展召回率 (含人工复核 MEDIUM): "
          f"{ma['extended_recall']:.6f}  "
          f"(原始 recall {rf_m['recall']:.6f})")

    # ── 4. 按攻击类型分析 ─────────────────────────────────────
    if attack_labels is not None:
        pa: Dict[str, Any] = {}
        print(f"\n[{tag}] 按攻击类型的告警分布")
        for lab in sorted(set(attack_labels)):
            if lab == "BENIGN":
                continue
            lm = attack_labels == lab
            total = int(lm.sum())
            bd = {lv.name: int((levels[lm] == lv).sum())
                  for lv in AlertLevel}
            detected = bd["CRITICAL"] + bd["HIGH"]
            rec = detected / total if total else 0.0
            pa[lab] = dict(total=total, detected=detected,
                           recall=round(rec, 4), **bd)
            flag = "[Y]" if rec >= 0.8 else "[N]"
            safe_lab = lab.encode("ascii", errors="replace").decode("ascii")
            print(f"  {flag} {safe_lab:30s}  {detected:>5d}/{total:<5d}  "
                  f"recall={rec:.4f}  "
                  f"C={bd['CRITICAL']}  H={bd['HIGH']}  "
                  f"M={bd['MEDIUM']}  L={bd['LOW']}")
        results["per_attack"] = pa

    return results


# ═══════════════════════════════════════════════════════════════
#  IP-level summary
# ═══════════════════════════════════════════════════════════════

def save_ip_summary(cfg: EnsembleConfig, pred_df: pd.DataFrame) -> None:
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

    def _level_cnt(series, level_name):
        return (series == level_name).sum()

    agg = (
        merged.groupby("Source IP", dropna=False)
        .agg(
            total_flows=("final_binary", "size"),
            attack_flows=("final_binary", "sum"),
            attack_ratio=("final_binary", "mean"),
            critical=pd.NamedAgg(
                "alert_level", lambda x: _level_cnt(x, "CRITICAL")),
            high=pd.NamedAgg(
                "alert_level", lambda x: _level_cnt(x, "HIGH")),
            medium=pd.NamedAgg(
                "alert_level", lambda x: _level_cnt(x, "MEDIUM")),
            avg_risk=("risk_score", "mean"),
            avg_rf_prob=("rf_prob", "mean"),
            avg_unsup=("unsup_score", "mean"),
        )
        .reset_index()
        .sort_values("attack_ratio", ascending=False)
    )

    out = cfg.processed_dir / "ip_attack_summary_ensemble.csv"
    agg.to_csv(out, index=False)
    print(f"\n[IP Summary] 已保存: {out}")
    print("[IP Summary] Top-10 可疑 IP:")
    for _, row in agg.head(10).iterrows():
        print(f"  {row['Source IP']}: "
              f"attack_ratio={row['attack_ratio']:.4f}, "
              f"flows={int(row['total_flows'])}, "
              f"C={int(row['critical'])} H={int(row['high'])} "
              f"M={int(row['medium'])}")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    pa = argparse.ArgumentParser(
        description="Ensemble detector: RF + Unsupervised, 4-level alerts")
    pa.add_argument("--processed-dir", type=Path,
                    default=PROJECT_ROOT / "processed")
    pa.add_argument("--model-dir", type=Path,
                    default=PROJECT_ROOT / "models")
    pa.add_argument("--device", type=str, default=None)
    pa.add_argument("--theta-rf", type=float, default=0.5)
    args = pa.parse_args()

    cfg = EnsembleConfig(
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
        theta_rf=args.theta_rf,
    )
    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available()
                                else "cpu"))
    print(f"[Ensemble] device={device}, θ_rf={cfg.theta_rf}")

    # ── 1. 加载全部模型 ──────────────────────────────────────
    models = load_models(cfg, device)

    # ── 2. 加载数据 ──────────────────────────────────────────
    print("\n[Data] 加载验证集与测试集 …")
    val_df = load_csv(cfg.processed_dir / "val_binary.csv")
    test_df = load_csv(cfg.processed_dir / "test_binary.csv")

    y_val = val_df[cfg.binary_label_col].to_numpy(dtype=int)
    y_test = test_df[cfg.binary_label_col].to_numpy(dtype=int)

    sample_ids = (test_df[cfg.sample_id_col].to_numpy()
                  if cfg.sample_id_col in test_df.columns
                  else np.arange(len(test_df)))
    atk_labels = (test_df[cfg.label_col].astype(str).str.strip().to_numpy()
                  if cfg.label_col in test_df.columns else None)

    X_val_rf = extract_and_clean(val_df, models["rf_feats"])
    X_test_rf = extract_and_clean(test_df, models["rf_feats"])

    X_val_us = models["scaler"].transform(
        extract_and_clean(val_df, models["unsup_feats"]))
    X_test_us = models["scaler"].transform(
        extract_and_clean(test_df, models["unsup_feats"]))

    del val_df, test_df
    print(f"[Data] val={len(y_val)}, test={len(y_test)}, "
          f"RF 特征={len(models['rf_feats'])}, "
          f"无监督特征={len(models['unsup_feats'])}")

    # ── 3. RF 打分 ───────────────────────────────────────────
    print("\n[RF] 计算攻击概率 …")
    rfp_val = compute_rf_probs(models["rf"], X_val_rf)
    rfp_test = compute_rf_probs(models["rf"], X_test_rf)
    del X_val_rf, X_test_rf

    # ── 4. 无监督打分 ────────────────────────────────────────
    print("[Unsup] 计算 IF / AE / VAE 异常分数 …")
    if_v, ae_v, vae_v = compute_unsup_raw(models, X_val_us, device)
    if_t, ae_t, vae_t = compute_unsup_raw(models, X_test_us, device)
    del X_val_us, X_test_us

    # ── 5. 搜索融合权重 ─────────────────────────────────────
    print("[Unsup] 搜索最优融合权重 …")
    w, w_auc = search_weights(if_v, ae_v, vae_v, y_val)
    cfg.unsup_weights = w
    print(f"  权重: IF={w[0]:.2f}  AE={w[1]:.2f}  VAE={w[2]:.2f}  "
          f"(val_auc={w_auc:.4f})")

    # ── 6. 融合无监督分数 (val 作归一化参考) ──────────────────
    us_val = fuse_scores(if_v, ae_v, vae_v, w)
    us_test = fuse_scores(if_t, ae_t, vae_t, w,
                          ref_if=if_v, ref_ae=ae_v, ref_vae=vae_v)

    # ── 7. 校准阈值 ─────────────────────────────────────────
    calibrate(rfp_val, us_val, y_val, cfg)

    # ── 8. 分配告警级别 ─────────────────────────────────────
    print("\n[Predict] 分配四级告警 …")
    lv_val = classify_alerts(rfp_val, us_val, cfg)
    lv_test = classify_alerts(rfp_test, us_test, cfg)

    # ── 9. 评估 ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Eval] ── 验证集 ──")
    val_res = evaluate("Val", y_val, rfp_val, us_val, lv_val, cfg)

    print("\n" + "=" * 60)
    print("[Eval] ── 测试集 ──")
    test_res = evaluate("Test", y_test, rfp_test, us_test, lv_test, cfg,
                        attack_labels=atk_labels)

    # ── 10. 保存输出 ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Save] 写入输出文件 …")

    pred_df = pd.DataFrame({
        cfg.sample_id_col: sample_ids,
        "y_true": y_test,
        "rf_prob": np.round(rfp_test, 6),
        "unsup_score": np.round(us_test, 6),
        "alert_level": [ALERT_NAMES[int(l)] for l in lv_test],
        "alert_level_code": lv_test,
        "final_binary": to_binary(lv_test),
        "risk_score": np.round(0.85 * rfp_test + 0.15 * us_test, 6),
        "detection_source": np.where(
            lv_test == AlertLevel.CRITICAL, "rf+unsup",
            np.where(lv_test == AlertLevel.HIGH, "rf",
                     np.where(lv_test == AlertLevel.MEDIUM,
                              "suspect", "benign"))),
    })
    pred_path = cfg.processed_dir / "test_predictions_ensemble.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"  预测结果:  {pred_path}")

    medium_df = pred_df[pred_df["alert_level"] == "MEDIUM"]
    if len(medium_df) > 0:
        sp = cfg.processed_dir / "suspicious_flows_medium.csv"
        medium_df.to_csv(sp, index=False)
        print(f"  可疑流量 (MEDIUM): {sp}  ({len(medium_df)} 条)")

    ens_cfg_out = {
        "theta_rf": cfg.theta_rf,
        "theta_gray": cfg.theta_gray,
        "theta_unsup_high": cfg.theta_unsup_high,
        "theta_unsup_extreme": cfg.theta_unsup_extreme,
        "unsup_weights": {"IF": w[0], "AE": w[1], "VAE": w[2]},
        "weight_search_val_auc": w_auc,
    }
    cp = cfg.model_dir / "ensemble_config.json"
    cp.write_text(_json(ens_cfg_out, indent=2, ensure_ascii=False), "utf-8")
    print(f"  融合配置:  {cp}")

    all_out = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": ens_cfg_out,
        "validation": val_res,
        "test": test_res,
    }
    mp = cfg.processed_dir / "metrics_ensemble.json"
    mp.write_text(_json(all_out, indent=2, ensure_ascii=False), "utf-8")
    print(f"  评估报告:  {mp}")

    save_ip_summary(cfg, pred_df)

    # ── 最终总结 ─────────────────────────────────────────────
    t = test_res
    bm = t["binary_metrics"]
    ma = t["medium_analysis"]
    ad = t["alert_distribution"]
    print("\n" + "=" * 60)
    print("[Summary] 融合系统测试集最终结果")
    print(f"  二分类: Acc={bm['accuracy']:.4f}  "
          f"Prec={bm['precision']:.4f}  "
          f"Rec={bm['recall']:.4f}  "
          f"F1={bm['f1']:.4f}  "
          f"AUC={bm.get('auc', 0):.4f}")
    print(f"  告警分布: "
          f"CRITICAL={ad['CRITICAL']['count']}  "
          f"HIGH={ad['HIGH']['count']}  "
          f"MEDIUM={ad['MEDIUM']['count']}  "
          f"LOW={ad['LOW']['count']}")
    print(f"  MEDIUM 额外发现的攻击: {ma['medium_attacks']} 条  "
          f"(扩展召回率 {ma['extended_recall']:.6f})")
    print("\n[Ensemble] 全部完成。")


if __name__ == "__main__":
    main()
