"""
attack_traceback.py
===================

攻击溯源模块：基于 ensemble_detector 的检测结果，实现三层溯源分析。

第一层 ─ 攻击源 IP 识别与威胁评分
第二层 ─ 攻击路径重建（攻击子图 + 时序路径）
第三层 ─ 证据链关联（每个高危 IP 的详细证据）

输出:
    traceback/attack_ip_ranking.csv      ─ IP 威胁评分排名
    traceback/attack_paths.json          ─ 结构化攻击路径
    traceback/attack_graph.html          ─ 交互式攻击网络图 (pyvis)
    traceback/attack_timeline.png        ─ 攻击时间线热力图
    traceback/attack_summary_report.json ─ 完整溯源报告
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from pyvis.network import Network as PyvisNetwork
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False

PROJECT_ROOT = Path(__file__).resolve().parent

warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class TracebackConfig:
    processed_dir: Path = PROJECT_ROOT / "processed"
    model_dir: Path = PROJECT_ROOT / "models"
    output_dir: Path = PROJECT_ROOT / "traceback"
    sample_id_col: str = "sample_id"

    # 攻击源判定阈值
    attack_ip_ratio_threshold: float = 0.10
    suspect_ip_ratio_threshold: float = 0.001

    # 威胁评分权重
    w_attack_ratio: float = 0.35
    w_graph_centrality: float = 0.20
    w_avg_risk: float = 0.25
    w_alert_severity: float = 0.20

    # 可视化参数
    top_n_ips_timeline: int = 20
    top_n_ips_graph: int = 50
    max_edges_graph: int = 500


# ═══════════════════════════════════════════════════════════════
#  Layer 1 — 攻击源 IP 识别与威胁评分
# ═══════════════════════════════════════════════════════════════

def load_all_data(cfg: TracebackConfig) -> Dict[str, pd.DataFrame]:
    """加载所有溯源所需数据。"""
    data: Dict[str, pd.DataFrame] = {}

    ip_summary_path = cfg.processed_dir / "ip_attack_summary_ensemble.csv"
    data["ip_summary"] = pd.read_csv(ip_summary_path)
    print(f"[Load] ip_attack_summary_ensemble.csv  ({len(data['ip_summary'])} IPs)")

    pred_path = cfg.processed_dir / "test_predictions_ensemble.csv"
    data["predictions"] = pd.read_csv(pred_path)
    print(f"[Load] test_predictions_ensemble.csv   ({len(data['predictions'])} flows)")

    trace_path = cfg.processed_dir / "trace_metadata.csv"
    trace = pd.read_csv(trace_path)
    if "split_binary" in trace.columns:
        trace = trace[trace["split_binary"] == "test"].copy()
    data["trace"] = trace
    print(f"[Load] trace_metadata.csv (test split) ({len(data['trace'])} flows)")

    edges_path = cfg.processed_dir / "trace_graph_edges.csv"
    if edges_path.exists():
        data["edges"] = pd.read_csv(edges_path)
        print(f"[Load] trace_graph_edges.csv           ({len(data['edges'])} edges)")

    return data


def build_flow_detail(data: Dict[str, pd.DataFrame],
                      cfg: TracebackConfig) -> pd.DataFrame:
    """将 trace_metadata 与 predictions join，得到每条流的完整信息。"""
    merged = data["trace"].merge(
        data["predictions"], on=cfg.sample_id_col, how="inner"
    )
    if "Timestamp" in merged.columns:
        merged["Timestamp"] = pd.to_datetime(merged["Timestamp"], errors="coerce")
    print(f"[Data] flow_detail: {len(merged)} flows (trace ∩ predictions)")
    return merged


def compute_graph_centrality(flow_detail: pd.DataFrame,
                             cfg: TracebackConfig) -> Dict[str, float]:
    """在攻击子图上计算 PageRank 作为图中心性指标。"""
    attack_flows = flow_detail[flow_detail["alert_level"].isin(
        ["CRITICAL", "HIGH", "MEDIUM"]
    )].copy()

    if attack_flows.empty or "Source IP" not in attack_flows.columns:
        return {}

    G = nx.DiGraph()
    edge_weights = (
        attack_flows.groupby(["Source IP", "Destination IP"])
        .agg(weight=("risk_score", "sum"))
        .reset_index()
    )
    for _, row in edge_weights.iterrows():
        G.add_edge(row["Source IP"], row["Destination IP"],
                    weight=row["weight"])

    if G.number_of_nodes() == 0:
        return {}

    pr = nx.pagerank(G, weight="weight", alpha=0.85)
    print(f"[Graph] 攻击子图: {G.number_of_nodes()} 节点, "
          f"{G.number_of_edges()} 边")
    return pr


def compute_threat_scores(ip_summary: pd.DataFrame,
                          pagerank: Dict[str, float],
                          cfg: TracebackConfig) -> pd.DataFrame:
    """综合多维指标计算每个 IP 的威胁评分。"""
    df = ip_summary.copy()

    df["alert_severity_score"] = (
        df["critical"] * 3 + df["high"] * 2 + df["medium"] * 1
    ) / (df["total_flows"] * 3).clip(lower=1)

    df["pagerank"] = df["Source IP"].map(pagerank).fillna(0.0)

    max_pr = df["pagerank"].max()
    df["norm_pagerank"] = df["pagerank"] / max_pr if max_pr > 0 else 0.0

    df["threat_score"] = (
        cfg.w_attack_ratio * df["attack_ratio"]
        + cfg.w_graph_centrality * df["norm_pagerank"]
        + cfg.w_avg_risk * df["avg_risk"]
        + cfg.w_alert_severity * df["alert_severity_score"]
    )

    df = df.sort_values("threat_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    # IP 角色标记
    def _role(row):
        if row["attack_ratio"] >= cfg.attack_ip_ratio_threshold:
            return "attacker"
        elif row["attack_ratio"] >= cfg.suspect_ip_ratio_threshold:
            return "suspect"
        else:
            return "normal"

    df["role"] = df.apply(_role, axis=1)

    return df


def print_ip_ranking(ranking: pd.DataFrame) -> None:
    """打印 IP 威胁排名 Top-20。"""
    print("\n" + "=" * 80)
    print("[Layer-1] 攻击源 IP 威胁评分排名 (Top-20)")
    print("-" * 80)
    print(f"  {'Rank':>4s}  {'Source IP':20s}  {'Score':>7s}  {'Role':>8s}  "
          f"{'Ratio':>7s}  {'Flows':>7s}  {'C':>6s}  {'H':>6s}  {'M':>4s}")
    print("-" * 80)
    for _, r in ranking.head(20).iterrows():
        print(f"  {int(r['rank']):4d}  {r['Source IP']:20s}  "
              f"{r['threat_score']:.4f}  {r['role']:>8s}  "
              f"{r['attack_ratio']:.4f}  {int(r['total_flows']):7d}  "
              f"{int(r['critical']):6d}  {int(r['high']):6d}  "
              f"{int(r['medium']):4d}")

    n_attacker = (ranking["role"] == "attacker").sum()
    n_suspect = (ranking["role"] == "suspect").sum()
    print(f"\n  攻击者 IP: {n_attacker},  可疑 IP: {n_suspect},  "
          f"正常 IP: {len(ranking) - n_attacker - n_suspect}")


# ═══════════════════════════════════════════════════════════════
#  Layer 2 — 攻击路径重建
# ═══════════════════════════════════════════════════════════════

def build_attack_graph(flow_detail: pd.DataFrame) -> nx.DiGraph:
    """构建完整的攻击加权有向图。"""
    attack = flow_detail[flow_detail["alert_level"].isin(
        ["CRITICAL", "HIGH", "MEDIUM"]
    )].copy()

    G = nx.DiGraph()

    if attack.empty:
        return G

    edges = (
        attack.groupby(["Source IP", "Destination IP", "alert_level"])
        .agg(
            flow_count=("sample_id", "count"),
            avg_risk=("risk_score", "mean"),
            avg_rf=("rf_prob", "mean"),
        )
        .reset_index()
    )

    for _, row in edges.iterrows():
        src, dst = row["Source IP"], row["Destination IP"]
        if G.has_edge(src, dst):
            G[src][dst]["flow_count"] += row["flow_count"]
            G[src][dst]["alerts"][row["alert_level"]] = (
                G[src][dst]["alerts"].get(row["alert_level"], 0)
                + row["flow_count"]
            )
        else:
            G.add_edge(
                src, dst,
                flow_count=row["flow_count"],
                avg_risk=row["avg_risk"],
                avg_rf=row["avg_rf"],
                alerts={row["alert_level"]: row["flow_count"]},
            )

    return G


def extract_attack_campaigns(flow_detail: pd.DataFrame,
                             ranking: pd.DataFrame) -> List[Dict[str, Any]]:
    """对每个攻击者 IP，按攻击类型提取攻击战役（campaign）。"""
    attackers = ranking[ranking["role"] == "attacker"]["Source IP"].tolist()

    attack_flows = flow_detail[
        (flow_detail["Source IP"].isin(attackers))
        & (flow_detail["alert_level"].isin(["CRITICAL", "HIGH", "MEDIUM"]))
    ].copy()

    if attack_flows.empty:
        return []

    campaigns: List[Dict[str, Any]] = []

    for src_ip in attackers:
        ip_flows = attack_flows[attack_flows["Source IP"] == src_ip]
        ip_rank = ranking[ranking["Source IP"] == src_ip].iloc[0]

        ip_campaigns: List[Dict[str, Any]] = []
        label_col = "Label" if "Label" in ip_flows.columns else None

        if label_col:
            for label, grp in ip_flows.groupby(label_col):
                if str(label).strip().upper() == "BENIGN":
                    continue

                targets = (
                    grp.groupby("Destination IP")
                    .agg(
                        flows=("sample_id", "count"),
                        critical=pd.NamedAgg(
                            "alert_level", lambda x: (x == "CRITICAL").sum()),
                        high=pd.NamedAgg(
                            "alert_level", lambda x: (x == "HIGH").sum()),
                        medium=pd.NamedAgg(
                            "alert_level", lambda x: (x == "MEDIUM").sum()),
                    )
                    .reset_index()
                    .sort_values("flows", ascending=False)
                )

                start_time = end_time = None
                if "Timestamp" in grp.columns:
                    ts = grp["Timestamp"].dropna()
                    if not ts.empty:
                        start_time = str(ts.min())
                        end_time = str(ts.max())

                target_list = []
                for _, t in targets.iterrows():
                    target_list.append({
                        "ip": t["Destination IP"],
                        "flows": int(t["flows"]),
                        "alert_levels": {
                            "CRITICAL": int(t["critical"]),
                            "HIGH": int(t["high"]),
                            "MEDIUM": int(t["medium"]),
                        },
                    })

                paths = [f"{src_ip} -> {t['ip']}" for t in target_list]

                ip_campaigns.append({
                    "attack_type": str(label),
                    "total_flows": int(grp.shape[0]),
                    "start_time": start_time,
                    "end_time": end_time,
                    "targets": target_list,
                    "paths": paths,
                })
        else:
            targets = (
                ip_flows.groupby("Destination IP")
                .agg(flows=("sample_id", "count"))
                .reset_index()
                .sort_values("flows", ascending=False)
            )
            ip_campaigns.append({
                "attack_type": "unknown",
                "total_flows": int(ip_flows.shape[0]),
                "targets": [
                    {"ip": t["Destination IP"], "flows": int(t["flows"])}
                    for _, t in targets.iterrows()
                ],
            })

        campaigns.append({
            "ip": src_ip,
            "threat_score": float(ip_rank["threat_score"]),
            "role": "attacker",
            "total_attack_flows": int(ip_flows.shape[0]),
            "campaigns": ip_campaigns,
        })

    return campaigns


def print_attack_paths(campaigns: List[Dict[str, Any]]) -> None:
    """打印攻击路径摘要。"""
    print("\n" + "=" * 80)
    print("[Layer-2] 攻击路径重建")
    print("=" * 80)
    for c in campaigns:
        print(f"\n  攻击源: {c['ip']}  "
              f"(threat_score={c['threat_score']:.4f}, "
              f"总攻击流={c['total_attack_flows']})")
        for camp in c["campaigns"]:
            at = camp["attack_type"]
            safe_at = at.encode("ascii", errors="replace").decode("ascii")
            print(f"    [{safe_at}]  flows={camp['total_flows']}", end="")
            if camp.get("start_time"):
                print(f"  period: {camp['start_time']} ~ {camp['end_time']}")
            else:
                print()
            for t in camp.get("targets", [])[:5]:
                print(f"      -> {t['ip']}  ({t['flows']} flows)")
            if len(camp.get("targets", [])) > 5:
                print(f"      ... 及其余 {len(camp['targets']) - 5} 个目标")


# ═══════════════════════════════════════════════════════════════
#  Layer 3 — 证据链关联
# ═══════════════════════════════════════════════════════════════

def build_evidence_chains(flow_detail: pd.DataFrame,
                          ranking: pd.DataFrame,
                          cfg: TracebackConfig) -> List[Dict[str, Any]]:
    """为每个攻击者 / 可疑 IP 构建证据链。"""
    target_ips = ranking[
        ranking["role"].isin(["attacker", "suspect"])
    ]["Source IP"].tolist()

    evidence_list: List[Dict[str, Any]] = []

    for ip in target_ips:
        ip_flows = flow_detail[flow_detail["Source IP"] == ip]
        ip_rank = ranking[ranking["Source IP"] == ip].iloc[0]

        alert_dist = ip_flows["alert_level"].value_counts().to_dict()

        attack_types = []
        if "Label" in ip_flows.columns:
            labels = ip_flows[ip_flows["alert_level"].isin(
                ["CRITICAL", "HIGH", "MEDIUM"]
            )]["Label"].value_counts()
            attack_types = [
                {"type": str(k), "count": int(v)}
                for k, v in labels.items()
                if str(k).strip().upper() != "BENIGN"
            ]

        time_evidence = {}
        if "Timestamp" in ip_flows.columns:
            ts = ip_flows["Timestamp"].dropna()
            if not ts.empty:
                time_evidence["first_seen"] = str(ts.min())
                time_evidence["last_seen"] = str(ts.max())
                attack_ts = ip_flows[
                    ip_flows["alert_level"].isin(["CRITICAL", "HIGH"])
                ]["Timestamp"].dropna()
                if not attack_ts.empty:
                    time_evidence["attack_start"] = str(attack_ts.min())
                    time_evidence["attack_end"] = str(attack_ts.max())
                    peak_hour = attack_ts.dt.hour.mode()
                    if len(peak_hour) > 0:
                        time_evidence["peak_attack_hour"] = int(peak_hour.iloc[0])

        targets = ip_flows[ip_flows["alert_level"].isin(
            ["CRITICAL", "HIGH", "MEDIUM"]
        )]["Destination IP"].value_counts().head(10)
        target_summary = [
            {"ip": str(k), "attack_flows": int(v)}
            for k, v in targets.items()
        ]

        evidence_list.append({
            "source_ip": ip,
            "role": str(ip_rank["role"]),
            "threat_score": float(ip_rank["threat_score"]),
            "total_flows": int(ip_rank["total_flows"]),
            "attack_flows": int(ip_rank["attack_flows"]),
            "attack_ratio": float(ip_rank["attack_ratio"]),
            "alert_distribution": {
                str(k): int(v) for k, v in alert_dist.items()
            },
            "attack_types": attack_types,
            "time_evidence": time_evidence,
            "top_targets": target_summary,
            "avg_risk_score": float(ip_rank["avg_risk"]),
            "avg_rf_prob": float(ip_rank["avg_rf_prob"]),
            "avg_unsup_score": float(ip_rank["avg_unsup"]),
        })

    print(f"\n[Layer-3] 为 {len(evidence_list)} 个 IP 构建了证据链")
    return evidence_list


# ═══════════════════════════════════════════════════════════════
#  Visualization — 交互式攻击图 (pyvis)
# ═══════════════════════════════════════════════════════════════

def generate_interactive_graph(flow_detail: pd.DataFrame,
                               ranking: pd.DataFrame,
                               cfg: TracebackConfig) -> Optional[Path]:
    """生成 pyvis 交互式 HTML 攻击网络图。"""
    if not HAS_PYVIS:
        print("[Viz] pyvis 未安装，跳过交互式图生成。")
        return None

    attack = flow_detail[flow_detail["alert_level"].isin(
        ["CRITICAL", "HIGH", "MEDIUM"]
    )].copy()

    if attack.empty:
        print("[Viz] 无攻击流量，跳过图生成。")
        return None

    ip_roles = dict(zip(ranking["Source IP"], ranking["role"]))
    ip_scores = dict(zip(ranking["Source IP"], ranking["threat_score"]))

    edge_agg = (
        attack.groupby(["Source IP", "Destination IP"])
        .agg(
            flow_count=("sample_id", "count"),
            critical=pd.NamedAgg(
                "alert_level", lambda x: (x == "CRITICAL").sum()),
            high=pd.NamedAgg(
                "alert_level", lambda x: (x == "HIGH").sum()),
            medium=pd.NamedAgg(
                "alert_level", lambda x: (x == "MEDIUM").sum()),
            avg_risk=("risk_score", "mean"),
        )
        .reset_index()
        .sort_values("flow_count", ascending=False)
    )

    edge_agg = edge_agg.head(cfg.max_edges_graph)

    all_ips = set(edge_agg["Source IP"]) | set(edge_agg["Destination IP"])

    net = PyvisNetwork(
        height="800px", width="100%", directed=True,
        bgcolor="#1a1a2e", font_color="white",
        notebook=False,
    )
    net.barnes_hut(
        gravity=-5000, central_gravity=0.3,
        spring_length=200, spring_strength=0.01,
    )

    role_colors = {
        "attacker": "#e74c3c",
        "suspect": "#f39c12",
        "victim": "#3498db",
        "normal": "#95a5a6",
    }

    for ip in all_ips:
        role = ip_roles.get(ip, "normal")
        score = ip_scores.get(ip, 0.0)

        in_attack_flows = edge_agg[edge_agg["Destination IP"] == ip]["flow_count"].sum()
        out_attack_flows = edge_agg[edge_agg["Source IP"] == ip]["flow_count"].sum()

        if role == "normal" and in_attack_flows > out_attack_flows:
            color = role_colors["victim"]
            display_role = "victim"
        else:
            color = role_colors.get(role, role_colors["normal"])
            display_role = role

        size = max(10, min(60, 10 + math.log2(max(1, out_attack_flows + in_attack_flows)) * 5))

        title = (
            f"<b>{ip}</b><br>"
            f"Role: {display_role}<br>"
            f"Threat Score: {score:.4f}<br>"
            f"Outgoing Attack Flows: {out_attack_flows}<br>"
            f"Incoming Attack Flows: {in_attack_flows}"
        )

        net.add_node(
            ip, label=ip, title=title,
            color=color, size=size,
            borderWidth=2,
            borderWidthSelected=4,
        )

    for _, row in edge_agg.iterrows():
        src, dst = row["Source IP"], row["Destination IP"]
        fc = int(row["flow_count"])
        width = max(1, min(10, math.log2(max(1, fc)) * 1.5))

        cr = row["critical"] / max(fc, 1)
        if cr > 0.5:
            edge_color = "#e74c3c"
        elif cr > 0.1:
            edge_color = "#f39c12"
        else:
            edge_color = "#7f8c8d"

        title = (
            f"{src} → {dst}<br>"
            f"Flows: {fc}<br>"
            f"C={int(row['critical'])} H={int(row['high'])} M={int(row['medium'])}<br>"
            f"Avg Risk: {row['avg_risk']:.4f}"
        )

        net.add_edge(
            src, dst,
            value=width, title=title,
            color=edge_color, arrows="to",
        )

    legend_html = """
    <div style="position:fixed;top:10px;right:10px;background:rgba(0,0,0,0.7);
                padding:15px;border-radius:8px;color:white;font-family:monospace;
                z-index:1000;font-size:13px;">
        <b>Attack Network Graph</b><br><br>
        <span style="color:#e74c3c;">●</span> Attacker<br>
        <span style="color:#f39c12;">●</span> Suspect<br>
        <span style="color:#3498db;">●</span> Victim<br>
        <span style="color:#95a5a6;">●</span> Normal<br><br>
        <b>Edge Color</b><br>
        <span style="color:#e74c3c;">━</span> High CRITICAL ratio<br>
        <span style="color:#f39c12;">━</span> Mixed alerts<br>
        <span style="color:#7f8c8d;">━</span> Mostly MEDIUM/HIGH
    </div>
    """

    out_path = cfg.output_dir / "attack_graph.html"
    net.save_graph(str(out_path))

    html_content = out_path.read_text(encoding="utf-8")
    html_content = html_content.replace("</body>", legend_html + "</body>")
    out_path.write_text(html_content, encoding="utf-8")

    print(f"[Viz] 交互式攻击图: {out_path}")
    print(f"  节点: {len(all_ips)}, 边: {len(edge_agg)}")
    return out_path


# ═══════════════════════════════════════════════════════════════
#  Visualization — 攻击时间线热力图
# ═══════════════════════════════════════════════════════════════

def generate_timeline_heatmap(flow_detail: pd.DataFrame,
                              ranking: pd.DataFrame,
                              cfg: TracebackConfig) -> Optional[Path]:
    """生成 Top-N 攻击 IP 的时间线热力图。"""
    if "Timestamp" not in flow_detail.columns:
        print("[Viz] 无 Timestamp 列，跳过时间线图。")
        return None

    attack = flow_detail[flow_detail["alert_level"].isin(
        ["CRITICAL", "HIGH", "MEDIUM"]
    )].copy()

    if attack.empty or attack["Timestamp"].isna().all():
        print("[Viz] 无有效时间戳的攻击流量，跳过。")
        return None

    attack = attack.dropna(subset=["Timestamp"])

    top_ips = ranking[ranking["role"].isin(["attacker", "suspect"])].head(
        cfg.top_n_ips_timeline
    )["Source IP"].tolist()

    if not top_ips:
        print("[Viz] 无攻击/可疑 IP，跳过时间线图。")
        return None

    attack = attack[attack["Source IP"].isin(top_ips)].copy()
    if attack.empty:
        return None

    attack["hour_bin"] = attack["Timestamp"].dt.floor("H")

    pivot = (
        attack.groupby(["Source IP", "hour_bin"])
        .size()
        .reset_index(name="count")
        .pivot(index="Source IP", columns="hour_bin", values="count")
        .fillna(0)
    )

    ip_order = [ip for ip in top_ips if ip in pivot.index]
    pivot = pivot.loc[ip_order]

    fig_height = max(4, len(ip_order) * 0.5 + 2)
    fig_width = max(12, pivot.shape[1] * 0.3 + 3)
    fig, ax = plt.subplots(figsize=(min(fig_width, 24), min(fig_height, 14)))

    log_pivot = np.log1p(pivot)

    sns.heatmap(
        log_pivot, ax=ax, cmap="YlOrRd", linewidths=0.3,
        cbar_kws={"label": "log(1 + flow_count)"},
        xticklabels=True, yticklabels=True,
    )

    ax.set_title("Attack Timeline Heatmap (Top Attacker/Suspect IPs)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Source IP", fontsize=11)

    xtick_labels = [ts.strftime("%m-%d %H:%M") if hasattr(ts, 'strftime')
                    else str(ts) for ts in pivot.columns]
    step = max(1, len(xtick_labels) // 20)
    display_labels = [l if i % step == 0 else ""
                      for i, l in enumerate(xtick_labels)]
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

    plt.tight_layout()
    out_path = cfg.output_dir / "attack_timeline.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[Viz] 时间线热力图: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════
#  Output — 保存所有结果
# ═══════════════════════════════════════════════════════════════

def _json_serial(obj):
    """JSON 序列化辅助。"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_all_outputs(ranking: pd.DataFrame,
                     campaigns: List[Dict[str, Any]],
                     evidence: List[Dict[str, Any]],
                     graph: nx.DiGraph,
                     cfg: TracebackConfig) -> None:
    """保存所有输出文件。"""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("[Save] 写入输出文件 …")

    # 1. IP 排名
    ranking_path = cfg.output_dir / "attack_ip_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    print(f"  IP 排名: {ranking_path}")

    # 2. 攻击路径 JSON
    paths_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "attack_sources": campaigns,
        "graph_stats": {
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "attacker_count": sum(1 for c in campaigns if c["role"] == "attacker"),
        },
    }
    paths_path = cfg.output_dir / "attack_paths.json"
    paths_path.write_text(
        json.dumps(paths_data, indent=2, ensure_ascii=False, default=_json_serial),
        encoding="utf-8",
    )
    print(f"  攻击路径: {paths_path}")

    # 3. 完整溯源报告
    n_attacker = sum(1 for e in evidence if e["role"] == "attacker")
    n_suspect = sum(1 for e in evidence if e["role"] == "suspect")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_ips_analyzed": len(ranking),
            "attacker_ips": n_attacker,
            "suspect_ips": n_suspect,
            "total_attack_edges": graph.number_of_edges(),
        },
        "ip_ranking_top10": [
            {
                "rank": int(r["rank"]),
                "ip": r["Source IP"],
                "threat_score": float(r["threat_score"]),
                "role": r["role"],
                "attack_ratio": float(r["attack_ratio"]),
            }
            for _, r in ranking.head(10).iterrows()
        ],
        "evidence_chains": evidence,
    }
    report_path = cfg.output_dir / "attack_summary_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_serial),
        encoding="utf-8",
    )
    print(f"  溯源报告: {report_path}")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    pa = argparse.ArgumentParser(description="攻击溯源分析模块")
    pa.add_argument("--processed-dir", type=Path,
                    default=PROJECT_ROOT / "processed")
    pa.add_argument("--model-dir", type=Path,
                    default=PROJECT_ROOT / "models")
    pa.add_argument("--output-dir", type=Path,
                    default=PROJECT_ROOT / "traceback")
    args = pa.parse_args()

    cfg = TracebackConfig(
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
    )

    print("=" * 80)
    print("[Attack Traceback] 开始攻击溯源分析")
    print("=" * 80)

    # ── 数据加载 ─────────────────────────────────
    data = load_all_data(cfg)
    flow_detail = build_flow_detail(data, cfg)

    # ── Layer 1: IP 威胁评分 ─────────────────────
    print("\n[Layer-1] 计算 IP 威胁评分 …")
    pagerank = compute_graph_centrality(flow_detail, cfg)
    ranking = compute_threat_scores(data["ip_summary"], pagerank, cfg)
    print_ip_ranking(ranking)

    # ── Layer 2: 攻击路径重建 ────────────────────
    print("\n[Layer-2] 重建攻击路径 …")
    attack_graph = build_attack_graph(flow_detail)
    campaigns = extract_attack_campaigns(flow_detail, ranking)
    print_attack_paths(campaigns)

    # ── Layer 3: 证据链 ──────────────────────────
    print("\n[Layer-3] 构建证据链 …")
    evidence = build_evidence_chains(flow_detail, ranking, cfg)

    # ── 保存结果 ─────────────────────────────────
    save_all_outputs(ranking, campaigns, evidence, attack_graph, cfg)

    # ── 可视化 ───────────────────────────────────
    print("\n[Viz] 生成可视化 …")
    generate_interactive_graph(flow_detail, ranking, cfg)
    generate_timeline_heatmap(flow_detail, ranking, cfg)

    # ── 最终总结 ─────────────────────────────────
    n_attacker = (ranking["role"] == "attacker").sum()
    n_suspect = (ranking["role"] == "suspect").sum()
    print("\n" + "=" * 80)
    print("[Summary] 攻击溯源分析完成")
    print(f"  攻击者 IP: {n_attacker}")
    print(f"  可疑 IP:   {n_suspect}")
    print(f"  攻击子图:  {attack_graph.number_of_nodes()} 节点, "
          f"{attack_graph.number_of_edges()} 边")
    print(f"  输出目录:  {cfg.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
