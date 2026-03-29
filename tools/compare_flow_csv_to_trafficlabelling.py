"""
对比 CICFlowMeter 生成的 flow CSV 与 TrafficLabelling 官方 CSV（列映射、Flow ID 尽力匹配、Label 分布）。
用法: python tools/compare_flow_csv_to_trafficlabelling.py --generated x.csv --reference rowdata/CSVs/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

TL_TO_CM: dict[str, str] = {
    "Source IP": "src_ip", "Source Port": "src_port", "Destination IP": "dst_ip", "Destination Port": "dst_port",
    "Protocol": "protocol", "Timestamp": "timestamp", "Flow Duration": "flow_duration",
    "Total Fwd Packets": "tot_fwd_pkts", "Total Backward Packets": "tot_bwd_pkts",
    "Total Length of Fwd Packets": "totlen_fwd_pkts", "Total Length of Bwd Packets": "totlen_bwd_pkts",
    "Fwd Packet Length Max": "fwd_pkt_len_max", "Fwd Packet Length Min": "fwd_pkt_len_min",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean", "Fwd Packet Length Std": "fwd_pkt_len_std",
    "Bwd Packet Length Max": "bwd_pkt_len_max", "Bwd Packet Length Min": "bwd_pkt_len_min",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean", "Bwd Packet Length Std": "bwd_pkt_len_std",
    "Flow Bytes/s": "flow_byts_s", "Flow Packets/s": "flow_pkts_s",
    "Flow IAT Mean": "flow_iat_mean", "Flow IAT Std": "flow_iat_std", "Flow IAT Max": "flow_iat_max", "Flow IAT Min": "flow_iat_min",
    "Fwd IAT Total": "fwd_iat_tot", "Fwd IAT Mean": "fwd_iat_mean", "Fwd IAT Std": "fwd_iat_std", "Fwd IAT Max": "fwd_iat_max", "Fwd IAT Min": "fwd_iat_min",
    "Bwd IAT Total": "bwd_iat_tot", "Bwd IAT Mean": "bwd_iat_mean", "Bwd IAT Std": "bwd_iat_std", "Bwd IAT Max": "bwd_iat_max", "Bwd IAT Min": "bwd_iat_min",
    "Fwd PSH Flags": "fwd_psh_flags", "Bwd PSH Flags": "bwd_psh_flags", "Fwd URG Flags": "fwd_urg_flags", "Bwd URG Flags": "bwd_urg_flags",
    "Fwd Header Length": "fwd_header_len", "Bwd Header Length": "bwd_header_len",
    "Fwd Packets/s": "fwd_pkts_s", "Bwd Packets/s": "bwd_pkts_s",
    "Min Packet Length": "pkt_len_min", "Max Packet Length": "pkt_len_max", "Packet Length Mean": "pkt_len_mean",
    "Packet Length Std": "pkt_len_std", "Packet Length Variance": "pkt_len_var",
    "FIN Flag Count": "fin_flag_cnt", "SYN Flag Count": "syn_flag_cnt", "RST Flag Count": "rst_flag_cnt",
    "PSH Flag Count": "psh_flag_cnt", "ACK Flag Count": "ack_flag_cnt", "URG Flag Count": "urg_flag_cnt",
    "CWE Flag Count": "cwr_flag_count", "ECE Flag Count": "ece_flag_cnt", "Down/Up Ratio": "down_up_ratio",
    "Average Packet Size": "pkt_size_avg", "Avg Fwd Segment Size": "fwd_seg_size_avg", "Avg Bwd Segment Size": "bwd_seg_size_avg",
    "Fwd Avg Bytes/Bulk": "fwd_byts_b_avg", "Fwd Avg Packets/Bulk": "fwd_pkts_b_avg", "Fwd Avg Bulk Rate": "fwd_blk_rate_avg",
    "Bwd Avg Bytes/Bulk": "bwd_byts_b_avg", "Bwd Avg Packets/Bulk": "bwd_pkts_b_avg", "Bwd Avg Bulk Rate": "bwd_blk_rate_avg",
    "Subflow Fwd Packets": "subflow_fwd_pkts", "Subflow Fwd Bytes": "subflow_fwd_byts",
    "Subflow Bwd Packets": "subflow_bwd_pkts", "Subflow Bwd Bytes": "subflow_bwd_byts",
    "Init_Win_bytes_forward": "init_fwd_win_byts", "Init_Win_bytes_backward": "init_bwd_win_byts",
    "act_data_pkt_fwd": "fwd_act_data_pkts", "min_seg_size_forward": "fwd_seg_size_min",
    "Active Mean": "active_mean", "Active Std": "active_std", "Active Max": "active_max", "Active Min": "active_min",
    "Idle Mean": "idle_mean", "Idle Std": "idle_std", "Idle Max": "idle_max", "Idle Min": "idle_min",
}

def _read_csv(path: Path):
    import pandas as pd
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252")

def _flow_id_variants(sip, dip, sport, dport, proto) -> list[str]:
    sip, dip = str(sip), str(dip)
    sport, dport = int(sport), int(dport)
    proto = int(proto)
    variants = [
        f"{dip}-{sip}-{dport}-{sport}-{proto}",
        f"{dip}-{sip}-{sport}-{dport}-{proto}",
        f"{sip}-{dip}-{sport}-{dport}-{proto}",
        f"{sip}-{dip}-{dport}-{sport}-{proto}",
    ]
    seen: set[str] = set()
    out: list[str] = []
    for x in variants:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--generated", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--max-rows", type=int, default=None)
    args = p.parse_args()
    import pandas as pd
    gen = _read_csv(args.generated)
    ref = _read_csv(args.reference)
    if args.max_rows is not None:
        gen = gen.head(args.max_rows)
        ref = ref.head(args.max_rows)
    gen.columns = gen.columns.astype(str).str.strip()
    ref.columns = ref.columns.astype(str).str.strip()
    print("=== 列名 ===")
    print(f"生成列数: {len(gen.columns)}, 参考列数: {len(ref.columns)}")
    tl_feature_cols = [c for c in ref.columns if c not in ("Flow ID", "Label")]
    mapped_pairs: list[tuple[str, str]] = []
    missing_in_gen: list[str] = []
    for tl in tl_feature_cols:
        cm = TL_TO_CM.get(tl)
        if cm is None:
            missing_in_gen.append(f"(无映射){tl}")
            continue
        if cm in gen.columns:
            mapped_pairs.append((tl, cm))
        else:
            missing_in_gen.append(f"{tl}->{cm}(缺)")
    cm_in_tl = set(TL_TO_CM.values())
    cm_only = [c for c in gen.columns if c not in cm_in_tl]
    print(f"映射且生成表存在: {len(mapped_pairs)}")
    if missing_in_gen:
        print(f"未映射或缺列(前25): {missing_in_gen[:25]}")
    if cm_only:
        print(f"生成表多出列(前20): {cm_only[:20]}")
    if "Flow ID" not in ref.columns:
        print("参考无 Flow ID，结束。")
        return
    need = {"src_ip", "dst_ip", "src_port", "dst_port", "protocol"}
    if not need.issubset(gen.columns):
        print("生成表缺五元组，无法匹配 Flow ID。")
        return
    ref_ids = set(ref["Flow ID"].astype(str).str.strip())
    matched = 0
    labels_matched: list[str] = []
    for _, row in gen.iterrows():
        for fid in _flow_id_variants(row["src_ip"], row["dst_ip"], row["src_port"], row["dst_port"], row["protocol"]):
            if fid in ref_ids:
                matched += 1
                sub = ref.loc[ref["Flow ID"].astype(str).str.strip() == fid]
                if len(sub) and "Label" in ref.columns:
                    labels_matched.append(str(sub["Label"].iloc[0]).strip())
                break
    print("\n=== Flow ID 匹配 ===")
    print(f"生成行: {len(gen)}, 参考行: {len(ref)}")
    print(f"命中: {matched} ({matched / max(len(gen),1):.2%})")
    if labels_matched:
        print(pd.Series(labels_matched).value_counts().to_string())
    print("\n说明: 方向/超时可能与官方不同；有监督标签以参考 CSV 为准。")

if __name__ == "__main__":
    main()
