"""
cicflowmeter_preprocessing.py
=============================
使用 CICFlowMeter 将 PCAP 转为 flow 特征 CSV（snake_case 列名）。
环境与依赖：Python 3.12+、requirements-pcap.txt；见 PCAP_MANUAL_SETUP_zh.txt。
支持单文件/目录、合并、--max-packets（editcap）；对比 TrafficLabelling：tools/compare_flow_csv_to_trafficlabelling.py。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _get_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    path_sep = os.pathsep
    tcpdump_path = shutil.which("tcpdump")
    if tcpdump_path:
        tcpdump_dir = os.path.dirname(tcpdump_path)
        env["PATH"] = tcpdump_dir + path_sep + env.get("PATH", "")
    else:
        extra_dirs = [r"C:\Tools", r"C:\Tools\WinDump"]
        existing = env.get("PATH", "")
        for d in extra_dirs:
            if os.path.isdir(d) and ((Path(d) / "tcpdump.exe").exists() or (Path(d) / "windump.exe").exists()):
                env["PATH"] = d + path_sep + existing
                break
    return env


def _get_cicflowmeter_cmd() -> list[str]:
    exe_dir = Path(sys.executable).resolve().parent
    if sys.platform == "win32":
        script = exe_dir / "Scripts" / "cicflowmeter.exe"
    else:
        script = exe_dir / "Scripts" / "cicflowmeter"
    if script.exists():
        return [str(script)]
    which = shutil.which("cicflowmeter")
    if which:
        return [which]
    return [sys.executable, "-m", "cicflowmeter"]


@dataclass
class CICFlowMeterConfig:
    pcap_file: Path = PROJECT_ROOT / "rowdata" / "PCAPs" / "Friday-WorkingHours.pcap"
    pcap_dir: Path | None = None
    output_path: Path = PROJECT_ROOT / "processed" / "cicflowmeter_friday.csv"
    merge: bool = False
    progress_poll_sec: float = 2.0
    progress_quiet_sec: float = 30.0
    max_packets: int | None = None
    slice_keep_path: Path | None = None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KiB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MiB"
    return f"{n / 1024**3:.2f} GiB"


def _find_editcap() -> str | None:
    for name in ("editcap", "editcap.exe"):
        w = shutil.which(name)
        if w:
            return w
    if sys.platform == "win32":
        for base in (
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        ):
            cand = Path(base) / "Wireshark" / "editcap.exe"
            if cand.is_file():
                return str(cand)
    return None


def slice_pcap_max_packets(
    pcap_path: Path,
    max_packets: int,
    *,
    keep_path: Path | None = None,
) -> tuple[Path, bool]:
    """
    用 editcap 保留前 max_packets 个包（Wireshark 包号为 1 起）。
    注意：不能用 ``-c N``，那是「按 N 个包拆成多个输出文件」，主 outfile 常为 0 字节。
    正确写法：``editcap -r in out 1-N`` 表示只写入第 1～N 号包。
    """
    if max_packets < 1:
        raise ValueError("max_packets 须 >= 1")
    exe = _find_editcap()
    if not exe:
        raise FileNotFoundError(
            "未找到 editcap（通常随 Wireshark 安装）。请安装 Wireshark 或将 editcap 加入 PATH。"
        )
    if keep_path is not None:
        keep_path = keep_path.resolve()
        keep_path.parent.mkdir(parents=True, exist_ok=True)
        out = keep_path
        is_temp = False
    else:
        fd, tmp = tempfile.mkstemp(suffix=".pcap", prefix="cicflow_slice_")
        os.close(fd)
        out = Path(tmp)
        is_temp = True
    # -r：保留所选包；默认可删除所选包。范围 1-N 为 editcap 的 1-based 包序号。
    cmd = [exe, "-r", str(pcap_path.resolve()), str(out), f"1-{max_packets}"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        if is_temp:
            out.unlink(missing_ok=True)
        msg = (r.stderr or r.stdout or "").strip() or f"exit {r.returncode}"
        raise RuntimeError(f"editcap 失败: {msg}")
    try:
        sz = out.stat().st_size
    except OSError:
        sz = 0
    if sz == 0:
        if is_temp:
            out.unlink(missing_ok=True)
        raise RuntimeError(
            "editcap 输出的 PCAP 为 0 字节。若你曾用错 -c 会得到此现象；"
            "当前已改为 -r 1-N。请确认输入文件非空且 max_packets 合理。"
        )
    print(
        f"[cicflowmeter_preprocessing] 已截取前 {max_packets} 个包 -> {out} ({_human_size(sz)}) "
        f"({'临时' if is_temp else '保留'})",
        flush=True,
    )
    return out, is_temp


def _run_subprocess_monitoring_csv(
    cmd: list[str],
    env: dict[str, str],
    csv_path: Path,
    pcap_size: int,
    *,
    label: str,
    poll_sec: float = 2.0,
    quiet_sec: float = 30.0,
) -> int:
    proc = subprocess.Popen(cmd, env=env)
    t0 = time.monotonic()
    min_print_gap = min(1.0, max(0.2, poll_sec * 0.5))

    def monitor() -> None:
        last_size = -1
        last_flows = -1
        last_print = 0.0
        last_growth = time.monotonic()
        byte_pos = 0
        newlines = 0
        read_ok = True
        while proc.poll() is None:
            time.sleep(poll_sec)
            now = time.monotonic()
            elapsed = now - t0
            if not csv_path.exists():
                if now - last_print >= quiet_sec:
                    print(
                        f"[cicflowmeter_preprocessing] 进度: {label} | "
                        f"尚无输出 CSV（首个 flow 写完前可能较久），已运行 {elapsed:.0f}s",
                        flush=True,
                    )
                    last_print = now
                continue
            try:
                sz = csv_path.stat().st_size
            except OSError:
                continue
            if read_ok and sz > byte_pos:
                try:
                    with open(csv_path, "rb") as f:
                        f.seek(byte_pos)
                        chunk = f.read()
                    newlines += chunk.count(b"\n")
                    byte_pos += len(chunk)
                except OSError:
                    read_ok = False
            flows = max(0, newlines - 1) if newlines else 0
            changed = sz != last_size or flows != last_flows
            if changed:
                last_growth = now
            if changed and (last_print == 0.0 or now - last_print >= min_print_gap):
                parts = [f"CSV {_human_size(sz)}"]
                if read_ok and newlines >= 2:
                    parts.append(f"约 {flows:,} 条 flow")
                elif read_ok and newlines == 1:
                    parts.append("已写表头，尚无数值行")
                elif not read_ok:
                    parts.append("行数暂不可读（输出文件被占用时 Windows 可能无法并行读取）")
                parts.append(f"输入 pcap {_human_size(pcap_size)}")
                print(
                    f"[cicflowmeter_preprocessing] 进度: {label} | "
                    f"{' | '.join(parts)} | 已运行 {elapsed:.0f}s",
                    flush=True,
                )
                last_print = now
                last_size = sz
                last_flows = flows
            elif (
                not changed
                and sz > 0
                and (now - last_growth) >= quiet_sec
                and (now - last_print) >= quiet_sec
            ):
                print(
                    f"[cicflowmeter_preprocessing] 进度: {label} | "
                    f"CSV 已 {_human_size(sz)}，一段时间内未继续增长（可能接近结束或仍在计算），"
                    f"已运行 {elapsed:.0f}s",
                    flush=True,
                )
                last_print = now

    th = threading.Thread(target=monitor, name="cicflowmeter_csv_monitor", daemon=True)
    th.start()
    try:
        return proc.wait()
    except BaseException:
        proc.kill()
        raise
    finally:
        th.join(timeout=poll_sec * 3)


def run_cicflowmeter_single(
    pcap_path: Path,
    csv_path: Path,
    *,
    file_index: int | None = None,
    file_total: int | None = None,
    progress_poll_sec: float = 2.0,
    progress_quiet_sec: float = 30.0,
) -> None:
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP 文件不存在: {pcap_path}")
    _ensure_parent(csv_path)
    sz = pcap_path.stat().st_size
    idx_str = ""
    if file_index is not None and file_total is not None:
        idx_str = f"[{file_index}/{file_total}] "
    print(
        f"[cicflowmeter_preprocessing] 开始 {idx_str}{pcap_path.name} "
        f"({_human_size(sz)}) -> {csv_path.name}",
        flush=True,
    )
    print(
        "[cicflowmeter_preprocessing] 提示: 大 pcap 可能需数十分钟；"
        f"将每约 {progress_poll_sec:.0f}s 根据输出 CSV 体积/flow 数刷新进度。",
        flush=True,
    )
    cmd = _get_cicflowmeter_cmd() + ["-f", str(pcap_path), "-c", str(csv_path)]
    print(f"[cicflowmeter_preprocessing] 命令: {' '.join(cmd)}", flush=True)
    label = f"{pcap_path.name}"
    if idx_str:
        label = f"{idx_str.strip()} {label}"
    t0 = time.monotonic()
    code = _run_subprocess_monitoring_csv(
        cmd,
        _get_subprocess_env(),
        csv_path,
        sz,
        label=label,
        poll_sec=progress_poll_sec,
        quiet_sec=progress_quiet_sec,
    )
    if code != 0:
        raise RuntimeError(f"CICFlowMeter 退出码非 0: {code}")
    dt = time.monotonic() - t0
    print(
        f"[cicflowmeter_preprocessing] 本文件结束，用时 {dt:.0f}s ({dt/60:.1f} min): {csv_path}",
        flush=True,
    )


def run_cicflowmeter_dir(
    pcap_dir: Path,
    output_dir: Path,
    merge: bool = False,
    progress_poll_sec: float = 2.0,
    progress_quiet_sec: float = 30.0,
) -> None:
    if not pcap_dir.is_dir():
        raise FileNotFoundError(f"目录不存在: {pcap_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pcap_files = sorted(pcap_dir.glob("*.pcap"))
    if not pcap_files:
        print(f"[cicflowmeter_preprocessing] 目录下未找到 .pcap 文件: {pcap_dir}")
        return
    total = len(pcap_files)
    print(
        f"[cicflowmeter_preprocessing] 目录模式: 共 {total} 个 pcap，输出目录: {output_dir}",
        flush=True,
    )
    for i, pcap_path in enumerate(pcap_files, start=1):
        csv_path = output_dir / f"{pcap_path.stem}.csv"
        run_cicflowmeter_single(
            pcap_path,
            csv_path,
            file_index=i,
            file_total=total,
            progress_poll_sec=progress_poll_sec,
            progress_quiet_sec=progress_quiet_sec,
        )
    if merge and len(pcap_files) > 1:
        try:
            import pandas as pd
            dfs = []
            for pcap_path in pcap_files:
                csv_path = output_dir / f"{pcap_path.stem}.csv"
                if csv_path.exists():
                    dfs.append(pd.read_csv(csv_path))
            if dfs:
                merged = pd.concat(dfs, ignore_index=True)
                merged_path = output_dir / "merged_flows.csv"
                merged.to_csv(merged_path, index=False)
                print(f"[cicflowmeter_preprocessing] 已合并为: {merged_path}")
        except Exception as e:
            print(f"[cicflowmeter_preprocessing] 合并 CSV 时出错（可忽略）: {e}")


def run_cicflowmeter_preprocessing(config: CICFlowMeterConfig | None = None) -> None:
    if config is None:
        config = CICFlowMeterConfig()
    if config.pcap_dir is not None:
        run_cicflowmeter_dir(
            config.pcap_dir,
            config.output_path,
            merge=config.merge,
            progress_poll_sec=config.progress_poll_sec,
            progress_quiet_sec=config.progress_quiet_sec,
        )
    else:
        pcap_in = config.pcap_file
        temp_slice: Path | None = None
        try:
            if config.max_packets is not None:
                pcap_in, is_tmp = slice_pcap_max_packets(
                    config.pcap_file,
                    config.max_packets,
                    keep_path=config.slice_keep_path,
                )
                if is_tmp:
                    temp_slice = pcap_in
            run_cicflowmeter_single(
                pcap_in,
                config.output_path,
                progress_poll_sec=config.progress_poll_sec,
                progress_quiet_sec=config.progress_quiet_sec,
            )
        finally:
            if temp_slice is not None:
                temp_slice.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CICFlowMeter PCAP -> flow CSV")
    ap.add_argument("--pcap", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--pcap-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--max-packets", type=int, default=None)
    ap.add_argument("--keep-slice", type=Path, default=None)
    ap.add_argument("--progress-poll", type=float, default=2.0)
    ap.add_argument("--progress-quiet", type=float, default=30.0)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    print("[cicflowmeter_preprocessing] 开始使用 CICFlowMeter 处理 PCAP...", flush=True)
    if args.pcap is not None and args.pcap_dir is not None:
        raise SystemExit("不能同时指定 --pcap 与 --pcap-dir")
    if args.max_packets is not None and args.pcap is None:
        raise SystemExit("--max-packets 仅可与 --pcap 同用")
    if args.keep_slice is not None and args.max_packets is None:
        raise SystemExit("--keep-slice 需配合 --max-packets")
    if args.pcap is not None:
        out = args.out_csv or (PROJECT_ROOT / "pcap_processed" / f"{args.pcap.stem}.csv")
        cfg = CICFlowMeterConfig(
            pcap_file=args.pcap.resolve(),
            pcap_dir=None,
            output_path=out.resolve(),
            merge=False,
            progress_poll_sec=args.progress_poll,
            progress_quiet_sec=args.progress_quiet,
            max_packets=args.max_packets,
            slice_keep_path=args.keep_slice.resolve() if args.keep_slice else None,
        )
    elif args.pcap_dir is not None:
        out_dir = args.out_dir or (PROJECT_ROOT / "pcap_processed")
        cfg = CICFlowMeterConfig(
            pcap_file=PROJECT_ROOT / "rowdata" / "PCAPs" / "Friday-WorkingHours.pcap",
            pcap_dir=args.pcap_dir.resolve(),
            output_path=out_dir.resolve(),
            merge=args.merge,
            progress_poll_sec=args.progress_poll,
            progress_quiet_sec=args.progress_quiet,
        )
    else:
        cfg = CICFlowMeterConfig(
            pcap_dir=PROJECT_ROOT / "rowdata" / "PCAPs",
            output_path=PROJECT_ROOT / "pcap_processed",
            merge=args.merge,
            progress_poll_sec=args.progress_poll,
            progress_quiet_sec=args.progress_quiet,
        )
    run_cicflowmeter_preprocessing(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[cicflowmeter_preprocessing] 发生错误: {exc}")
        raise
    else:
        print("[cicflowmeter_preprocessing] 处理完成。")