"""
Patch cicflowmeter 0.5.0 sniffer.py in .venv-pcap after pip install.

1) main() wrong positional args -> fields gets bool (AttributeError: split)
2) Offline AsyncSniffer used BPF filter -> Scapy calls tcpdump/windump on Windows;
   without WinDump, raises Scapy_Exception: tcpdump is not available.
   FlowSession already drops non-TCP/UDP, so offline filter=None is safe.

Run:
  .venv-pcap\\Scripts\\python.exe tools/patch_cicflowmeter_sniffer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PATCHES: list[tuple[str, str, str]] = [
    (
        "main() create_sniffer args",
        """    sniffer, session = create_sniffer(
        args.input_file,
        args.input_interface,
        args.output_mode,
        args.output,
        args.fields,
        args.verbose,
    )""",
        """    sniffer, session = create_sniffer(
        args.input_file,
        args.input_interface,
        args.output_mode,
        args.output,
        input_directory=None,
        fields=args.fields,
        verbose=args.verbose,
    )""",
    ),
    (
        "offline file filter (create_sniffer)",
        """    if input_file:
        sniffer = AsyncSniffer(
            offline=input_file,
            filter="ip and (tcp or udp)",
            prn=session.process,
            store=False,
        )""",
        """    if input_file:
        # Offline: no tcpdump/windump on Windows; FlowSession drops non-TCP/UDP.
        sniffer = AsyncSniffer(
            offline=input_file,
            filter=None,
            prn=session.process,
            store=False,
        )""",
    ),
    (
        "offline filter (process_directory_merged)",
        """            sniffer = AsyncSniffer(
                offline=str(pcap_file),
                filter="ip and (tcp or udp)",
                prn=session.process,
                store=False,
            )""",
        """            sniffer = AsyncSniffer(
                offline=str(pcap_file),
                filter=None,
                prn=session.process,
                store=False,
            )""",
    ),
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    venv_sniffer = root / ".venv-pcap" / "Lib" / "site-packages" / "cicflowmeter" / "sniffer.py"
    if not venv_sniffer.is_file():
        print(f"[patch] not found: {venv_sniffer}")
        sys.exit(1)
    text = venv_sniffer.read_text(encoding="utf-8")
    original = text
    applied = []
    for name, old, new in PATCHES:
        if old in text:
            text = text.replace(old, new, 1)
            applied.append(name)
    if text == original:
        print("[patch] nothing to do (already patched or unexpected upstream content).")
        return
    venv_sniffer.write_text(text, encoding="utf-8")
    print(f"[patch] updated: {venv_sniffer}")
    for a in applied:
        print(f"  - {a}")


if __name__ == "__main__":
    main()
