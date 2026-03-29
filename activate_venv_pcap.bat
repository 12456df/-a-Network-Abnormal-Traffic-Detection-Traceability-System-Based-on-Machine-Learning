@echo off
REM PCAP 虚拟环境一键激活（不触发 PowerShell 执行策略）
cd /d "%~dp0"
if not exist ".venv-pcap\Scripts\activate.bat" (
    echo [.venv-pcap] not found. Run: py -3.12 -m venv .venv-pcap
    exit /b 1
)
call ".venv-pcap\Scripts\activate.bat"
echo [OK] .venv-pcap activated ^(Python 3.12^)
python --version
