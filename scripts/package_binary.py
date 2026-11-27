"""Package a smoke-tested CI executable and instructions for its pinned network."""

import hashlib
from pathlib import Path
import sys
import zipfile

root = Path(__file__).resolve().parent.parent
target = sys.argv[1]
config = dict(
    line.split("=", 1)
    for line in (root / "config/upstream.env").read_text().splitlines()
    if line and not line.startswith("#")
)
exe = "stockcrab.exe" if target.startswith("windows-") else "stockcrab"
network = config["STOCKCRAB_NETWORK"]
command = f".\\{exe} bench" if exe.endswith(".exe") else f"./{exe} bench"
output = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "dist"
output.mkdir(exist_ok=True)
archive = output / f"stockcrab-{target}.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
    package.write(Path(sys.argv[3]) if len(sys.argv) > 3 else root / "target" / "release" / exe, exe)
    for name in ("LICENSE", "AUTHORS"):
        package.write(root / name, name)
    package.writestr("RUN.txt", f"""Stockcrab {target}

Download {network} directly from Stockfish and place it next to {exe}.
https://tests.stockfishchess.org/api/nn/{network}

Network SHA-256
{config["STOCKCRAB_NETWORK_SHA256"]}

Run from the extracted directory.
{command}

Expected Nodes searched: {config["BENCH_NODES"]}
""")

digest = hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n")
