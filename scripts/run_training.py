import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from risk_model.cli import run_cli


def main():
    argv = sys.argv[1:]
    if len(argv) >= 2 and not argv[1].startswith("-"):
        argv = [argv[0], "--config", argv[1], *argv[2:]]
    raise SystemExit(run_cli(argv))


if __name__ == "__main__":
    main()
