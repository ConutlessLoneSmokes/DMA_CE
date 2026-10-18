"""Run/resume one declared stage; keep source and configuration fixed within a study ID."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dma_ce.suite import STAGES, Study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--parent", help="Existing pilot data run with exactly matching training/validation physical settings")
    args = parser.parse_args()
    result = Study(ROOT, args.config, args.run_id, args.device, args.parent).run(args.stage)
    print(result)
