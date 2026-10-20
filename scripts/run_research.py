"""Run/resume one declared stage; keep source and configuration fixed within a study ID."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dma_ce.suite import STAGES, Study, TrainingBudgetBoundary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--continue-from", help="Copy a validation-only study and extend diffusion from its optimizer/RNG state")
    parser.add_argument("--diffusion-checkpoints", type=int, nargs="+", help="Additional checkpoints above the parent's maximum")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--parent", help="Existing pilot data run with exactly matching training/validation physical settings")
    args = parser.parse_args()
    try:
        result = Study(ROOT, args.config, args.run_id, args.device, args.parent,
            args.continue_from, args.diffusion_checkpoints).run(args.stage)
    except TrainingBudgetBoundary as error:
        print(f"Budget limited: {error}. Validation results are saved; extend the solver/training budget in a new study before final evaluation.", file=sys.stderr)
        sys.exit(2)
    print(result)
