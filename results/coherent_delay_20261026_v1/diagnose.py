"""Summarize preserved inference weights, timing and stopping records."""
import json
from pathlib import Path

import torch

out = Path(__file__).resolve().parent
run = out.parents[1] / "runs" / out.name
settings = json.loads((run / "config_resolved.json").read_text())["probe"]
frozen = json.loads((run / "frozen.json").read_text())
result = []
for cohort in ("new", "historical"):
    for local, snr in enumerate(settings["snrs_db"]):
        saved = torch.load(run / f"predictions/{cohort}_{local}.pt", weights_only=True, map_location="cpu")
        gamma, model = saved["gamma"], saved["model"]
        assert len(gamma) == (32 if cohort == "new" else 16)
        mass = gamma.gather(1, model["indices"]).sum(1) / gamma.sum(1)
        choice = frozen["strength_by_context"][str(local)]
        row = {"cohort": cohort, "snr_db": snr, "scenes": len(gamma),
            "base_nonconverged": int((saved["base_optimality_residual"] > 1e-3).sum()),
            "spatial_mass_mean": float(mass.mean()), "delay_top_mass_mean": float(model["weights"].sum(-1).mean()),
            "base_ms_per_scene": 1000 * saved["timing"]["base_seconds"] / len(gamma),
            "delay_ms_per_scene": 1000 * saved["timing"]["delay_model_seconds"] / len(gamma),
            "selected_posterior_ms_per_scene": 1000 * saved["timing"]["posterior_seconds"]["rho_" + str(choice)] / len(gamma)}
        if cohort == "historical":
            row.update(full_nonconverged=int((~saved["full_sbl_info"]["sbl_converged"]).sum()),
                full_mean_iterations=float(saved["full_sbl_info"]["sbl_iterations"].double().mean()),
                baseline_replay_error=saved["legacy_replay_relative_squared_error"])
        result.append(row)
(out / "mechanism_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
