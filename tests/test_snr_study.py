import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dma_ce.data import prepare
from dma_ce.experiment import file_hash
from dma_ce.snr_study import summarize_results, sweep_cache, train_network, validate_comparison, verify_frozen_models


class SNRStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.config = json.loads((ROOT / "configs/research_smoke.json").read_text())
        cls.parent = prepare(cls.config, "cpu", lambda *a, **k: None, split_names=("train", "validation"))

    def test_snr_sweep_preserves_parent_and_pairs_noise(self):
        original = self.parent["splits"]["validation"]
        sweep = sweep_cache(self.config, self.parent, "validation", 2, [-10., 10., 30.])
        self.assertEqual(sweep["channel_scale"], self.parent["channel_scale"])
        for entry, context in zip(sweep["splits"]["validation"], sweep["contexts"]):
            reference = next(e for e, c in zip(original, self.parent["contexts"])
                if c["bandwidth_hz"] == context["bandwidth_hz"] and c["configurations"] == context["configurations"])
            self.assertTrue(torch.equal(reference["truth"][:2], entry["truth"]))
            if context["snr_db"] == 10.:
                self.assertTrue(torch.equal(reference["observation"][:2], entry["observation"]))
            self.assertEqual(context["noise_complex_variance"], context["training_white_power"] / 10 ** (context["snr_db"] / 10))
        for start in range(0, len(sweep["contexts"]), 3):
            standards = []
            for c, e in zip(sweep["contexts"][start:start + 3], sweep["splits"]["validation"][start:start + 3]):
                clean = torch.einsum("kjn,bkn->bkj", c["matrix"], e["truth"])
                standards.append((e["observation"] - clean) / np.sqrt(c["noise_complex_variance"]))
            self.assertTrue(torch.allclose(standards[0], standards[-1], atol=3e-5, rtol=3e-5))

    def test_summary_averages_linear_errors_without_replicating_baselines(self):
        spec = json.loads((ROOT / "configs/snr_comparison_smoke.json").read_text())
        spec.update(screen_scenes=2, seeds=[201, 202])
        rows = []
        for method in ("prior_learned", "prior_fixed", "lmmse_joint", "sbl_dense", "ddpm_latest_avg16", "deterministic_latest"):
            seeds = [-1] if method in ("lmmse_joint", "sbl_dense") else spec["seeds"]
            for seed in seeds:
                for scene, error in enumerate((.01, 1.)):
                    rows.append({"context": 0, "method": method, "scene": scene, "seed": seed,
                        "observation_sha256": str(scene), "nmse_linear": error, "bandwidth_hz": 3e9,
                        "configurations": 8, "snr_db": 10., "total_batch_ms_per_sample": 1.,
                        "network_calls": 3, "solver_applicable": method == "sbl_dense", "sbl_converged": True})
        with tempfile.TemporaryDirectory() as folder:
            summary = summarize_results(rows, spec, Path(folder))
            self.assertTrue(all(abs(r["mean_nmse_db"] - 10 * np.log10(.505)) < 1e-10 for r in summary))
            self.assertEqual(next(r for r in summary if r["method"] == "sbl_dense")["training_seeds"], 0)
            self.assertEqual(next(r for r in summary if r["method"] == "prior_learned")["training_seeds"], 2)
            corrupted = copy.deepcopy(rows)
            corrupted[-1]["observation_sha256"] = "different"
            with self.assertRaises(AssertionError):
                summarize_results(corrupted, spec, Path(folder))

    def test_comparison_configuration_rejects_invalid_budgets(self):
        spec = json.loads((ROOT / "configs/snr_comparison_smoke.json").read_text())
        validate_comparison(spec)
        for key, value in (("snrs_db", [10., -10.]), ("snrs_db", [float("nan")]),
                ("seeds", [201, 201]), ("prior_checkpoints", [4, 8]), ("screen_scenes", True),
                ("sbl_tolerance", 0.), ("strength_grid", [-1.])):
            invalid = {**spec, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_comparison(invalid)

    def test_training_resume_retains_optimizer_rng_and_checkpoint_selection(self):
        spec = json.loads((ROOT / "configs/snr_comparison_smoke.json").read_text())
        spec.update(prior_steps=8, prior_checkpoints=[2, 4, 8])
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            for interrupted in (False, True):
                torch.manual_seed(81)
                model = torch.nn.Linear(2, 1)
                calls = [0]
                def loss(batch_rng, noise_rng):
                    calls[0] += 1
                    if interrupted and calls[0] == 5:
                        raise RuntimeError("simulated interruption")
                    x = torch.randn(3, 2, generator=batch_rng)
                    target = torch.randn(3, 1, generator=noise_rng)
                    return (model(x) - target).square().mean()
                folder = destination / str(interrupted)
                arguments = (model, folder, spec, 81, "prior", loss,
                    lambda: float(model.weight.square().sum()), torch.device("cpu"), lambda *a, **k: None)
                if interrupted:
                    with self.assertRaisesRegex(RuntimeError, "simulated"):
                        train_network(*arguments)
                    saved = torch.load(folder / "last.pt", weights_only=True)
                    self.assertEqual(saved["step"], 4)
                    self.assertEqual(json.loads((folder / "validation.json").read_text())[-1]["step"], 4)
                train_network(*arguments)
            full = torch.load(destination / "False/last.pt", weights_only=True)
            resumed = torch.load(destination / "True/last.pt", weights_only=True)
            self.assertTrue(all(torch.equal(v, resumed["state_dict"][k]) for k, v in full["state_dict"].items()))
            self.assertTrue(torch.equal(full["batch_rng"], resumed["batch_rng"]))
            self.assertTrue(torch.equal(full["noise_rng"], resumed["noise_rng"]))
            self.assertEqual((destination / "False/validation.json").read_bytes(), (destination / "True/validation.json").read_bytes())

    def test_all_frozen_models_are_checked_even_when_predictions_can_be_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "models").mkdir()
            frozen = {"models": {}, "latest_models": {}}
            for seed in (201, 202):
                item = {}
                for kind in ("prior", "update"):
                    path = root / "models" / f"{kind}_{seed}.pt"
                    path.write_bytes(bytes([seed]))
                    item.update({kind + "_path": path.relative_to(root).as_posix(), kind + "_sha256": file_hash(path)})
                frozen["models"][str(seed)] = item
            verify_frozen_models(root, frozen)
            (root / "models/update_202.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                verify_frozen_models(root, frozen)


if __name__ == "__main__":
    unittest.main()
