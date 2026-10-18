import copy
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dma_ce.conditional import ConditionalCNN, reliability_features
from dma_ce.data import calibrate, prepare, reestimate
from dma_ce.research import communication_rates, learning_rate, loss_weights, matched_width, paired_effects, recipe, train_job
from dma_ce.suite import Study, stable_budget, validate_study


class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        cls.config = json.loads((ROOT / "configs/research_smoke.json").read_text())
        cls.cache = prepare(cls.config, "cpu", lambda *a, **k: None, split_names=("train", "validation"))

    def test_per_tone_has_no_cross_frequency_dependency(self):
        torch.manual_seed(19)
        model = ConditionalCNN(8, "per_tone").eval()
        state, coarse, reliable = torch.randn(2, 2, 7, 5), torch.randn(2, 2, 7, 5), torch.randn(2, 1, 7, 5)
        before = model(state, coarse, reliable, torch.zeros(2))
        coarse[..., 2] += 20
        after = model(state, coarse, reliable, torch.zeros(2))
        self.assertTrue(torch.equal(before[..., [0, 1, 3, 4]], after[..., [0, 1, 3, 4]]))
        self.assertGreater(float((before[..., 2] - after[..., 2]).detach().abs().max()), 1e-5)
        target = sum(p.numel() for p in ConditionalCNN(32).parameters())
        width = matched_width(32, "per_tone")
        actual = sum(p.numel() for p in ConditionalCNN(width, "per_tone").parameters())
        self.assertLess(abs(actual / target - 1), .04)

    def test_scalar_averages_variance_before_log(self):
        diagonal = torch.tensor([[[1., 9.], [1., 9.]]])
        constants = {"reliability_scale": 1., "reliability_log_mean": 0., "reliability_log_std": 1.}
        actual = reliability_features(diagonal, constants, "scalar")
        self.assertTrue(torch.allclose(actual, torch.full_like(actual, np.log10(5))))
        self.assertEqual(float(reliability_features(diagonal, constants, "none").abs().max()), 0.)

    def test_condition_loss_equalizes_training_residual_energy(self):
        entries = [{"truth": torch.ones(2, 2, 2) * value, "coarse": torch.zeros(2, 2, 2)} for value in (1., 4.)]
        weights = loss_weights(entries, "condition")
        self.assertAlmostEqual(float(weights[0]), float(weights[1]) * 16)
        self.assertTrue(torch.equal(loss_weights(entries, "uniform"), torch.ones(2)))

    def test_reestimation_preserves_observations_and_recalibrates_on_train(self):
        config = copy.deepcopy(self.config)
        config["sbl"].update(angle_points=9, range_points=3, iterations=10)
        changed = reestimate(config, self.cache, "cpu", lambda *a, **k: None)
        self.assertEqual(changed["geometry_hashes"], self.cache["geometry_hashes"])
        for split in changed["splits"]:
            for a, b in zip(changed["splits"][split], self.cache["splits"][split]):
                for key in ("truth", "observation", "variance"):
                    self.assertTrue(torch.equal(a[key], b[key]))
        train = changed["splits"]["train"]
        errors = torch.stack([(e["truth"] - e["coarse"]).abs().square().mean() for e in train]).mean()
        predicted = torch.stack([e["reliability"].mean() for e in train]).mean()
        self.assertAlmostEqual(changed["reliability_scale"], float(errors / predicted), places=6)

    def test_calibration_exceeds_torch_quantile_element_limit(self):
        count = 2 ** 24 + 1
        entry = {"truth": torch.ones(1).expand(1, 1, count), "coarse": torch.zeros(1).expand(1, 1, count),
            "reliability": torch.ones(1).expand(1, 1, count)}
        actual = calibrate({"splits": {"train": [entry]}})
        self.assertEqual(actual["calibration_edges"], [0.] * 6)
        self.assertEqual(actual["reliability_scale"], 1.)

    def test_fresh_mismatch_uses_training_scale_noise_and_states(self):
        config = copy.deepcopy(self.config)
        config["channel"]["paths"] = 3
        observations = prepare(config, "cpu", lambda *a, **k: None, split_names=("screen",), reference=self.cache, estimate=False)
        self.assertEqual(set(observations["splits"]), {"screen"})
        self.assertEqual(observations["channel_scale"], self.cache["channel_scale"])
        for a, b in zip(observations["contexts"], self.cache["contexts"]):
            self.assertEqual(a["noise_complex_variance"], b["noise_complex_variance"])
            self.assertTrue(torch.equal(a["matrix"], b["matrix"]))
        self.assertNotIn("coarse", observations["splits"]["screen"][0])

    def test_parent_cache_reuse_and_physical_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for folder in ("src", "scripts", "tests", "configs", "docs"):
                (root / folder).mkdir()
            for name in ("AGENTS.md", "README.md", "pyproject.toml", ".gitignore", ".gitattributes"):
                (root / name).write_text("test fixture", encoding="utf-8")
            parent = root / "runs" / "parent"
            parent.mkdir(parents=True)
            (parent / "config_resolved.json").write_text(json.dumps(self.config))
            torch.save(self.cache, parent / "data.pt")
            config_path = root / "configs" / "study.json"
            config_path.write_text(json.dumps(self.config))
            study = Study(root, config_path, "reuse", "cpu", "parent")
            study.log = lambda *a, **k: None
            reused = study.cache("base_observations", study.config)
            for split in ("train", "validation"):
                for a, b in zip(reused["splits"][split], self.cache["splits"][split]):
                    self.assertTrue(torch.equal(a["observation"], b["observation"]))
            changed = copy.deepcopy(self.config)
            changed["system"]["snrs_db"] = [11.]
            config_path.write_text(json.dumps(changed))
            invalid = Study(root, config_path, "mismatch", "cpu", "parent")
            invalid.log = lambda *a, **k: None
            with self.assertRaises(ValueError):
                invalid.cache("base_observations", invalid.config)

    def test_training_prefix_and_resume_are_exact(self):
        config = copy.deepcopy(self.config)
        config["research"]["train_curve_samples"] = 2
        cache = {**self.cache, "splits": {s: values[:1] for s, values in self.cache["splits"].items()}}
        spec = recipe(config, reliability="none")
        with tempfile.TemporaryDirectory() as temporary:
            for device in (("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)):
                with self.subTest(device=device):
                    path = Path(temporary) / device
                    complete = train_job(config, cache, "same", "hash", spec, 77, 6, path / "full", device, lambda *a, **k: None)
                    prefix = train_job(config, cache, "same", "hash", spec, 77, 2, path / "resume", device, lambda *a, **k: None)
                    prefix_state = torch.load(prefix, weights_only=True)["state_dict"]
                    expected_prefix = torch.load(complete.parent / "step_2.pt", weights_only=True)["state_dict"]
                    self.assertTrue(all(torch.equal(prefix_state[k], v) for k, v in expected_prefix.items()))
                    resumed = train_job(config, cache, "same", "hash", spec, 77, 6, path / "resume", device, lambda *a, **k: None)
                    expected = torch.load(complete, weights_only=True)["state_dict"]
                    actual = torch.load(resumed, weights_only=True)["state_dict"]
                    self.assertTrue(all(torch.equal(actual[k], v) for k, v in expected.items()))
        self.assertEqual(learning_rate(.001, 1, 20000), learning_rate(.001, 1, 20000))
        self.assertGreater(learning_rate(.001, 2000, 20000), learning_rate(.001, 8000, 20000))

    def test_rate_uses_one_shared_state_and_pilot_overhead(self):
        matrix = torch.eye(2, dtype=torch.complex64)[None].expand(2, -1, -1)
        truth = torch.tensor([[[3., 2.], [0., 2.]]], dtype=torch.complex64)
        context = {"matrix": matrix, "training_symbols": 2, "switches": 1}
        gross, net, oracle = communication_rates(truth, truth, context, torch.ones(1), {"coherence_symbols": 10, "switch_symbols": 1})
        self.assertAlmostEqual(float(gross), np.log2(5), places=6)
        self.assertAlmostEqual(float(net), .7 * np.log2(5), places=6)
        self.assertTrue(torch.equal(net, oracle))

    def test_stability_requires_convergence_and_moment_stability(self):
        a = [{"mean_nmse_db": -10., "mean_variance": .2, "convergence_rate": .99}]
        limits = {"nmse_db": .1, "variance_relative": .05, "overall_convergence": .95, "context_convergence": .9}
        self.assertTrue(stable_budget(a, a, limits)[0])
        self.assertFalse(stable_budget([{**a[0], "convergence_rate": .8}], a, limits)[0])
        self.assertFalse(stable_budget(a, [{**a[0], "mean_variance": .3}], limits)[0])

    def test_crossed_bootstrap_keeps_geometry_conditions_paired(self):
        rows = [{"method": method, "seed": seed, "scene": scene, "context": context,
            "nmse_linear": (scene + 1) * factor} for method, factor in (("sbl_stable", 1), ("refined", 2))
            for seed, scene, context in itertools.product((1, 2), range(4), range(2))]
        effect = paired_effects(rows, 20, 19)[0]
        for key in ("delta_linear_mean_db", "ci95_low_db", "ci95_high_db"):
            self.assertAlmostEqual(effect[key], 10 * np.log10(2))

    def test_server_and_smoke_configuration_boundaries(self):
        validate_study(self.config)
        validate_study(json.loads((ROOT / "configs/research.json").read_text()))
        changed = copy.deepcopy(self.config)
        changed["research"]["final_seeds"] = changed["research"]["training_seeds"]
        with self.assertRaises(ValueError):
            validate_study(changed)
        changed = copy.deepcopy(self.config)
        changed["research"]["expansions"][0]["name"] = "../escape"
        with self.assertRaises(ValueError):
            validate_study(changed)


if __name__ == "__main__":
    unittest.main()
