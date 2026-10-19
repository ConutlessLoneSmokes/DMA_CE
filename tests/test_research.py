import copy
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dma_ce.conditional import ConditionalCNN, reliability_features
from dma_ce.data import calibrate, prepare, reestimate
from dma_ce.continuation import verify_lineage
from dma_ce.experiment import file_hash, write_json
from dma_ce.research import checkpoint_budgets, communication_rates, learning_rate, loss_weights, matched_width, paired_effects, recipe, train_job
from dma_ce.suite import Study, TrainingBudgetBoundary, stable_budget, validate_study


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

    def test_diffusion_extension_preserves_prefix_optimizer_rng_and_logs(self):
        extended = copy.deepcopy(self.config)
        extended["research"]["diffusion_checkpoints"] = [2, 4, 6, 8, 10]
        cache = {**self.cache, "splits": {s: values[:1] for s, values in self.cache["splits"].items()}}
        spec = recipe(extended, reliability="none", kind="diffusion")
        with tempfile.TemporaryDirectory() as temporary:
            for device in (("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)):
                with self.subTest(device=device):
                    path = Path(temporary) / device
                    full = train_job(extended, cache, "same", "hash", spec, 77, 10, path / "full", device, lambda *a, **k: None)
                    prefix = train_job(self.config, cache, "same", "hash", spec, 77, 6, path / "resume", device, lambda *a, **k: None)
                    preserved = {p.name: p.read_bytes() for p in prefix.parent.iterdir() if p.name != "last.pt"}
                    train_job(extended, cache, "same", "hash", spec, 77, 8, path / "resume", device, lambda *a, **k: None)
                    resumed = train_job(extended, cache, "same", "hash", spec, 77, 10, path / "resume", device, lambda *a, **k: None)
                    expected = torch.load(full.parent / "last.pt", weights_only=True)
                    actual = torch.load(resumed.parent / "last.pt", weights_only=True)
                    for key in ("state_dict", "batch_rng", "noise_rng", "optimizer"):
                        def equal(a, b):
                            if torch.is_tensor(a):
                                return torch.equal(a, b)
                            if isinstance(a, dict):
                                return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
                            if isinstance(a, list):
                                return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
                            return a == b
                        self.assertTrue(equal(expected[key], actual[key]), key)
                    for name, raw in preserved.items():
                        self.assertTrue((prefix.parent / name).read_bytes().startswith(raw) if name.endswith(".jsonl")
                            else (prefix.parent / name).read_bytes() == raw, name)
                    self.assertEqual(actual["optimizer"]["param_groups"][0]["lr"], self.config["learning"]["learning_rate"] * .1)
                    with self.assertRaises(ValueError):
                        train_job(extended, cache, "same", "hash", recipe(extended), 77, 8, path / "det", device, lambda *a, **k: None)
        validate_study(extended)
        extended["research"]["diffusion_checkpoints"] = [2, 6, 8]
        with self.assertRaises(ValueError):
            validate_study(extended)

    def continuation_parent(self, root):
        for folder in ("src", "scripts", "tests", "configs", "docs"):
            (root / folder).mkdir()
        for name in ("AGENTS.md", "README.md", "pyproject.toml", ".gitignore", ".gitattributes"):
            (root / name).write_text("test fixture", encoding="utf-8")
        config = copy.deepcopy(self.config)
        config["device"] = "cpu"
        config["research"].update(training_seeds=[101], final_seeds=[201], averages=[1, 4], expansions=[], mismatches=[])
        config_path = root / "configs" / "test.json"
        config_path.write_text(json.dumps(config))
        parent = Study(root, config_path, "parent")
        parent.log = lambda *a, **k: None
        parent.cache("base_stable", config)
        chosen = {"cache_id": "base_stable", "architecture": "cross_frequency", "loss": "uniform",
            "reliability": "scalar", "steps": 6, "iterations": 5, "mean_nmse_linear": 1.}
        for stage in ("baseline", "curves", "factorial", "reliability", "budget"):
            record = {"status": "complete", "selection": chosen}
            write_json(parent.path / "stages" / f"{stage}.json", record)
            write_json(parent.export / f"{stage}_selection.json", record)
        parent.run("diffusion")
        return parent

    def test_continuation_is_independent_resumable_and_validation_auditable(self):
        from scripts.audit_research import audit
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = self.continuation_parent(root)
            before = {p: file_hash(p) for folder in (parent.path, parent.export) for p in folder.rglob("*") if p.is_file()}
            (root / "src" / "version.py").write_text("# new continuation implementation\n")
            child = Study(root, None, "child", continue_from="parent", diffusion_checkpoints=[8, 10])
            child.log = lambda *a, **k: None
            self.assertEqual(checkpoint_budgets(child.config, "diffusion"), [2, 4, 6, 8, 10])
            self.assertEqual(child.config["research"]["checkpoints"], [2, 4, 6])
            for stage in ("baseline", "curves", "factorial", "reliability", "budget"):
                self.assertEqual(child.result(stage), parent.result(stage))
            child.run("diffusion")
            lineage = verify_lineage(child.path, child.export)
            self.assertEqual(len(lineage["origin_states"]), 2)
            self.assertTrue(all(file_hash(p) == value for p, value in before.items()))
            self.assertFalse((child.path / "cache/base_stable.pt").samefile(parent.path / "cache/base_stable.pt"))
            Study(root, None, "child", continue_from="parent", diffusion_checkpoints=[8, 10])
            report = audit("child", root, validation_only=True)
            self.assertEqual(report["status"], "passed")
            self.assertTrue(report["continuation_verified"] and report["ready_for_final"])
            self.assertEqual(report["maximum_metric_difference"], 0.)
            self.assertEqual(audit("child", root)["status"], "not_ready")
            grandchild = Study(root, None, "grandchild", continue_from="child", diffusion_checkpoints=[12])
            grandchild.log = lambda *a, **k: None
            grandchild.run("diffusion")
            self.assertEqual(checkpoint_budgets(grandchild.config, "diffusion"), [2, 4, 6, 8, 10, 12])
            self.assertEqual(verify_lineage(grandchild.path, grandchild.export)["parent_run_id"], "child")
            with self.assertRaises(ValueError):
                Study(root, None, "child", continue_from="parent", diffusion_checkpoints=[8, 12])
            with self.assertRaisesRegex(ValueError, "RNG device"):
                Study(root, None, "child", "cuda", continue_from="parent", diffusion_checkpoints=[8, 10])
            old = child.path / lineage["origin_states"][0]["snapshot_path"]
            old.write_bytes(old.read_bytes() + b"changed")
            with self.assertRaises(ValueError):
                verify_lineage(child.path, child.export)
            self.assertIsNotNone(verify_lineage(grandchild.path, grandchild.export))

    def test_continuation_rejects_screening_changed_budget_and_missing_optimizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = self.continuation_parent(root)
            for points in (None, [6], [8, 8], [10, 8]):
                with self.assertRaises(ValueError):
                    Study(root, None, "invalid", continue_from="parent", diffusion_checkpoints=points)
            final = parent.path / "stages/final.json"
            final.write_text('{"status":"running"}')
            with self.assertRaisesRegex(ValueError, "screening"):
                Study(root, None, "screened", continue_from="parent", diffusion_checkpoints=[8])
            final.unlink()
            last = next((parent.path / "models").rglob("last.pt"))
            saved = torch.load(last, weights_only=True)
            saved.pop("optimizer")
            torch.save(saved, last)
            with self.assertRaisesRegex(ValueError, "optimizer/RNG"):
                Study(root, None, "no_optimizer", continue_from="parent", diffusion_checkpoints=[8])
            self.assertFalse((root / "runs/no_optimizer").exists())

    def test_latest_interval_and_budget_limited_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study = self.continuation_parent(root)
            stage = study.path / "stages/diffusion.json"
            stage.unlink()
            study.config["research"].update(training_seeds=[101], averages=[1], training_stability_db=.1)
            study.result = lambda name: {"cache_id": "unused", "architecture": "cross_frequency", "loss": "uniform",
                "reliability": "none", "mean_nmse_linear": 1.}
            study.get_cache = lambda name: {}
            study.train = lambda *a: root / "step_6.pt"
            study.table = lambda name, rows: study.export / "diffusion"
            values = {2: 1., 4: 2., 6: 4.}
            study.score = lambda cache, checkpoint, method, **kw: [{"method": method, "nmse_linear": values[int(checkpoint.stem[5:])],
                "total_batch_ms_per_sample": 1.}]
            with patch("dma_ce.suite.plot_tradeoff"):
                selected = study.diffusion()
                self.assertEqual(selected["selected"]["steps"], 2)
                self.assertAlmostEqual(selected["last_interval_improvement_db"], 10 * np.log10(.5))
                values.update({2: 4., 4: 2., 6: 1.})
                with self.assertRaises(TrainingBudgetBoundary):
                    study.run("diffusion")
            self.assertEqual(json.loads(stage.read_text())["status"], "budget_limited")
            self.assertTrue((study.export / "report.md").exists())

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
