import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dma_ce._upstream.ddpm import VPSchedule
from dma_ce._upstream.geometry import spherical_array_response, far_field_array_response, ula_positions
from dma_ce.conditional import ConditionalCNN, predict, reliability_features
from dma_ce.data import build_hardware, encode, decode, sample_scenes
from dma_ce.experiment import apply_dc, freeze_correction, run, validate_config
from dma_ce.sbl import consistency, joint_sbl, posterior


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)

    def test_posterior_matches_dense_channel_conditioning(self):
        dtype = torch.complex128
        dictionary = torch.randn(3, 5, 7, dtype=dtype)
        matrix = torch.randn(3, 2, 5, dtype=dtype)
        sensing = matrix @ dictionary
        gamma = torch.rand(2, 7, dtype=torch.float64) + .2
        variance = torch.tensor([.3, .7], dtype=torch.float64)
        y = torch.randn(2, 3, 2, dtype=dtype)
        _, _, mean, diagonal = posterior(y, sensing, dictionary, variance, gamma)
        for batch in range(2):
            covariance = dictionary @ torch.diag(gamma[batch]).to(dtype) @ dictionary.mH
            cross = covariance @ matrix.mH
            observed = matrix @ cross + variance[batch] * torch.eye(2, dtype=dtype)
            dense_mean = (cross @ torch.linalg.solve(observed, y[batch, ..., None])).squeeze(-1)
            dense_cov = covariance - cross @ torch.linalg.solve(observed, cross.mH)
            self.assertTrue(torch.allclose(mean[batch], dense_mean, atol=1e-10, rtol=1e-10))
            self.assertTrue(torch.allclose(diagonal[batch], dense_cov.diagonal(dim1=-2, dim2=-1).real, atol=1e-10, rtol=1e-10))

    def test_more_noise_increases_gaussian_uncertainty(self):
        dictionary = torch.randn(2, 4, 5, dtype=torch.complex128)
        sensing = torch.randn(2, 3, 5, dtype=torch.complex128)
        y = torch.randn(1, 2, 3, dtype=torch.complex128)
        gamma = torch.ones(1, 5, dtype=torch.float64)
        small = posterior(y, sensing, dictionary, torch.tensor([.1]), gamma)[3]
        large = posterior(y, sensing, dictionary, torch.tensor([1.]), gamma)[3]
        self.assertTrue((large >= small - 1e-9).all())

    def test_coefficient_only_posterior_preserves_sbl_updates(self):
        dictionary = torch.randn(2, 5, 9, dtype=torch.complex128)
        sensing = torch.randn(2, 3, 9, dtype=torch.complex128)
        y = torch.randn(3, 2, 3, dtype=torch.complex128)
        variance = torch.full((3,), .1, dtype=torch.float64)
        initial = (y.abs().square().mean((1, 2)) - variance).clamp_min(1e-6) / sensing.abs().square().sum(-1).mean()
        gamma = initial[:, None].expand(-1, 9).clone().clamp_min(1e-8)
        for _ in range(20):
            mean, diagonal, _, _ = posterior(y, sensing, dictionary, variance, gamma)
            fast = posterior(y, sensing, dictionary, variance, gamma, channel_moments=False)
            self.assertTrue(torch.equal(mean, fast[0]) and torch.equal(diagonal, fast[1]))
            gamma = (.5 * gamma + .5 * (mean.abs().square() + diagonal).mean(1)).clamp_min(1e-8)
        expected = posterior(y, sensing, dictionary, variance, gamma)
        actual = joint_sbl(y, sensing, dictionary, variance, iterations=20, tolerance=1e-12)
        self.assertTrue(torch.equal(actual[0], expected[2]))
        self.assertTrue(torch.equal(actual[1], expected[3]))

    def test_sbl_recovers_a_sparse_observed_channel(self):
        dictionary = torch.eye(6, dtype=torch.complex128)[None].expand(3, -1, -1)
        sensing = torch.randn(3, 4, 6, dtype=torch.complex128)
        truth = torch.zeros(2, 3, 6, dtype=torch.complex128)
        truth[:, :, 2] = torch.tensor([1 + 1j, -.8 + .2j])[:, None]
        y = torch.einsum("kjg,bkg->bkj", sensing, truth)
        estimate, variance, diagnostics = joint_sbl(y, sensing, dictionary,
            torch.full((2,), 1e-4, dtype=torch.float64), iterations=150, tolerance=1e-4)
        self.assertLess(float((estimate - truth).abs().square().sum() / truth.abs().square().sum()), .01)
        self.assertTrue((variance > 0).all())
        self.assertTrue((diagnostics["iterations"] <= 150).all())

    def test_negative_noise_is_rejected(self):
        with self.assertRaises(ValueError):
            posterior(torch.zeros(1, 1, 1, dtype=torch.complex64),
                torch.ones(1, 1, 1, dtype=torch.complex64), torch.ones(1, 1, 1, dtype=torch.complex64),
                torch.tensor([-1.]), torch.ones(1, 1))

    def test_consistency_reduces_observation_residual(self):
        matrix = torch.randn(3, 2, 5, dtype=torch.complex128)
        estimate = torch.randn(4, 3, 5, dtype=torch.complex128)
        y = torch.randn(4, 3, 2, dtype=torch.complex128)
        result = consistency(estimate, matrix, y, torch.ones(4), .3)
        before = (y - torch.einsum("kjn,bkn->bkj", matrix, estimate)).abs().square().sum()
        after = (y - torch.einsum("kjn,bkn->bkj", matrix, result)).abs().square().sum()
        self.assertLess(float(after), float(before))

    def test_shared_tuning_produces_frequency_diversity(self):
        config = json.loads((ROOT / "configs/smoke.json").read_text())
        system = config["system"]
        bank = np.random.default_rng(3).uniform(285e9, 315e9, (8, system["antennas"]))
        hw = build_hardware(config, 30e9, bank)
        self.assertEqual(tuple(hw["matrix"].shape), (16, 8, 16))
        self.assertGreater(hw["rayleigh_distance_m"], config["channel"]["range_max_m"])
        first, last = hw["matrix"][0, 0], hw["matrix"][-1, 0]
        correlation = torch.abs(torch.vdot(first, last)) / (first.norm() * last.norm())
        self.assertLess(float(correlation), .99)

    def test_spherical_response_has_far_field_limit(self):
        positions = ula_positions(32, .0005)
        spherical = spherical_array_response(300e9, positions, 1e5, .3)
        far = far_field_array_response(300e9, positions, .3)
        self.assertLess(float(np.linalg.norm(spherical - far)), 2e-5)

    def test_scene_splits_and_complex_layout(self):
        config = json.loads((ROOT / "configs/smoke.json").read_text())
        a, b = sample_scenes(config, 0, 10), sample_scenes(config, 1, 10)
        self.assertFalse(np.array_equal(a[0], b[0]))
        h = torch.randn(2, 7, 5, dtype=torch.complex64)
        self.assertTrue(torch.equal(decode(encode(h)), h))

    def test_ddim_analytical_denoiser_recovers_its_target(self):
        schedule = VPSchedule()
        coarse = torch.randn(2, 2, 8, 8)
        class Exact(torch.nn.Module):
            def forward(self, state, condition, reliability, step):
                alpha = schedule.alpha_bar[step.long() - 1].to(state)[:, None, None, None]
                return (state - alpha.sqrt() * condition) / (1 - alpha).sqrt()
        output = predict(Exact(), schedule, coarse, torch.zeros(2, 1, 8, 8), "conditional", 1., 20, seed=3)
        self.assertTrue(torch.allclose(output, decode(coarse), atol=2e-5, rtol=2e-5))

    def test_same_sampler_seed_and_capacity(self):
        model = ConditionalCNN(width=16).eval()
        schedule = VPSchedule()
        coarse = torch.randn(2, 2, 8, 8)
        reliability = torch.randn(2, 1, 8, 8)
        first = predict(model, schedule, coarse, reliability, "residual", .2, 5, seed=4)
        second = predict(model, schedule, coarse, reliability + 100, "residual", .2, 5, seed=4)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(model.pre[0].in_channels, 5)
        self.assertEqual(model.post[-1].out_channels, 2)

    def test_deterministic_zero_reliability_control(self):
        model = ConditionalCNN(width=16).eval()
        schedule = VPSchedule()
        coarse = torch.randn(2, 2, 8, 8)
        reliability = torch.randn(2, 1, 8, 8)
        first = predict(model, schedule, coarse, reliability, "deterministic_no_reliability", .2, 5, seed=4)
        second = predict(model, schedule, coarse, reliability + 100, "deterministic_no_reliability", .2, 5, seed=9)
        self.assertTrue(torch.equal(first, second))
        with_v = predict(model, schedule, coarse, reliability, "deterministic", .2, 5, seed=4)
        self.assertFalse(torch.allclose(first, with_v))

    def test_validation_can_select_no_observation_correction(self):
        config = json.loads((ROOT / "configs/smoke.json").read_text())
        config["evaluation"].update(dc_multipliers=[0., .1, 1.], dc_selection_model="deterministic_no_reliability")
        class Zero(torch.nn.Module):
            def forward(self, state, coarse, reliability, step):
                return torch.zeros_like(state)
        coarse = torch.randn(2, 4, 4, dtype=torch.complex64)
        matrix = torch.randn(4, 2, 4, dtype=torch.complex64)
        entry = {"truth": coarse, "coarse": coarse, "reliability": torch.ones(2, 4, 4),
            "observation": torch.einsum("kjn,bkn->bkj", matrix, coarse) + .1,
            "variance": torch.ones(2)}
        cache = {"splits": {"validation": [entry]}, "contexts": [{"matrix": matrix}],
            "reliability_scale": 1., "reliability_log_mean": 0., "reliability_log_std": 1.,
            "residual_real_scale": 1., "dc_reference_complex_variance": 1.}
        frozen = freeze_correction(config, cache, {"deterministic_no_reliability": Zero()}, "cpu", lambda *a, **k: None)
        self.assertEqual(frozen["selected_multiplier"], 0.)
        self.assertTrue(torch.equal(apply_dc(coarse, entry, cache["contexts"][0], 0., "cpu")[0], coarse))

    def test_new_screen_seed_preserves_training_and_validation(self):
        config = json.loads((ROOT / "configs/smoke.json").read_text())
        original = [sample_scenes(config, split, 10) for split in range(3)]
        config["data"]["screen_seed"] = config["seed"] + 1
        fresh = [sample_scenes(config, split, 10) for split in range(3)]
        for split in (0, 1):
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(original[split], fresh[split])))
        self.assertFalse(np.array_equal(original[2][0], fresh[2][0]))

    def test_reliability_uses_frozen_training_constants(self):
        cache = {"reliability_scale": 2., "reliability_log_mean": 0., "reliability_log_std": 1.}
        diagonal = torch.full((3, 5, 7), .5)
        self.assertTrue(torch.equal(reliability_features(diagonal, cache), torch.zeros(3, 1, 7, 5)))

    def test_config_and_run_path_boundary(self):
        config = json.loads((ROOT / "configs/pilot.json").read_text())
        validate_config(config)
        config["system"]["configurations"] = [1000]
        with self.assertRaises(ValueError):
            validate_config(config)
        with self.assertRaises(ValueError):
            run(ROOT / "configs/smoke.json", "../escape", ROOT)

    def test_upstream_provenance(self):
        records = json.loads((ROOT / "docs/UPSTREAM_REUSE.json").read_text())
        self.assertEqual(len(records), 4)
        for record in records:
            digest = hashlib.sha256((ROOT / record["destination"]).read_bytes()).hexdigest()
            self.assertEqual(digest, record["destination_sha256"])


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
