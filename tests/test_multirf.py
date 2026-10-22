import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dma_ce.data import build_hardware, physical_channels, sample_scenes
from dma_ce.factor_lmmse import apply, fit
from dma_ce.multirf_study import validate_multirf
from dma_ce.physical_study import case_hardware, make_entries, prepare_case
from dma_ce.physics import element_channel, scene_points
from dma_ce.structured import apply_lmmse, lmmse_statistics


class MultiRFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        table = Path(directory.name) / "absorption.txt"
        np.savetxt(table, [[280e9, .001], [320e9, .001]])
        self.config = json.loads((ROOT / "configs/multirf_snr_smoke.json").read_text())
        self.config["channel"].update(absorption_table_path=str(table),
            absorption_table_sha256=hashlib.sha256(table.read_bytes()).hexdigest())

    def test_factorization_matches_dense_joint_shrinkage_and_per_tone(self):
        torch.manual_seed(17)
        for measurements in (2, 7):
            training = torch.randn(13, 3, 5, dtype=torch.complex128)
            matrix = torch.randn(3, measurements, 5, dtype=torch.complex128)
            observation = torch.randn(4, 3, measurements, dtype=torch.complex128)
            for strength in (0., .37, 1.):
                for variance in (.1, 1e-4):
                    expected = apply_lmmse(lmmse_statistics(training, matrix, variance,
                        frequency_shrinkage=strength), observation)
                    actual = apply(fit(training, matrix, variance, strength), observation)
                    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=2e-7)

    def test_rf_blocks_whitening_feed_reset_and_resource_counts(self):
        config = self.config
        validate_multirf(config)
        bank = np.full((3, 8), 300e9)
        hw = build_hardware(config, 30e9, bank)
        a = hw["matrix"].numpy()
        self.assertEqual(a.shape, (4, 6, 8))
        for state in range(3):
            np.testing.assert_array_equal(a[:, state*2, 4:], 0)
            np.testing.assert_array_equal(a[:, state*2+1, :4], 0)
            np.testing.assert_allclose(a[:, state*2, :4], a[:, state*2+1, 4:], rtol=1e-7)
            gain = hw["noise_gain"][:, state*2:state*2+2].numpy()
            raw = a[:, state*2:state*2+2].conj() * np.sqrt(gain)[..., None]
            covariance = raw @ raw.conj().transpose(0, 2, 1)
            covariance += hw["thermal_budget"]["receiver_noise_fraction"] * np.eye(2)
            white = covariance / np.sqrt(gain[:, :, None] * gain[:, None, :])
            np.testing.assert_allclose(white, np.broadcast_to(np.eye(2), white.shape), atol=2e-7)
        self.assertEqual(hw["positions"].shape, (8, 3))
        cache = prepare_case(config, config["cases"][0])
        selected, _, hardware = case_hardware(config, config["cases"][0])
        entries, ids = make_entries(config, selected, hardware, cache, "screen")
        self.assertFalse(set(ids) & set(cache["geometry_ids"]["train"] + cache["geometry_ids"]["validation"]))
        for context, entry in zip(cache["contexts"], entries):
            self.assertEqual(context["matrix"].shape, (4, 6, 8))
            self.assertEqual(entry["observation"].shape, (4, 4, 6))
            self.assertEqual(context["pilot_re"], 12)
            self.assertEqual(context["measurements_per_tone"], 6)
            self.assertAlmostEqual(10*np.log10(context["training_white_power"] / context["noise_complex_variance"]), context["snr_db"])
            self.assertEqual(entry["physical_noise_sha256"], entries[0]["physical_noise_sha256"])
            self.assertLess(entry["pilot_equivalence_error"], 1e-5)

    def test_planar_los_uses_every_element_position(self):
        config = self.config
        config["channel"].update(paths=1, antenna_gain_dbi=0.)
        hw = build_hardware(config, 30e9, np.full((3, 8), 300e9))
        scenes = sample_scenes(config, 0, 2)
        actual = physical_channels(hw, scenes).numpy()
        for i in range(2):
            r, a = scenes[0][i, 0], scenes[1][i, 0]
            expected = element_channel(hw["frequencies"], hw["positions"], r, a,
                absorption_table=hw["absorption_table"], elevation_deg=config["channel"]["elevation_deg"])
            np.testing.assert_allclose(actual[i], expected, rtol=1e-6, atol=1e-12)
        point = scene_points(np.asarray(2.), np.asarray(.3), 30.)
        self.assertAlmostEqual(np.linalg.norm(point), 2.)
        self.assertGreater(np.ptp(np.linalg.norm(hw["positions"] - point, axis=-1)), 0.)

    def test_invalid_architecture_and_learning_budgets_rejected(self):
        for part, key, value in (("system", "rf_chains", 3), ("system", "elements_per_strip", 5),
                ("channel", "elevation_deg", float("nan")), ("ddpm", "averages", 0),
                ("ddpm", "prior_checkpoints", [8, 4]), ("ddpm", "dc_multipliers", [-1.])):
            config = copy.deepcopy(self.config)
            config[part][key] = value
            with self.assertRaises(ValueError):
                validate_multirf(config)


if __name__ == "__main__":
    unittest.main()
