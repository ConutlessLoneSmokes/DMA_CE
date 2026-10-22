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
from dma_ce._upstream.atmosphere import AbsorptionTable, C0_M_PER_S, free_space_amplitude
from dma_ce._upstream.geometry import los_element_channel
from dma_ce.data import build_hardware, physical_channels, sample_scenes
from dma_ce.physical_study import case_hardware, make_entries, prepare_case, validate
from dma_ce.physics import BOLTZMANN_J_PER_K, paper_channel, reflection_magnitude, thermal_budget


class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.table_path = Path(self.directory.name) / "absorption.txt"
        np.savetxt(self.table_path, [[280e9, .001], [320e9, .001]])
        self.config = json.loads((ROOT / "configs/physical_baselines_smoke.json").read_text())
        self.config["channel"].update(absorption_table_path=str(self.table_path),
            absorption_table_sha256=hashlib.sha256(self.table_path.read_bytes()).hexdigest())

    def test_thermal_bandwidth_subcarrier_sum_and_receiver_whitening(self):
        settings = self.config["system"]["thermal_noise"]
        narrow = thermal_budget(settings, 3e9, 32)
        wide = thermal_budget(settings, 30e9, 32)
        self.assertAlmostEqual(wide["antenna_noise_per_tone_w"] / narrow["antenna_noise_per_tone_w"], 10.)
        self.assertAlmostEqual(32 * narrow["antenna_noise_per_tone_w"] /
            (BOLTZMANN_J_PER_K * settings["antenna_temperature_k"] * 3e9), 1.)
        response_power = np.array([.2, 1., 8.])
        raw = narrow["antenna_noise_per_tone_w"] * response_power + narrow["receiver_noise_per_tone_w"]
        whitened = raw / (response_power + narrow["receiver_noise_fraction"])
        np.testing.assert_allclose(whitened, narrow["antenna_noise_per_tone_w"], rtol=1e-14)

    def test_reflection_normal_incidence_and_roughness(self):
        frequencies = np.array([280e9, 320e9])
        np.testing.assert_allclose(reflection_magnitude(frequencies, 0., 1.5, 0.), .2)
        np.testing.assert_array_equal(reflection_magnitude(frequencies, 0., 1., 0.), 0.)
        rough = reflection_magnitude(frequencies, 0., 1.5, 8.8e-5)
        self.assertTrue(0 < rough[1] < rough[0] < .2)
        for kwargs in ((-1., 1.5, 0.), (0., .9, 0.), (0., 1.5, -.1)):
            with self.assertRaises(ValueError):
                reflection_magnitude(frequencies, *kwargs)

    def test_single_los_matches_existing_spreading_absorption_and_phase(self):
        frequencies = np.array([290e9, 310e9])
        positions = np.array([-.003, 0., .003])
        table = AbsorptionTable.from_txt(self.table_path)
        hardware = {"frequencies": frequencies, "positions": positions, "absorption_table": table,
            "channel_settings": {"antenna_gain_dbi": 15.}}
        actual = paper_channel(hardware, (np.array([[.25]]), np.array([[.3]]), np.ones((1, 1))))[0]
        expected = los_element_channel(frequencies, positions, .25, .3, absorption_table=table)
        np.testing.assert_allclose(actual, 10 ** (15/20) * expected, rtol=1e-13, atol=1e-15)

    def test_single_bounce_uses_total_travelled_distance_once(self):
        frequencies = np.array([290e9, 310e9])
        positions = np.array([0.])
        radii, angles = np.array([[.3, .2]]), np.array([[0., .4]])
        settings = {"antenna_gain_dbi": 0., "reflection_refractive_index": 1.5, "surface_roughness_m": 0.}
        hardware = {"frequencies": frequencies, "positions": positions, "absorption_table": None, "channel_settings": settings}
        actual = paper_channel(hardware, (radii, angles, np.array([[0., 1.]])))[0, :, 0]
        point = np.array([.2*np.sin(.4), .2*np.cos(.4)])
        before = np.linalg.norm(np.array([0., .3]) - point)
        total = before + .2
        self.assertGreaterEqual(total, .3)
        incidence = .5*np.arccos((before**2 + .2**2 - .3**2)/(2*before*.2))
        expected = reflection_magnitude(frequencies, incidence, 1.5, 0.) * free_space_amplitude(frequencies, total)
        expected = expected * np.exp(-2j*np.pi*frequencies*total/C0_M_PER_S) / np.sqrt(2)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)

    def test_physical_gain_generation_preserves_geometry_without_rayleigh_draws(self):
        physical = copy.deepcopy(self.config)
        physical["channel"]["model"] = "li2026"
        gaussian = copy.deepcopy(physical)
        gaussian["channel"]["model"] = "gaussian"
        a, b = sample_scenes(physical, 0, 7), sample_scenes(gaussian, 0, 7)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[2], np.ones_like(a[2]))
        self.assertFalse(np.allclose(b[2], 1))

    def test_absorption_flag_is_honored_and_table_hash_is_checked(self):
        config = copy.deepcopy(self.config)
        config["channel"].update(model="gaussian", paths=1, antenna_gain_dbi=0.)
        bank = np.full((4, 8), 300e9)
        absorbed = build_hardware(config, 3e9, bank)
        config["channel"]["include_absorption"] = False
        clear = build_hardware(config, 3e9, bank)
        scenes = sample_scenes(config, 0, 3)
        self.assertTrue(torch.all(physical_channels(absorbed, scenes).abs() < physical_channels(clear, scenes).abs()))
        config["channel"].update(include_absorption=True, absorption_table_sha256="0"*64)
        with self.assertRaises(ValueError):
            build_hardware(config, 3e9, bank)

    def test_power_sweep_fixes_physical_noise_and_changes_only_pilot_power(self):
        case = self.config["cases"][-1]
        cache = prepare_case(self.config, case)
        selected, _, hardware = case_hardware(self.config, case)
        entries, ids = make_entries(self.config, selected, hardware, cache, "screen")
        self.assertFalse(set(ids) & set(cache["geometry_ids"]["train"] + cache["geometry_ids"]["validation"]))
        for i in range(0, len(entries), 2):
            low, high = cache["contexts"][i:i+2]
            self.assertEqual(entries[i]["physical_noise_sha256"], entries[i+1]["physical_noise_sha256"])
            self.assertEqual(low["physical_white_noise_per_tone_w"], high["physical_white_noise_per_tone_w"])
            self.assertAlmostEqual(low["noise_complex_variance"] / high["noise_complex_variance"], 100.)
            self.assertAlmostEqual(high["snr_db"]-low["snr_db"], 20.)
            self.assertLess(entries[i]["pilot_equivalence_error"], 1e-5)
            self.assertLess(entries[i+1]["pilot_equivalence_error"], 1e-5)

    def test_observation_and_training_caches_support_safe_tensor_reload(self):
        case = self.config["cases"][-1]
        cache = prepare_case(self.config, case)
        path = Path(self.directory.name) / "cache.pt"
        torch.save(cache, path)
        restored = torch.load(path, weights_only=True)
        self.assertEqual(restored["geometry_ids"], cache["geometry_ids"])
        for expected, actual in zip(cache["contexts"], restored["contexts"]):
            self.assertIsInstance(actual["snr_db"], float)
            self.assertEqual(actual["snr_db"], expected["snr_db"])
        for expected, actual in zip(cache["validation"], restored["validation"]):
            self.assertTrue(torch.equal(actual["observation"], expected["observation"]))

    def test_invalid_thermal_and_physical_configuration(self):
        settings = self.config["system"]["thermal_noise"]
        for key, value in (("antenna_temperature_k", 0.), ("receiver_reference_temperature_k", -1.), ("noise_figure_db", -1.)):
            with self.assertRaises(ValueError):
                thermal_budget({**settings, key: value}, 3e9, 32)
        validate(self.config)
        self.config["system"]["transmit_power_dbm"] = [0, 0]
        with self.assertRaises(ValueError):
            validate(self.config)


if __name__ == "__main__":
    unittest.main()
