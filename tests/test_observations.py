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
from dma_ce.data import tensor_hash
from dma_ce.physical_study import case_hardware, make_entries, prepare_case, score_context, validate


class ObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        table = Path(directory.name) / "absorption.txt"
        np.savetxt(table, [[280e9, .001], [320e9, .001]])
        self.config = json.loads((ROOT / "configs/observation_baselines_smoke.json").read_text())
        self.config["channel"].update(absorption_table_path=str(table),
            absorption_table_sha256=hashlib.sha256(table.read_bytes()).hexdigest())

    def test_snr_calibration_preserves_thermal_noise_and_training_only_scale(self):
        config = self.config
        validate(config)
        case = config["cases"][0]
        cache = prepare_case(config, case)
        selected, _, hardware = case_hardware(config, case)
        entries, ids = make_entries(config, selected, hardware, cache, "screen")
        self.assertFalse(set(ids) & set(cache["geometry_ids"]["train"] + cache["geometry_ids"]["validation"]))
        previous = {}
        for context, entry in zip(cache["contexts"], entries):
            actual = 10*np.log10(context["training_white_power"] / context["noise_complex_variance"])
            self.assertAlmostEqual(actual, context["snr_db"], places=11)
            expected_variance = context["physical_white_noise_per_tone_w"] / (
                context["pilot_power_per_tone_w"] * cache["channel_scale"]**2)
            self.assertAlmostEqual(expected_variance / context["noise_complex_variance"], 1., places=12)
            self.assertLess(entry["pilot_equivalence_error"], 1e-5)
            key = context["bandwidth_hz"], context["configurations"]
            self.assertEqual(previous.setdefault(key, entry["physical_noise_sha256"]), entry["physical_noise_sha256"])
            largest = next(c["matrix"] for c in cache["contexts"] if c["bandwidth_hz"] == key[0]
                and c["configurations"] == max(config["system"]["configurations"]))
            self.assertTrue(torch.equal(context["matrix"], largest[:, :key[1]]))
        changed = copy.deepcopy(config)
        changed["system"].update(configurations=[8], snrs_db=[15])
        self.assertEqual(prepare_case(changed, case)["channel_scale"], cache["channel_scale"])

    def test_noiseless_ls_covers_under_square_and_overdetermined_observations(self):
        cache = prepare_case(self.config, self.config["cases"][0])
        for count in self.config["system"]["configurations"]:
            context = next(c for c in cache["contexts"] if c["configurations"] == count)
            matrix = context["matrix"].to(torch.complex128)
            truth = cache["validation"][0]["truth"].to(torch.complex128)
            clean = torch.einsum("kjn,bkn->bkj", matrix, truth)
            estimate = torch.einsum("knj,bkj->bkn", torch.linalg.pinv(matrix), clean)
            self.assertTrue(torch.allclose(torch.einsum("kjn,bkn->bkj", matrix, estimate), clean, atol=1e-10, rtol=1e-10))
            if count >= self.config["system"]["antennas"]:
                self.assertTrue(torch.allclose(estimate, truth, atol=1e-10, rtol=1e-10))

    def test_mse_units_and_shared_observation_identity(self):
        cache = prepare_case(self.config, self.config["cases"][0])
        context, entry = cache["contexts"][0], cache["validation"][0]
        prediction = entry["truth"] + (.1+.2j)
        rows = score_context(self.config, cache, context, entry,
            {"outputs": {"ls_min_norm": prediction}, "solver": {}})
        for row in rows:
            scene = row["scene"]
            self.assertAlmostEqual(row["mse_normalized"], .05, places=6)
            self.assertAlmostEqual(row["mse_physical"] / cache["channel_scale"]**2, .05, places=6)
            self.assertEqual(row["observation_sha256"], tensor_hash(entry["observation"][scene]))
            self.assertFalse(row["solver_applicable"])

    def test_invalid_or_ambiguous_axes_rejected(self):
        for key, value in (("transmit_power_dbm", [0]), ("snrs_db", []), ("snrs_db", [0, 0]),
                           ("snrs_db", [4000]), ("snrs_db", [-4000]), ("configurations", [0])):
            config = copy.deepcopy(self.config)
            config["system"][key] = value
            with self.assertRaises(ValueError):
                validate(config)


if __name__ == "__main__":
    unittest.main()
