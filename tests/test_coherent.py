from pathlib import Path
import math
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dma_ce.coherent import coherent_posterior
from dma_ce.sbl import posterior


class CoherentTests(unittest.TestCase):
    def test_zero_strength_is_the_same_independent_gaussian_posterior(self):
        torch.manual_seed(310)
        dtype = torch.complex128
        dictionary = torch.randn(4, 5, 7, dtype=dtype)
        matrix = torch.randn(4, 2, 5, dtype=dtype)
        y = torch.randn(2, 4, 2, dtype=dtype)
        gamma = torch.rand(2, 7, dtype=torch.float64) + .2
        variance = torch.tensor([.1, .3], dtype=torch.float64)
        model = {"indices": torch.tensor([[0, 2], [1, 4]]),
            "delays": torch.tensor([[[.1, .3], [.2, .4]], [[.4, .1], [.3, .2]]]),
            "weights": torch.tensor([[[.3, .2], [.1, .5]], [[.5, .1], [.2, .4]]]),
            "envelope": torch.ones(4, 1, dtype=torch.float64)}
        actual = coherent_posterior(y, matrix, dictionary, variance, gamma, model, 0.)
        _, _, expected, _ = posterior(y, matrix @ dictionary, dictionary, variance, gamma)
        torch.testing.assert_close(actual["mean"], expected, rtol=1e-10, atol=1e-10)

    def test_low_rank_posterior_and_evidence_match_full_complex_gaussian(self):
        torch.manual_seed(311)
        dtype = torch.complex128
        k, n, m, g = 4, 5, 2, 7
        dictionary = torch.randn(k, n, g, dtype=dtype)
        matrix = torch.randn(k, m, n, dtype=dtype)
        y = torch.randn(2, k, m, dtype=dtype)
        gamma = torch.rand(2, g, dtype=torch.float64) + .2
        variance = torch.tensor([.1, .3], dtype=torch.float64)
        envelope = torch.randn(k, 3, dtype=torch.float64)
        envelope /= envelope.norm(dim=-1, keepdim=True)
        model = {"indices": torch.tensor([[0, 2], [1, 4]]),
            "delays": torch.tensor([[[.1, .3], [.2, .4]], [[.4, .1], [.3, .2]]], dtype=torch.float64),
            "weights": torch.tensor([[[.3, .2], [.1, .5]], [[.5, .1], [.2, .4]]], dtype=torch.float64),
            "envelope": envelope}
        strength = .65
        actual = coherent_posterior(y, matrix, dictionary, variance, gamma, model, strength)
        observation_operator = torch.block_diag(*list(matrix @ dictionary))
        channel_operator = torch.block_diag(*list(dictionary))
        for batch in range(2):
            covariance = torch.zeros(k, g, k, g, dtype=dtype)
            for atom in range(g):
                block = torch.eye(k, dtype=dtype)
                if atom in model["indices"][batch]:
                    position = model["indices"][batch].tolist().index(atom)
                    weights = model["weights"][batch, position]
                    block *= 1 - strength * weights.sum()
                    for delay, weight in zip(model["delays"][batch, position], weights):
                        basis = torch.exp(-2j * math.pi * torch.arange(k, dtype=torch.float64) * delay)[:, None] * envelope
                        block += strength * weight * (basis @ basis.mH)
                torch.testing.assert_close(block.diag().real, torch.ones(k, dtype=torch.float64))
                self.assertGreaterEqual(float(torch.linalg.eigvalsh(block).min()), -1e-12)
                covariance[:, atom, :, atom] = gamma[batch, atom] * block
            covariance = covariance.reshape(k * g, k * g)
            observed_covariance = observation_operator @ covariance @ observation_operator.mH + variance[batch] * torch.eye(k * m, dtype=dtype)
            solved = torch.linalg.solve(observed_covariance, y[batch].flatten())
            expected = (channel_operator @ covariance @ observation_operator.mH @ solved).reshape(k, n)
            objective = (torch.linalg.slogdet(observed_covariance).logabsdet + torch.vdot(y[batch].flatten(), solved).real) / k
            torch.testing.assert_close(actual["mean"][batch], expected, rtol=1e-9, atol=1e-9)
            torch.testing.assert_close(actual["objective_per_tone"][batch], objective, rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(observed_covariance @ actual["precision_observation"][batch].flatten(),
                y[batch].flatten(), rtol=1e-10, atol=1e-10)

    def test_known_off_grid_delay_is_recovered_modulo_the_sampling_period(self):
        from dma_ce.coherent import delay_model
        k = 32
        frequencies = torch.linspace(285e9, 315e9, k, dtype=torch.float64)
        powers = torch.tensor([[2., 1., .2]], dtype=torch.float64)
        delays = torch.tensor([.21789, .63111, .4], dtype=torch.float64)
        coefficients = torch.exp(-2j * math.pi * torch.arange(k, dtype=torch.float64)[:, None] * delays)[None]
        uncertainty = torch.full_like(coefficients.real, 1e-4)
        settings = {"coherent_atoms": 2, "delay_grid": 128, "delay_hypotheses": 2,
            "delay_refinement_steps": 3, "coefficient_variance_floor": .001,
            "envelope_weights": [1., .1, .02]}
        model = delay_model(coefficients, uncertainty, powers, frequencies, settings)
        for atom, estimates in zip(model["indices"][0], model["delays"][0]):
            difference = (estimates - delays[atom] + .5) % 1 - .5
            self.assertLess(float(difference.abs().min()), 2e-5)
        self.assertTrue((model["weights"].sum(-1) <= 1 + 1e-12).all())
        self.assertTrue((model["weights"] >= 0).all())

    def test_delay_period_and_invalid_prior_inputs(self):
        dtype = torch.complex128
        matrix = torch.eye(2, dtype=dtype).expand(4, -1, -1)
        y = torch.tensor([[[1j, 1], [1, 1j], [-1j, -1], [-1, -1j]]], dtype=dtype)
        gamma = torch.ones(1, 2, dtype=torch.float64)
        noise = torch.tensor([.1], dtype=torch.float64)
        model = {"indices": torch.tensor([[0, 1]]), "delays": torch.tensor([[[.2], [.7]]], dtype=torch.float64),
            "weights": torch.tensor([[[1.], [1.]]], dtype=torch.float64), "envelope": torch.ones(4, 1, dtype=torch.float64)}
        original = coherent_posterior(y, matrix, matrix, noise, gamma, model, 1.)
        shifted = coherent_posterior(y, matrix, matrix, noise, gamma, {**model, "delays": model["delays"] + 1}, 1.)
        torch.testing.assert_close(original["mean"], shifted["mean"], rtol=1e-11, atol=1e-11)
        for strength in (-.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                coherent_posterior(y, matrix, matrix, noise, gamma, model, strength)
        for bad in ({**model, "weights": model["weights"] * -1},
                {**model, "weights": model["weights"] * 1.1}, {**model, "indices": torch.tensor([[0, 0]])}):
            with self.assertRaises(ValueError):
                coherent_posterior(y, matrix, matrix, noise, gamma, bad, .5)


if __name__ == "__main__":
    unittest.main()
