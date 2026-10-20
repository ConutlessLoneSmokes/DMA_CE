from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dma_ce._upstream.ddpm import DenoisingCNN, VPSchedule
from dma_ce.data import decode, encode
from dma_ce.sbl import consistency
from dma_ce.structured import VarianceUpdate, apply_lmmse, gaussian_update, lmmse_statistics, structured_recover


class StructuredTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(317)

    def gaussian(self, dtype=torch.complex128):
        mean = torch.randn(2, 3, 4, dtype=dtype)
        matrix = torch.randn(3, 2, 4, dtype=dtype)
        observation = torch.randn(2, 3, 2, dtype=dtype)
        variance = torch.tensor([.3, .7], dtype=mean.real.dtype)
        diagonal = torch.rand_like(mean.real) + .2
        return mean, matrix, observation, variance, diagonal

    def test_gaussian_matches_dense_complex_conditioning(self):
        mean, matrix, observation, variance, diagonal = self.gaussian()
        actual, actual_diagonal = gaussian_update(mean, matrix, observation, variance, diagonal)
        for batch in range(len(mean)):
            covariance = torch.diag_embed(diagonal[batch]).to(mean.dtype)
            cross = covariance @ matrix.mH
            observed_covariance = matrix @ cross + variance[batch] * torch.eye(2, dtype=mean.dtype)
            expected = mean[batch] + (cross @ torch.linalg.solve(observed_covariance,
                (observation[batch] - (matrix @ mean[batch, ..., None]).squeeze(-1))[..., None])).squeeze(-1)
            expected_covariance = covariance - cross @ torch.linalg.solve(observed_covariance, cross.mH)
            self.assertTrue(torch.allclose(actual[batch], expected, atol=1e-11, rtol=1e-11))
            self.assertTrue(torch.allclose(actual_diagonal[batch],
                expected_covariance.diagonal(dim1=-2, dim2=-1).real, atol=1e-11, rtol=1e-11))
        self.assertTrue((actual_diagonal <= diagonal + 1e-12).all())

    def test_scalar_update_equals_existing_consistency(self):
        mean, matrix, observation, variance, diagonal = self.gaussian()
        actual = gaussian_update(mean, matrix, observation, variance, torch.full_like(diagonal, .6))[0]
        expected = consistency(mean, matrix, observation, variance, .6)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))

    def test_noise_and_unobserved_component(self):
        mean = torch.tensor([[[1 + 2j, 3 + 4j]]], dtype=torch.complex128)
        matrix = torch.tensor([[[1., 0.]]], dtype=mean.dtype)
        observation = torch.tensor([[[5 - 2j]]], dtype=mean.dtype)
        diagonal = torch.tensor([[[.8, 1.7]]], dtype=torch.float64)
        small = gaussian_update(mean, matrix, observation, torch.tensor([.1]), diagonal)
        large = gaussian_update(mean, matrix, observation, torch.tensor([10.]), diagonal)
        self.assertEqual(small[0][0, 0, 1], mean[0, 0, 1])
        self.assertEqual(small[1][0, 0, 1], diagonal[0, 0, 1])
        self.assertTrue((large[1] >= small[1]).all())
        self.assertLess(float((small[0][..., 0] - observation[..., 0]).abs()),
            float((large[0][..., 0] - observation[..., 0]).abs()))

    def test_gaussian_gradients_include_complex_matrix_and_variances(self):
        values = (torch.randn(1, 1, 2, dtype=torch.complex128),
            torch.randn(1, 1, 2, dtype=torch.complex128), torch.randn(1, 1, 1, dtype=torch.complex128),
            torch.tensor([.4], dtype=torch.float64), torch.tensor([[[.7, 1.2]]], dtype=torch.float64))
        values = tuple(x.requires_grad_() for x in values)
        self.assertTrue(torch.autograd.gradcheck(gaussian_update, values, atol=1e-5, rtol=1e-4))

    def test_joint_lmmse_uses_unbiased_complex_cross_frequency_covariance(self):
        training = torch.randn(7, 2, 2, dtype=torch.complex64)
        training[:, 1] = (1 + 2j) * training[:, 0] + .1 * training[:, 1]
        matrix = torch.randn(2, 1, 2, dtype=torch.complex64)
        statistics = lmmse_statistics(training, matrix, .2)
        h = training.to(torch.complex128).flatten(1)
        centered = h - h.mean(0)
        covariance = centered.T @ centered.conj() / 6
        operator = torch.block_diag(*matrix.to(torch.complex128).unbind())
        cross = covariance @ operator.mH
        observed_covariance = operator @ cross + .2 * torch.eye(2, dtype=torch.complex128)
        gain = cross @ torch.linalg.inv(observed_covariance)
        expected_diagonal = (covariance - gain @ cross.mH).diagonal().real.reshape(2, 2)
        self.assertEqual(statistics["gain"].dtype, torch.complex128)
        self.assertEqual(statistics["diagonal"].dtype, torch.float64)
        self.assertTrue(torch.allclose(statistics["gain"], gain, atol=1e-11, rtol=1e-11))
        self.assertTrue(torch.allclose(statistics["diagonal"], expected_diagonal, atol=1e-11, rtol=1e-11))
        self.assertGreater(float(statistics["gain"][:2, 1].abs().sum()), .01)
        self.assertGreater(float(covariance.imag.abs().max()), .01)
        observation = torch.randn(3, 2, 1, dtype=torch.complex64)
        expected = (h.mean(0)[None] + (observation.flatten(1).to(torch.complex128)
            - (operator @ h.mean(0))[None]) @ gain.T).reshape(3, 2, 2).to(observation.dtype)
        actual = apply_lmmse(statistics, observation)
        self.assertEqual(actual.dtype, observation.dtype)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_lmmse_constant_training_distribution(self):
        training = torch.full((4, 2, 3), 1 + 2j, dtype=torch.complex64)
        matrix = torch.randn(2, 1, 3, dtype=torch.complex64)
        statistics = lmmse_statistics(training, matrix, .1)
        self.assertEqual(float(statistics["diagonal"].sum()), 0.)
        result = apply_lmmse(statistics, torch.randn(2, 2, 1, dtype=torch.complex64))
        self.assertTrue(torch.equal(result, training[:2]))

    def test_variance_network_initialization_and_bounded_multipliers(self):
        mean, _, _, variance, diagonal = self.gaussian(torch.complex64)
        updater = VarianceUpdate(4)
        self.assertTrue(torch.equal(updater(mean, diagonal, mean * 2, variance), diagonal))
        for bias, factor in ((100., 4.), (-100., -4.)):
            with torch.no_grad():
                updater.net[-1].bias.fill_(bias)
            result = updater(mean, diagonal, mean * 2, variance)
            self.assertTrue(torch.allclose(result, diagonal * torch.tensor(factor).exp()))
            self.assertTrue((result > 0).all())

    def test_real_variance_time_indices_and_tweedie_formula(self):
        schedule = VPSchedule(6, .1, .2)
        ratios = (1 - schedule.alpha_bar) / schedule.alpha_bar
        chosen = torch.tensor([1, 4])
        diagonal = (2 * ratios[chosen] / .8)[:, None, None].expand(2, 2, 3).clone()
        mean = torch.randn(2, 2, 3, dtype=torch.complex128)
        matrix = torch.zeros(2, 1, 3, dtype=mean.dtype)
        observation = torch.ones(2, 2, 1, dtype=mean.dtype)
        class RecordingPrior(torch.nn.Module):
            def forward(self, image, time):
                self.image, self.time = image.detach().clone(), time.clone()
                return torch.ones_like(image)
        prior = RecordingPrior()
        actual, final_diagonal = structured_recover(prior, schedule, matrix, observation,
            torch.ones(2), mean, diagonal, iterations=1, strength=.8)
        alpha = schedule.alpha_bar[chosen][:, None, None, None]
        expected = decode(encode(mean) - ((1 - alpha) / alpha).sqrt() * torch.ones_like(encode(mean)))
        self.assertTrue(torch.equal(prior.time, chosen + 1))
        self.assertTrue(torch.allclose(prior.image, alpha.sqrt() * encode(mean), atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.equal(final_diagonal, diagonal * .8))

    def test_predeclared_variance_halves_between_iterations(self):
        schedule = VPSchedule(6, .1, .2)
        ratios = (1 - schedule.alpha_bar) / schedule.alpha_bar
        mean = torch.ones(1, 1, 2, dtype=torch.complex128)
        diagonal = torch.full_like(mean.real, 2 * ratios[4])
        class RecordingPrior(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.times = []
            def forward(self, image, time):
                self.times.append(time.clone())
                return torch.zeros_like(image)
        prior = RecordingPrior()
        _, result_diagonal = structured_recover(prior, schedule, torch.zeros(1, 1, 2, dtype=mean.dtype),
            torch.ones(1, 1, 1, dtype=mean.dtype), torch.ones(1), mean, diagonal, iterations=3)
        expected = [int((ratios - ratios[4] * .5 ** i).abs().argmin()) + 1 for i in range(3)]
        self.assertEqual([int(time[0]) for time in prior.times], expected)
        self.assertTrue(torch.equal(result_diagonal, diagonal / 4))

    def test_zero_initialized_variance_network_equals_fixed_structure(self):
        mean, matrix, observation, variance, diagonal = self.gaussian(torch.complex64)
        prior = DenoisingCNN(width=8).requires_grad_(False)
        schedule = VPSchedule(6)
        fixed = structured_recover(prior, schedule, matrix, observation, variance, mean, diagonal)
        learned = structured_recover(prior, schedule, matrix, observation, variance, mean, diagonal,
            updater=VarianceUpdate(4))
        for expected, actual in zip(fixed, learned):
            self.assertTrue(torch.equal(actual, expected))

    def test_updater_gradient_crosses_frozen_prior_inputs(self):
        mean, matrix, observation, variance, diagonal = self.gaussian(torch.complex64)
        prior = DenoisingCNN(width=8).requires_grad_(False)
        requires_grad = []
        handle = prior.register_forward_pre_hook(lambda module, inputs: requires_grad.append(inputs[0].requires_grad))
        updater = VarianceUpdate(4)
        result, final_diagonal = structured_recover(prior, VPSchedule(6), matrix, observation, variance,
            mean, diagonal, updater=updater, iterations=3)
        (result.abs().square().mean() + final_diagonal.mean()).backward()
        handle.remove()
        self.assertEqual(requires_grad, [False, True, True])
        self.assertTrue(all(parameter.grad is None for parameter in prior.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in updater.parameters() if p.grad is not None), 0.)
        self.assertTrue(torch.isfinite(result).all() and torch.isfinite(final_diagonal).all())

    def test_invalid_gaussian_and_variance_features(self):
        values = self.gaussian(torch.complex64)
        cases = ((0, values[0].real), (1, values[1][..., :2]), (2, values[2][:, :1]),
            (3, -values[3]), (3, torch.tensor([1, 2])), (4, values[4] * float("nan")),
            (4, torch.zeros_like(values[4])), (1, values[1].to(torch.complex128)))
        for index, invalid in cases:
            bad = list(values)
            bad[index] = invalid
            with self.subTest(index=index, shape=invalid.shape), self.assertRaises(ValueError):
                gaussian_update(*bad)
        with self.assertRaises(ValueError):
            VarianceUpdate(0)
        with self.assertRaises(ValueError):
            VarianceUpdate()(values[0], values[4], values[0][:, :1], values[3])

    def test_invalid_lmmse_statistics_and_observations(self):
        training = torch.randn(4, 2, 3, dtype=torch.complex64)
        matrix = torch.randn(2, 1, 3, dtype=torch.complex64)
        for h, a, noise in ((training[:1], matrix, .1), (training.real, matrix, .1),
                (training, matrix[..., :2], .1), (training, matrix, 0.), (training, matrix, float("nan"))):
            with self.subTest(noise=noise), self.assertRaises(ValueError):
                lmmse_statistics(h, a, noise)
        statistics = lmmse_statistics(training, matrix, .1)
        for stats, y in (({}, torch.ones(1, 2, 1, dtype=training.dtype)),
                (statistics, torch.ones(1, 1, 1, dtype=training.dtype)),
                ({**statistics, "gain": statistics["gain"][:, :1]}, torch.ones(1, 2, 1, dtype=training.dtype))):
            with self.assertRaises(ValueError):
                apply_lmmse(stats, y)

    def test_invalid_structured_settings_and_prior(self):
        mean, matrix, observation, variance, diagonal = self.gaussian(torch.complex64)
        prior = DenoisingCNN(width=8).requires_grad_(False)
        for settings in ({"iterations": 0}, {"iterations": True}, {"strength": 0.}, {"strength": float("inf")},
                {"updater": "invalid"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                structured_recover(prior, VPSchedule(6), matrix, observation, variance, mean, diagonal, **settings)
        with self.assertRaises(ValueError):
            structured_recover(DenoisingCNN(width=8), VPSchedule(6), matrix, observation, variance, mean, diagonal)
        schedule = VPSchedule(6)
        schedule.alpha_bar[1] = 1.
        with self.assertRaises(ValueError):
            structured_recover(prior, schedule, matrix, observation, variance, mean, diagonal)


if __name__ == "__main__":
    unittest.main()
