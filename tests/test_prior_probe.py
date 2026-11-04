import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dma_ce.physics import paper_channel, scene_points
from dma_ce.prior_probe import covariance_sbl, fit_paths, initial_geometry, propagation_basis, refine_geometry
from dma_ce.sbl import joint_sbl


class PriorProbeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.dtype = torch.complex128
        self.dictionary = torch.randn(3, 5, 7, dtype=self.dtype)
        self.matrix = torch.randn(3, 2, 5, dtype=self.dtype)
        self.y = torch.randn(2, 3, 2, dtype=self.dtype)
        self.noise = torch.tensor([.2, .4], dtype=torch.float64)
        self.factor = torch.randn(3, 5, 2, dtype=self.dtype)

    def test_zero_covariance_component_reproduces_original_sbl(self):
        actual = covariance_sbl(self.y, self.matrix, self.dictionary, self.noise, self.factor, 0., 8)
        expected, _, info = joint_sbl(self.y, self.matrix @ self.dictionary, self.dictionary,
            self.noise, iterations=8, tolerance=1e-3, early_stopping=False)
        torch.testing.assert_close(actual['mean'], expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual['objective_per_tone'], info['objective_per_tone'])
        torch.testing.assert_close(actual['optimality_residual'], info['optimality_residual'])

    def test_augmented_posterior_matches_dense_gaussian_conditioning(self):
        result = covariance_sbl(self.y, self.matrix, self.dictionary, self.noise, self.factor, .3, 4)
        for batch in range(len(self.y)):
            covariance = (self.dictionary @ torch.diag(result['gamma'][batch]).to(self.dtype)
                @ self.dictionary.mH+.3*self.factor @ self.factor.mH)
            cross = covariance @ self.matrix.mH
            observed = self.matrix @ cross+self.noise[batch]*torch.eye(2, dtype=self.dtype)
            expected = (cross @ torch.linalg.solve(observed, self.y[batch, ..., None])).squeeze(-1)
            torch.testing.assert_close(result['mean'][batch], expected, rtol=1e-10, atol=1e-10)

    def test_augmented_evidence_decreases_with_fixed_statistical_component(self):
        objectives = [covariance_sbl(self.y, self.matrix, self.dictionary, self.noise,
            self.factor, .3, n)['objective_per_tone'] for n in (1, 3, 8)]
        self.assertTrue(all((b <= a+1e-10).all() for a, b in zip(objectives, objectives[1:])))

    def physical_inputs(self, paths=3):
        f = np.linspace(298.5e9, 301.5e9, 8)
        positions = np.column_stack((np.linspace(-.008, .008, 12), np.zeros(12), np.zeros(12)))
        settings = {'paths': paths, 'elevation_deg': 30., 'reflection_refractive_index': 1.7,
            'surface_roughness_m': 1e-5, 'range_min_m': .5, 'range_max_m': 4., 'angle_limit_deg': 60.}
        geometry = torch.tensor([[[-.3, .7], [.2, 1.3], [.8, 1.8]]], dtype=torch.float64)[:, :paths]
        return geometry, torch.tensor(f), torch.tensor(positions), torch.zeros(len(f), dtype=torch.float64), settings

    def test_torch_physical_paths_match_independent_numpy_channel(self):
        geometry, f, positions, kappa, settings = self.physical_inputs()
        radius, angle = geometry[..., 1].numpy(), geometry[..., 0].numpy()
        points = scene_points(radius, angle, settings['elevation_deg'])
        before = np.linalg.norm(points-points[:, :1], axis=-1)
        gains = np.ones_like(radius, dtype=np.complex128)
        expected = paper_channel({'channel_settings': settings, 'frequencies': f.numpy(),
            'positions': positions.numpy(), 'absorption_table': None}, (radius, angle, gains))
        parts = propagation_basis(geometry, f, positions, kappa, settings, 1.)
        coefficient = torch.exp(-2j*math.pi*float(f.mean())*torch.tensor(before+radius)/299792458.)
        actual = torch.einsum('bknl,bl->bkn', parts, coefficient)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-8, atol=1e-13)

    def test_physical_geometry_has_correct_differentiable_response(self):
        geometry, f, positions, kappa, settings = self.physical_inputs()
        geometry.requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(
            lambda g: propagation_basis(g, f, positions, kappa, settings, 1e-5),
            (geometry,), eps=1e-7, atol=1e-4, rtol=1e-3))

    def test_reflector_coincident_with_transmitter_is_rejected_like_original_model(self):
        geometry, f, positions, kappa, settings = self.physical_inputs()
        geometry[:, 1] = geometry[:, 0]
        hardware = {'channel_settings': settings, 'frequencies': f.numpy(),
            'positions': positions.numpy(), 'absorption_table': None}
        with self.assertRaisesRegex(ValueError, 'reflector cannot coincide'):
            paper_channel(hardware, (geometry[..., 1].numpy(), geometry[..., 0].numpy(),
                np.ones((1, settings['paths']), dtype=np.complex128)))
        with self.assertRaisesRegex(ValueError, 'reflector cannot coincide'):
            propagation_basis(geometry, f, positions, kappa, settings, 1.)

    def test_both_path_coefficient_models_recover_a_known_noiseless_channel(self):
        parts = torch.randn(2, 4, 6, 3, dtype=self.dtype)
        matrix = torch.eye(6, dtype=self.dtype).expand(4, -1, -1)
        coefficient = torch.tensor([[1+.2j, -.3+.7j, .4-.1j]], dtype=self.dtype).expand(2, -1)
        truth = torch.einsum('bknl,bl->bkn', parts, coefficient)
        for shared in (False, True):
            estimate, loss, _ = fit_paths(parts, matrix, truth, shared)
            torch.testing.assert_close(estimate, truth, rtol=1e-6, atol=1e-6)
            self.assertTrue((loss < 1e-10).all())

    def test_refinement_improves_a_known_nearby_geometry_without_truth_input(self):
        geometry, f, positions, kappa, settings = self.physical_inputs(paths=1)
        matrix = torch.eye(len(positions), dtype=self.dtype).expand(len(f), -1, -1)
        truth = propagation_basis(geometry, f, positions, kappa, settings, 1e-5).sum(-1)
        initial = geometry+torch.tensor([[[.015, .0001]]])
        result = refine_geometry(initial, matrix, truth, f, positions, kappa, settings, 1e-5,
            shared=True, steps=35, learning_rate=.0005)
        self.assertTrue((result['observation_loss'] < .9*result['initial_observation_loss']).all())
        self.assertTrue(((result['geometry'][..., 1] >= .5) & (result['geometry'][..., 1] <= 4)).all())

    def test_peak_selection_is_distinct_even_with_coherent_atoms(self):
        config = {'channel': {'paths': 3, 'angle_limit_deg': 60., 'range_min_m': .5, 'range_max_m': 4.},
            'sbl': {'angle_points': 2, 'range_points': 2}}
        dictionary = torch.ones(2, 1, 4, dtype=self.dtype)
        geometry, indices = initial_geometry(torch.tensor([[4., 3., 2., 1.]]), dictionary, config)
        self.assertEqual(indices.tolist(), [[0, 1, 2]])
        self.assertTrue(torch.isfinite(geometry).all())

    def test_invalid_covariance_strength_is_rejected(self):
        for value in (-.1, float('nan')):
            with self.assertRaises(ValueError):
                covariance_sbl(self.y, self.matrix, self.dictionary, self.noise, self.factor, value, 1)


if __name__ == '__main__':
    unittest.main()
