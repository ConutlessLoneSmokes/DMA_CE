"""Thermal link budgets and Li et al. (2026), equations (11), (18)--(21)."""
import math

import numpy as np

from ._upstream.atmosphere import C0_M_PER_S, free_space_amplitude
from ._upstream.geometry import los_element_channel


BOLTZMANN_J_PER_K = 1.380649e-23


def scene_points(radius, angle, elevation_deg):
    elevation = np.deg2rad(elevation_deg)
    return np.stack((radius * np.sin(angle) * np.cos(elevation),
        radius * np.cos(angle) * np.cos(elevation), radius * np.sin(elevation)), axis=-1)


def element_channel(frequency, positions, radius, angle, *, absorption_table=None,
        include_absorption=True, elevation_deg=0.):
    if positions.ndim == 1:
        return los_element_channel(frequency, positions, radius, angle,
            absorption_table=absorption_table, include_absorption=include_absorption)
    distance = np.linalg.norm(positions - scene_points(np.asarray(radius), np.asarray(angle), elevation_deg), axis=-1)
    if (distance <= 0).any():
        raise ValueError("source cannot coincide with an array element")
    amplitude = free_space_amplitude(frequency[:, None], distance[None])
    if include_absorption:
        if absorption_table is None:
            raise ValueError("absorption table required")
        amplitude *= np.exp(-.5 * absorption_table.coefficient(frequency)[:, None] * distance[None])
    return amplitude * np.exp(-2j * np.pi * frequency[:, None] * distance[None] / C0_M_PER_S)


def thermal_budget(settings, bandwidth_hz, tones):
    """Power-normalized subcarriers: sum_k N_k = k_B T B, sum_k P_k = P_tx."""
    antenna = settings["antenna_temperature_k"]
    reference = settings["receiver_reference_temperature_k"]
    figure = settings["noise_figure_db"]
    if (not all(math.isfinite(x) for x in (antenna, reference, figure, bandwidth_hz))
            or min(antenna, reference, bandwidth_hz) <= 0 or figure < 0
            or type(tones) is not int or tones < 1):
        raise ValueError("positive temperatures, bandwidth and tones, and nonnegative noise figure required")
    receiver = reference * (10 ** (figure / 10) - 1)
    spacing = bandwidth_hz / tones
    return {"subcarrier_bandwidth_hz": spacing, "antenna_temperature_k": antenna,
        "receiver_equivalent_temperature_k": receiver, "receiver_noise_fraction": receiver / antenna,
        "antenna_noise_per_tone_w": BOLTZMANN_J_PER_K * antenna * spacing,
        "receiver_noise_per_tone_w": BOLTZMANN_J_PER_K * receiver * spacing,
        "antenna_noise_full_band_w": BOLTZMANN_J_PER_K * antenna * bandwidth_hz}


def reflection_magnitude(frequencies_hz, incidence_rad, refractive_index, roughness_m):
    """TE Fresnel magnitude times Rayleigh roughness; a declared, constant material index."""
    frequency = np.asarray(frequencies_hz, dtype=float)
    if (not np.isfinite(frequency).all() or (frequency <= 0).any()
            or not all(math.isfinite(x) for x in (incidence_rad, refractive_index, roughness_m))
            or not 0 <= incidence_rad <= np.pi / 2 or refractive_index < 1 or roughness_m < 0):
        raise ValueError("positive frequencies and valid incidence, refractive index and roughness required")
    if refractive_index == 1:
        return np.zeros_like(frequency)
    cosine = np.cos(incidence_rad)
    refracted_cosine = np.sqrt(1 - (np.sin(incidence_rad) / refractive_index) ** 2)
    fresnel = (cosine - refractive_index * refracted_cosine) / (cosine + refractive_index * refracted_cosine)
    roughness = np.exp(-8 * (np.pi * frequency * roughness_m * cosine / C0_M_PER_S) ** 2)
    return abs(fresnel) * roughness


def paper_channel(hardware, scenes):
    """One direct path plus single-bounce paths; no random Rayleigh path multiplier."""
    radii, angles, gains = scenes
    settings = hardware["channel_settings"]
    frequency, positions = hardware["frequencies"], hardware["positions"]
    table = hardware["absorption_table"]
    kappa = np.zeros_like(frequency) if table is None else table.coefficient(frequency)
    gain = 10 ** (settings.get("antenna_gain_dbi", 0.) / 20) / np.sqrt(radii.shape[1])
    output = []
    for radius, angle, coefficients in zip(radii, angles, gains):
        points = (scene_points(radius, angle, settings["elevation_deg"]) if positions.ndim == 2 else
            np.column_stack((radius * np.sin(angle), radius * np.cos(angle))))
        transmitter = points[0]
        channel = np.zeros((len(frequency), len(positions)), dtype=np.complex128)
        for path, point in enumerate(points):
            before = 0. if path == 0 else np.linalg.norm(transmitter - point)
            after = (np.linalg.norm(positions - point, axis=-1) if positions.ndim == 2 else
                np.hypot(positions - point[0], point[1]))
            total = before + after
            reflection = np.ones_like(frequency)
            if path:
                if before <= np.finfo(float).eps:
                    raise ValueError("a reflector cannot coincide with the transmitter")
                cosine = (before ** 2 + radius[path] ** 2 - radius[0] ** 2) / (2 * before * radius[path])
                incidence = .5 * np.arccos(np.clip(cosine, -1, 1))
                reflection = reflection_magnitude(frequency, incidence,
                    settings["reflection_refractive_index"], settings["surface_roughness_m"])
            amplitude = free_space_amplitude(frequency[:, None], total[None])
            amplitude *= np.exp(-.5 * kappa[:, None] * total[None]) * reflection[:, None]
            channel += coefficients[path] * amplitude * np.exp(
                -2j * np.pi * frequency[:, None] * total[None] / C0_M_PER_S)
        output.append(gain * channel)
    return np.stack(output)
