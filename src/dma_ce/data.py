"""Paired physical scenes, fixed feasible DMA states, and same-observation coarse estimates."""
import hashlib
import itertools
import time

import numpy as np
import torch

from ._upstream.atmosphere import C0_M_PER_S, free_space_amplitude
from ._upstream.geometry import ofdm_frequencies, ula_positions, los_element_channel, rayleigh_distance_m
from ._upstream.dma import dma_weight_matrix
from .sbl import joint_sbl


def encode(channel):
    return torch.stack((channel.real, channel.imag), dim=1).transpose(-1, -2).contiguous()


def decode(image):
    return torch.complex(image[:, 0], image[:, 1]).transpose(-1, -2).contiguous()


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def build_hardware(config, bandwidth, bank):
    system, channel, settings = config["system"], config["channel"], config["sbl"]
    carrier, n = system["carrier_hz"], system["antennas"]
    spacing = system["spacing_wavelengths"] * C0_M_PER_S / carrier
    positions = ula_positions(n, spacing)
    frequencies = ofdm_frequencies(carrier, bandwidth, system["tones"])
    feed = np.arange(n) * spacing
    raw = np.stack([dma_weight_matrix(frequencies, row, feed,
        quality_factor=system["quality_factor"], reference_frequency_hz=carrier,
        waveguide_effective_index=system["waveguide_effective_index"],
        waveguide_end_power_fraction=system["waveguide_end_power_fraction"]) for row in bank], axis=1)
    noise_gain = np.sum(abs(raw) ** 2, axis=-1) + system["rf_noise_fraction"]
    matrix = raw.conj() / np.sqrt(noise_gain)[..., None]
    angles = np.deg2rad(np.linspace(-channel["angle_limit_deg"], channel["angle_limit_deg"], settings["angle_points"]))
    ranges = 1 / np.linspace(1 / channel["range_max_m"], 1 / channel["range_min_m"], settings["range_points"])
    atoms = []
    for angle, radius in itertools.product(angles, ranges):
        atom = los_element_channel(frequencies, positions, radius, angle, include_absorption=False)
        reference = free_space_amplitude(frequencies, radius) * np.exp(-2j * np.pi * frequencies * radius / C0_M_PER_S)
        atom = atom / reference[:, None]
        atoms.append(atom / np.linalg.norm(atom, axis=-1, keepdims=True))
    dictionary = np.stack(atoms, axis=-1)
    return {"matrix": torch.tensor(matrix, dtype=torch.complex64),
        "dictionary": torch.tensor(dictionary, dtype=torch.complex64),
        "noise_gain": torch.tensor(noise_gain, dtype=torch.float64),
        "positions": positions, "frequencies": frequencies,
        "rayleigh_distance_m": rayleigh_distance_m(carrier, positions)}


def sample_scenes(config, split_index, count):
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], split_index, 113]))
    channel = config["channel"]
    shape = (count, channel["paths"])
    ranges = rng.uniform(channel["range_min_m"], channel["range_max_m"], shape)
    angles = np.deg2rad(rng.uniform(-channel["angle_limit_deg"], channel["angle_limit_deg"], shape))
    gains = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2 * channel["paths"])
    return ranges, angles, gains


def physical_channels(hardware, scenes):
    ranges, angles, gains = scenes
    channels = []
    for radius, angle, gain in zip(ranges, angles, gains):
        paths = [g * los_element_channel(hardware["frequencies"], hardware["positions"], r, a,
            include_absorption=False) for r, a, g in zip(radius, angle, gain)]
        channels.append(np.sum(paths, axis=0))
    return torch.tensor(np.stack(channels), dtype=torch.complex64)


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def prepare(config, device, log):
    system = config["system"]
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 41]))
    bank = rng.uniform(system["carrier_hz"] - system["tuning_bandwidth_hz"] / 2,
        system["carrier_hz"] + system["tuning_bandwidth_hz"] / 2,
        (max(system["configurations"]), system["antennas"]))
    hardware = [build_hardware(config, bandwidth, bank) for bandwidth in system["bandwidths_hz"]]
    scenes, physical = {}, {}
    for split_index, split in enumerate(("train", "validation", "screen")):
        count = config["data"][split]
        scenes[split] = sample_scenes(config, split_index, count)
        physical[split] = [physical_channels(item, scenes[split]) for item in hardware]
    scale = float(torch.stack([h.abs().square().mean() for h in physical["train"]]).mean().sqrt() / np.sqrt(2))
    if not np.isfinite(scale) or scale <= 0:
        raise ArithmeticError("invalid training-only global channel scale")
    result = {"contexts": [], "splits": {}, "channel_scale": scale,
        "state_bank": torch.tensor(bank), "geometry_hashes": {}}
    for split, values in scenes.items():
        packed = np.concatenate((values[0], values[1], values[2].real, values[2].imag), axis=1)
        result["geometry_hashes"][split] = hashlib.sha256(packed.tobytes()).hexdigest()
        result["splits"][split] = []
    if len(set(result["geometry_hashes"].values())) != len(scenes):
        raise AssertionError("scene splits overlap")
    for band_index, hw in enumerate(hardware):
        dictionary = hw["dictionary"].to(device)
        for configurations, snr_db in itertools.product(system["configurations"], system["snrs_db"]):
            matrix = hw["matrix"][:, :configurations].contiguous()
            training_h = physical["train"][band_index] / scale
            train_y = torch.einsum("kjn,bkn->bkj", matrix, training_h)
            power = float(train_y.abs().square().mean())
            variance = power / 10 ** (snr_db / 10)
            context_index = len(result["contexts"])
            result["contexts"].append({"bandwidth_hz": system["bandwidths_hz"][band_index],
                "configurations": configurations, "snr_db": snr_db,
                "pilot_re": configurations * system["tones"], "rf_chains": 1,
                "training_symbols": configurations, "switches": configurations - 1,
                "training_white_power": power, "noise_complex_variance": variance,
                "physical_element_noise_variance": variance * scale ** 2,
                "matrix": matrix, "matrix_sha256": tensor_hash(matrix),
                "raw_noise_gain": hw["noise_gain"][:, :configurations],
                "rayleigh_distance_m": hw["rayleigh_distance_m"]})
            sensing = torch.einsum("kjn,kng->kjg", matrix.to(device), dictionary)
            for split_index, split in enumerate(("train", "validation", "screen")):
                count = config["data"][split]
                truth = physical[split][band_index] / scale
                clean = torch.einsum("kjn,bkn->bkj", matrix, truth)
                noise_rng = np.random.default_rng(np.random.SeedSequence([config["seed"], split_index, band_index, 811]))
                shape = (count, system["tones"], len(bank))
                standard = (noise_rng.normal(size=shape) + 1j * noise_rng.normal(size=shape)) / np.sqrt(2)
                noise = torch.tensor(standard[:, :, :configurations], dtype=torch.complex64) * np.sqrt(variance)
                observed = clean + noise
                coarse, reliability, iteration, converged, changes = [], [], [], [], []
                sync(device)
                started = time.perf_counter()
                for start in range(0, count, config["sbl"]["batch_size"]):
                    batch_y = observed[start:start + config["sbl"]["batch_size"]].to(device)
                    batch_variance = torch.full((len(batch_y),), variance, device=device)
                    mean, diagonal, diagnostics = joint_sbl(batch_y, sensing, dictionary, batch_variance,
                        iterations=config["sbl"]["iterations"], tolerance=config["sbl"]["tolerance"])
                    coarse.append(mean.cpu()); reliability.append(diagonal.cpu())
                    iteration.append(diagnostics["iterations"].cpu())
                    converged.append(diagnostics["converged"].cpu())
                    changes.append(diagnostics["relative_change"].cpu())
                    log("coarse_batch", context=context_index, split=split, completed=min(start + len(batch_y), count), samples=count)
                sync(device)
                elapsed = time.perf_counter() - started
                result["splits"][split].append({"truth": truth, "coarse": torch.cat(coarse),
                    "reliability": torch.cat(reliability), "observation": observed,
                    "variance": torch.full((count,), variance), "context_index": context_index,
                    "sbl_iterations": torch.cat(iteration), "sbl_converged": torch.cat(converged),
                    "sbl_relative_change": torch.cat(changes), "coarse_ms_per_sample": elapsed * 1000 / count,
                    "realized_snr_db": 10 * torch.log10(clean.abs().square().sum((1, 2)) / noise.abs().square().sum((1, 2)))})
                log("coarse", context=context_index, split=split, seconds=round(elapsed, 3),
                    converged=int(torch.cat(converged).sum()), samples=count)
    train = result["splits"]["train"]
    mean_error = torch.stack([(x["truth"] - x["coarse"]).abs().square().mean() for x in train]).mean()
    predicted = torch.stack([x["reliability"].mean() for x in train]).mean()
    result["reliability_scale"] = float(mean_error / predicted.clamp_min(1e-12))
    result["residual_real_scale"] = float((mean_error / 2).sqrt())
    result["dc_reference_complex_variance"] = float(mean_error)
    logs = torch.cat([(x["reliability"] * result["reliability_scale"]).clamp_min(1e-10).log10().flatten() for x in train])
    result["reliability_log_mean"], result["reliability_log_std"] = float(logs.mean()), float(logs.std().clamp_min(1e-3))
    result["calibration_edges"] = torch.quantile(logs, torch.linspace(0, 1, 6)).tolist()
    return result
