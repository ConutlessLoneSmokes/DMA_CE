"""Paired physical scenes, fixed feasible DMA states, and same-observation coarse estimates."""
import hashlib
import itertools
import time

import numpy as np
import torch

from ._upstream.atmosphere import AbsorptionTable, C0_M_PER_S, free_space_amplitude
from ._upstream.geometry import ofdm_frequencies, ula_positions, los_element_channel, rayleigh_distance_m
from ._upstream.dma import dma_weight_matrix
from .sbl import SOLVER_VERSION, joint_sbl
from .physics import element_channel, paper_channel, thermal_budget


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
    rf = system.get("rf_chains", 1)
    if type(rf) is not int or rf < 1 or n % rf or bank.ndim != 2 or bank.shape[1] != n:
        raise ValueError("positive RF count dividing the antenna count and matching state bank required")
    elements = n // rf
    spacing = system["spacing_wavelengths"] * C0_M_PER_S / carrier
    positions = ula_positions(n, spacing)
    if rf > 1:
        strip_spacing = system["strip_spacing_wavelengths"] * C0_M_PER_S / carrier
        if (not np.isfinite(strip_spacing) or strip_spacing <= 0
                or not np.isfinite(channel["elevation_deg"]) or not -90 < channel["elevation_deg"] < 90):
            raise ValueError("positive strip spacing and fixed elevation in (-90, 90) required")
        positions = np.column_stack((np.tile(ula_positions(elements, spacing), rf), np.zeros(n),
            np.repeat(ula_positions(rf, strip_spacing), elements)))
    frequencies = ofdm_frequencies(carrier, bandwidth, system["tones"])
    feed = np.tile(np.arange(elements) * spacing, rf)
    response = np.stack([dma_weight_matrix(frequencies, row, feed,
        quality_factor=system["quality_factor"], reference_frequency_hz=carrier,
        waveguide_effective_index=system["waveguide_effective_index"],
        waveguide_end_power_fraction=system["waveguide_end_power_fraction"]) for row in bank], axis=1)
    if rf == 1:
        raw = response
    else:
        raw = np.zeros((len(frequencies), len(bank), rf, n), dtype=complex)
        for strip in range(rf):
            span = slice(strip * elements, (strip + 1) * elements)
            raw[:, :, strip, span] = response[:, :, span]
        raw = raw.reshape(len(frequencies), len(bank) * rf, n)
    budget = thermal_budget(system["thermal_noise"], bandwidth, system["tones"]) if "thermal_noise" in system else None
    receiver_fraction = budget["receiver_noise_fraction"] if budget else system["rf_noise_fraction"]
    noise_gain = np.sum(abs(raw) ** 2, axis=-1) + receiver_fraction
    matrix = raw.conj() / np.sqrt(noise_gain)[..., None]
    absorption = None
    if channel.get("include_absorption", False):
        from pathlib import Path
        path = Path(channel["absorption_table_path"])
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != channel["absorption_table_sha256"].lower():
            raise ValueError("absorption table hash differs from the declared source")
        absorption = AbsorptionTable.from_txt(path, max_frequency_hz=1e12)
    angles = np.deg2rad(np.linspace(-channel["angle_limit_deg"], channel["angle_limit_deg"], settings["angle_points"]))
    ranges = 1 / np.linspace(1 / channel["range_max_m"], 1 / channel["range_min_m"], settings["range_points"])
    atoms = []
    for angle, radius in itertools.product(angles, ranges):
        atom = element_channel(frequencies, positions, radius, angle,
            absorption_table=absorption, include_absorption=absorption is not None,
            elevation_deg=channel.get("elevation_deg", 0.))
        reference = free_space_amplitude(frequencies, radius) * np.exp(-2j * np.pi * frequencies * radius / C0_M_PER_S)
        atom = atom / reference[:, None]
        atoms.append(atom / np.linalg.norm(atom, axis=-1, keepdims=True))
    dictionary = np.stack(atoms, axis=-1)
    return {"matrix": torch.tensor(matrix, dtype=torch.complex64),
        "dictionary": torch.tensor(dictionary, dtype=torch.complex64),
        "noise_gain": torch.tensor(noise_gain, dtype=torch.float64),
        "channel_settings": channel, "absorption_table": absorption, "thermal_budget": budget,
        "positions": positions, "frequencies": frequencies,
        "rf_chains": rf, "elements_per_strip": elements,
        "rayleigh_distance_m": (2 * float(np.sum(np.ptp(positions, axis=0)**2)) * carrier / C0_M_PER_S
            if positions.ndim == 2 else rayleigh_distance_m(carrier, positions))}


def sample_scenes(config, split_index, count):
    seed = config["data"].get("screen_seed", config["seed"]) if split_index == 2 else config["seed"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, split_index, 113]))
    channel = config["channel"]
    shape = (count, channel["paths"])
    ranges = rng.uniform(channel["range_min_m"], channel["range_max_m"], shape)
    angles = np.deg2rad(rng.uniform(-channel["angle_limit_deg"], channel["angle_limit_deg"], shape))
    gains = (np.ones(shape, dtype=np.complex128) if channel.get("model", "gaussian") == "li2026" else
        (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2 * channel["paths"]))
    return ranges, angles, gains


def physical_channels(hardware, scenes):
    settings = hardware.get("channel_settings", {})
    if settings.get("model", "gaussian") == "li2026":
        return torch.tensor(paper_channel(hardware, scenes), dtype=torch.complex64)
    ranges, angles, gains = scenes
    channels = []
    for radius, angle, gain in zip(ranges, angles, gains):
        paths = [g * element_channel(hardware["frequencies"], hardware["positions"], r, a,
            absorption_table=hardware.get("absorption_table"),
            include_absorption=hardware.get("absorption_table") is not None,
            elevation_deg=settings.get("elevation_deg", 0.)) for r, a, g in zip(radius, angle, gain)]
        channels.append(np.sum(paths, axis=0) * 10 ** (settings.get("antenna_gain_dbi", 0.) / 20))
    return torch.tensor(np.stack(channels), dtype=torch.complex64)


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


CALIBRATION_KEYS = ("reliability_scale", "residual_real_scale", "dc_reference_complex_variance",
    "reliability_log_mean", "reliability_log_std", "calibration_edges")


def estimate_entry(entry, context, dictionary, settings, device, log):
    if (type(settings["iterations"]) is not int or settings["iterations"] < 1
            or type(settings["batch_size"]) is not int or settings["batch_size"] < 1
            or not np.isfinite(settings["tolerance"]) or settings["tolerance"] <= 0
            or type(settings.get("early_stopping", True)) is not bool):
        raise ValueError("positive SBL iterations, batch size and finite tolerance required")
    sensing = context["matrix"].to(device) @ dictionary
    output = {key: [] for key in ("coarse", "reliability", "sbl_iterations", "sbl_converged", "sbl_relative_change",
        "sbl_optimality_residual", "sbl_objective_per_tone")}
    sync(device)
    started = time.perf_counter()
    for start in range(0, len(entry["observation"]), settings["batch_size"]):
        stop = start + settings["batch_size"]
        mean, diagonal, info = joint_sbl(entry["observation"][start:stop].to(device), sensing, dictionary,
            entry["variance"][start:stop].to(device), iterations=settings["iterations"], tolerance=settings["tolerance"],
            early_stopping=settings.get("early_stopping", True))
        for key, value in zip(output, (mean, diagonal, info["iterations"], info["converged"], info["relative_change"],
                info["optimality_residual"], info["objective_per_tone"])):
            output[key].append(value.cpu())
        log("coarse_batch", context=entry["context_index"], completed=min(stop, len(entry["observation"])))
    sync(device)
    return {**entry, **{key: torch.cat(values) for key, values in output.items()},
        "sbl_solver": SOLVER_VERSION, "sbl_early_stopping": settings.get("early_stopping", True),
        "coarse_ms_per_sample": (time.perf_counter() - started) * 1000 / len(entry["observation"])}


def calibrate(result):
    train = result["splits"]["train"]
    mean_error = torch.stack([(x["truth"] - x["coarse"]).abs().square().mean() for x in train]).mean()
    predicted = torch.stack([x["reliability"].mean() for x in train]).mean()
    result["reliability_scale"] = float(mean_error / predicted.clamp_min(1e-12))
    result["residual_real_scale"] = float((mean_error / 2).clamp_min(1e-12).sqrt())
    result["dc_reference_complex_variance"] = float(mean_error)
    logs = torch.cat([(x["reliability"] * result["reliability_scale"]).clamp_min(1e-10).log10().flatten() for x in train])
    result["reliability_log_mean"], result["reliability_log_std"] = float(logs.mean()), float(logs.std().clamp_min(1e-3))
    # torch.quantile rejects more than 2**24 entries, exceeded by the 128 x 128 cache.
    result["calibration_edges"] = np.quantile(logs.numpy(), np.linspace(0, 1, 6)).tolist()
    return result


def reestimate(config, cache, device, log, splits=None):
    """Change only the estimator; all CSI, observations, noise, states and physical scales stay fixed."""
    result = {**cache, "splits": {}}
    hardware = {}
    for split in (splits or tuple(cache["splits"])):
        if len(cache["splits"][split]) != len(cache["contexts"]):
            raise ValueError("one observation entry per known hardware context required")
        result["splits"][split] = []
        for entry, context in zip(cache["splits"][split], cache["contexts"]):
            band = context["bandwidth_hz"]
            if band not in hardware:
                hardware[band] = build_hardware(config, band, cache["state_bank"].numpy())["dictionary"].to(device)
            result["splits"][split].append(estimate_entry(entry, context, hardware[band], config["sbl"], device, log))
    return calibrate(result) if "train" in result["splits"] else result


def prepare(config, device, log, *, split_names=("train", "validation", "screen"), reference=None, estimate=True):
    if (not split_names or len(set(split_names)) != len(split_names)
            or any(s not in ("train", "validation", "screen") for s in split_names)
            or ("train" not in split_names and reference is None)
            or (not estimate and (reference is None or "train" in split_names))):
        raise ValueError("valid splits and a training reference for evaluation-only data required")
    system = config["system"]
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 41]))
    bank = rng.uniform(system["carrier_hz"] - system["tuning_bandwidth_hz"] / 2,
        system["carrier_hz"] + system["tuning_bandwidth_hz"] / 2,
        (max(system["configurations"]), system["antennas"]))
    if reference is not None:
        bank = reference["state_bank"].numpy()
        if (bank.shape != (max(system["configurations"]), system["antennas"])
                or len(reference["contexts"]) != len(system["bandwidths_hz"]) * len(system["configurations"]) * len(system["snrs_db"])):
            raise ValueError("training reference does not match the requested hardware")
    hardware = [build_hardware(config, bandwidth, bank) for bandwidth in system["bandwidths_hz"]]
    scenes, physical = {}, {}
    for split in split_names:
        split_index = ("train", "validation", "screen").index(split)
        count = config["data"][split]
        scenes[split] = sample_scenes(config, split_index, count)
        physical[split] = [physical_channels(item, scenes[split]) for item in hardware]
    scale = (reference["channel_scale"] if reference is not None else
        float(torch.stack([h.abs().square().mean() for h in physical["train"]]).mean().sqrt() / np.sqrt(2)))
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
            measurements = configurations * hw["rf_chains"]
            matrix = hw["matrix"][:, :measurements].contiguous()
            if reference is None:
                training_h = physical["train"][band_index] / scale
                train_y = torch.einsum("kjn,bkn->bkj", matrix, training_h)
                power = float(train_y.abs().square().mean())
            else:
                known = reference["contexts"][len(result["contexts"])]
                if (known["bandwidth_hz"] != system["bandwidths_hz"][band_index]
                        or known["configurations"] != configurations or known["snr_db"] != snr_db
                        or tuple(known["matrix"].shape) != tuple(matrix.shape)):
                    raise ValueError("evaluation context does not match the training reference")
                power = known["training_white_power"]
            variance = power / 10 ** (snr_db / 10)
            context_index = len(result["contexts"])
            result["contexts"].append({"bandwidth_hz": system["bandwidths_hz"][band_index],
                "configurations": configurations, "snr_db": snr_db,
                "pilot_re": configurations * system["tones"], "rf_chains": hw["rf_chains"],
                "measurements_per_tone": measurements,
                "training_symbols": configurations, "switches": configurations - 1,
                "training_white_power": power, "noise_complex_variance": variance,
                "physical_element_noise_variance": variance * scale ** 2,
                "matrix": matrix, "matrix_sha256": tensor_hash(matrix),
                "raw_noise_gain": hw["noise_gain"][:, :measurements],
                "rayleigh_distance_m": hw["rayleigh_distance_m"]})
            for split in split_names:
                split_index = ("train", "validation", "screen").index(split)
                count = config["data"][split]
                truth = physical[split][band_index] / scale
                clean = torch.einsum("kjn,bkn->bkj", matrix, truth)
                seed = config["data"].get("screen_seed", config["seed"]) if split == "screen" else config["seed"]
                noise_rng = np.random.default_rng(np.random.SeedSequence([seed, split_index, band_index, 811]))
                shape = (count, system["tones"], len(bank) * hw["rf_chains"])
                standard = (noise_rng.normal(size=shape) + 1j * noise_rng.normal(size=shape)) / np.sqrt(2)
                noise = torch.tensor(standard[:, :, :measurements], dtype=torch.complex64) * np.sqrt(variance)
                observed = clean + noise
                entry = {"truth": truth, "observation": observed,
                    "variance": torch.full((count,), variance), "context_index": context_index,
                    "realized_snr_db": 10 * torch.log10(clean.abs().square().sum((1, 2)) / noise.abs().square().sum((1, 2)))}
                if estimate:
                    entry = estimate_entry(entry, result["contexts"][-1], dictionary, config["sbl"], device, log)
                    log("coarse", context=context_index, split=split,
                        converged=int(entry["sbl_converged"].sum()), samples=count)
                result["splits"][split].append(entry)
    if "train" in split_names:
        return calibrate(result)
    result.update({key: reference[key] for key in CALIBRATION_KEYS})
    return result
