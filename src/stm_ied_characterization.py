# -*- coding: utf-8 -*-
"""
IED_characterization.py

Goal
----
Run your STM32-like IED detector independently on hippocampus and thalamus
channels, generate IED timestamps for both channels, and classify each
hippocampal IED as:

    - hpc_to_thal_recruited
    - hpc_only
    - thal_to_hpc_preceded
    - simultaneous_or_volume_conduction
    - ambiguous / artifact-ready flags later

This file assumes your detector class/functions are in:

    offline_ied_detector_test.py

and specifically reuses:

    offline_ied_detector_test.run_detector()
    offline_ied_detector_test.simulate_signed_24bit_transfer()

Your detector module should be in the same folder as this file, or otherwise
available on the Python path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyabf
from scipy.signal import butter, sosfiltfilt, sosfilt, hilbert

import offline_ied_detector_test as iedmod

# ============================================================
# User settings
# ============================================================

ABF_DIR = Path.cwd().parent / "IED_Data"
ABF_PATH = None  # set to r"C:\path\to\file.abf" for one specific file

PROCESS_ALL_ABFS = False
SWEEP_INDEX = 0

# Channel indices in ABF
HPC_CHANNEL = 0
THAL_CHANNEL = 1

HPC_LABEL = "Hippocampus"
THAL_LABEL = "Thalamus"

# Detector preprocessing
FILTER_LOW_HZ = 50.0
FILTER_HIGH_HZ = 85.0
FILTER_ORDER = 4
USE_ZERO_PHASE_FILTER = True
STREAM_FILTERED_SIGNAL = True

# Simulate your Python -> UART -> STM32 int24/int32 transfer
SIMULATE_UART_24BIT_SCALING = True
GAIN = 1000

# Event-time refinement
# The detector timestamp is the threshold-crossing/decision time.
# We optionally refine it to a local raw/filtered peak or max slope.
REFINE_EVENT_TIMES = True
REFINE_SIGNAL_SOURCE = "filtered"  # "raw" or "filtered"
REFINE_METHOD = "max_abs_peak"     # "max_abs_peak", "max_slope", "positive_peak", "negative_peak"
REFINE_SEARCH_BEFORE_S = 0.025
REFINE_SEARCH_AFTER_S = 0.025

# Physiological matching window
# Exclude very short latencies to avoid volume conduction / zero-lag common events.
MIN_ABS_LATENCY_S = 0.005    # 10 ms exclusion around zero-lag
MAX_ABS_LATENCY_S = 0.150    # +/- 500 ms physiological search range

# Optional diagnostic plot
MAKE_PLOTS = True
PLOT_START_S = 500.0
PLOT_END_S = 540.0

OUTPUT_DIR = Path.cwd() / "IED_characterization_results"

# Optional hippocampal IED-centered time-frequency plots.
# These should be computed from the RAW hippocampal signal, not the already
# 50-85 Hz filtered detector signal, otherwise you only see the band you imposed.
MAKE_TIME_FREQUENCY_PLOTS = True
TF_MAX_EVENTS_TO_PLOT = 50

# Epoch around each detected hippocampal IED
TF_PRE_S = 0.500
TF_POST_S = 0.500

# Offline exploratory frequency range.
# Keep this wider than your embedded detector band so you can justify/refine it.
TF_FREQS_HZ = np.arange(5.0, 201.0, 2.0)
TF_BANDWIDTH_HZ = 5.0

# Baseline is relative to each IED time.
# This must sit inside [-TF_PRE_S, +TF_POST_S].
TF_BASELINE_WINDOW_S = (-0.450, -0.350)

# "z" = robust baseline-normalized power: power > median + 5*MAD-scaled SD.
#       This is closest to your embedded threshold idea.
# "ratio" = power / median baseline power; threshold 5.0 literally means 5x baseline.
TF_THRESHOLD_MODE = "z"
TF_THRESHOLD = 5.0

# Percentile clipping only affects the heatmap color range, not the threshold contour.
TF_COLOR_PERCENTILE_CLIP = (1, 99)


# ============================================================
# Data containers
# ============================================================

@dataclass
class ChannelDetectionResult:
    label: str
    channel_index: int
    raw: np.ndarray
    filtered: np.ndarray
    stream_signal: np.ndarray
    detector_events_s: np.ndarray
    refined_events_s: np.ndarray
    debug: dict


# ============================================================
# ABF loading
# ============================================================

def find_abf_files() -> list[Path]:
    """Return ABF files to process."""
    if ABF_PATH is not None:
        return [Path(ABF_PATH)]

    abf_files = []
    for dirpath, _, filenames in os.walk(ABF_DIR):
        for filename in filenames:
            if filename.lower().endswith(".abf"):
                abf_files.append(Path(dirpath) / filename)

    abf_files = sorted(abf_files)

    if not abf_files:
        raise FileNotFoundError(f"No ABF files found in {ABF_DIR}")

    if PROCESS_ALL_ABFS:
        return abf_files

    return [abf_files[0]]


def load_abf_channel(
    abf: pyabf.ABF,
    sweep_index: int,
    channel_index: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Load one channel from one ABF sweep.

    Returns
    -------
    t_s : np.ndarray
        Time vector in seconds.
    y : np.ndarray
        Signal vector.
    fs : float
        Sampling frequency.
    """
    abf.setSweep(sweep_index, channel=channel_index)
    y = abf.sweepY.copy()
    t_s = abf.sweepX.copy()
    fs = float(abf.dataRate)
    return t_s, y, fs


# ============================================================
# Preprocessing and detector wrapper
# ============================================================

def filter_signal_for_detector(raw: np.ndarray, fs: float) -> np.ndarray:
    """Apply the same Python bandpass used before sending/debugging detector input."""
    if not STREAM_FILTERED_SIGNAL:
        return raw.astype(float)

    sos = butter(
        FILTER_ORDER,
        [FILTER_LOW_HZ, FILTER_HIGH_HZ],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    if USE_ZERO_PHASE_FILTER:
        return sosfiltfilt(sos, raw)
    else:
        return sosfilt(sos, raw)


def make_stream_signal(signal_float: np.ndarray) -> np.ndarray:
    """
    Make the actual signal passed into your detector.

    This can simulate the Python float -> 24-bit UART integer transfer.
    """
    if SIMULATE_UART_24BIT_SCALING:
        return iedmod.simulate_signed_24bit_transfer(signal_float, GAIN)
    return signal_float.astype(float)


def detect_ieds_on_channel(
    raw: np.ndarray,
    fs: float,
    channel_index: int,
    label: str,
) -> ChannelDetectionResult:
    """
    Filter one channel, run your detector, and optionally refine event times.
    """
    filtered = filter_signal_for_detector(raw, fs)
    stream_signal = make_stream_signal(filtered if STREAM_FILTERED_SIGNAL else raw)

    detector_events, debug = iedmod.run_detector(stream_signal, fs)
    detector_events = np.asarray(detector_events, dtype=float)

    if REFINE_EVENT_TIMES:
        if REFINE_SIGNAL_SOURCE.lower() == "raw":
            refine_signal = raw
        elif REFINE_SIGNAL_SOURCE.lower() == "filtered":
            refine_signal = filtered
        else:
            raise ValueError("REFINE_SIGNAL_SOURCE must be 'raw' or 'filtered'.")

        refined_events = refine_event_times(
            detector_events_s=detector_events,
            signal_y=refine_signal,
            fs=fs,
            method=REFINE_METHOD,
            search_before_s=REFINE_SEARCH_BEFORE_S,
            search_after_s=REFINE_SEARCH_AFTER_S,
        )
    else:
        refined_events = detector_events.copy()

    return ChannelDetectionResult(
        label=label,
        channel_index=channel_index,
        raw=raw,
        filtered=filtered,
        stream_signal=stream_signal,
        detector_events_s=detector_events,
        refined_events_s=refined_events,
        debug=debug,
    )


# ============================================================
# Event refinement
# ============================================================

def refine_event_times(
    detector_events_s: np.ndarray,
    signal_y: np.ndarray,
    fs: float,
    method: str = "max_abs_peak",
    search_before_s: float = 0.025,
    search_after_s: float = 0.025,
) -> np.ndarray:
    """
    Refine detector timestamps to a local signal landmark.

    This is useful because the detector timestamp may be the threshold-crossing
    or decision time rather than the true IED peak or max slope.

    Methods
    -------
    max_abs_peak:
        Time of largest absolute deflection in the search window.

    max_slope:
        Time of largest absolute sample-to-sample slope in the search window.

    positive_peak:
        Time of maximum positive deflection.

    negative_peak:
        Time of maximum negative deflection.
    """
    signal_y = np.asarray(signal_y)
    detector_events_s = np.asarray(detector_events_s, dtype=float)

    refined = []

    n = len(signal_y)
    before_n = int(round(search_before_s * fs))
    after_n = int(round(search_after_s * fs))

    for ev_s in detector_events_s:
        center_idx = int(round(ev_s * fs))
        start_idx = max(0, center_idx - before_n)
        end_idx = min(n, center_idx + after_n + 1)

        if end_idx <= start_idx + 2:
            refined.append(ev_s)
            continue

        seg = signal_y[start_idx:end_idx]

        if method == "max_abs_peak":
            local_idx = int(np.argmax(np.abs(seg)))

        elif method == "positive_peak":
            local_idx = int(np.argmax(seg))

        elif method == "negative_peak":
            local_idx = int(np.argmin(seg))

        elif method == "max_slope":
            # diff has length len(seg)-1. Use +1 so timestamp lands on second sample.
            dy = np.diff(seg)
            local_idx = int(np.argmax(np.abs(dy))) + 1

        else:
            raise ValueError(
                "method must be one of: "
                "'max_abs_peak', 'max_slope', 'positive_peak', 'negative_peak'"
            )

        refined_idx = start_idx + local_idx
        refined.append(refined_idx / fs)

    return np.asarray(refined, dtype=float)


# ============================================================
# Event classification / matching
# ============================================================

def classify_hpc_ieds_by_thalamic_timing(
    hpc_events_s: np.ndarray,
    thal_events_s: np.ndarray,
    min_abs_latency_s: float = 0.005,
    max_abs_latency_s: float = 0.150,
) -> pd.DataFrame:
    """
    Classify each hippocampal IED based on nearby thalamic IED timing.

    Important:
    - We do NOT collapse before and after thalamic events into one latency.
    - Hpc->Thal recruitment uses the closest following thalamic IED.
    - Preceding thalamic IEDs are stored as context.
    - Events with thalamic IEDs on both sides are flagged as complex/bidirectional-context.
    """

    hpc_events_s = np.asarray(hpc_events_s, dtype=float)
    thal_events_s = np.asarray(thal_events_s, dtype=float)

    rows = []

    for i, hpc_t in enumerate(hpc_events_s):
        dt = thal_events_s - hpc_t

        after_mask = (dt >= min_abs_latency_s) & (dt <= max_abs_latency_s)
        before_mask = (dt <= -min_abs_latency_s) & (dt >= -max_abs_latency_s)
        near_zero_mask = np.abs(dt) < min_abs_latency_s
        any_window_mask = np.abs(dt) <= max_abs_latency_s

        after_dts = dt[after_mask]
        before_dts = dt[before_mask]
        near_zero_dts = dt[near_zero_mask]
        any_dts = dt[any_window_mask]

        has_after = len(after_dts) > 0
        has_before = len(before_dts) > 0
        has_near_zero = len(near_zero_dts) > 0
        has_both_sides = has_after and has_before

        # --------------------------------------------------------
        # Store separate before/after latencies
        # --------------------------------------------------------
        nearest_after_latency_s = np.nan
        nearest_after_thal_time_s = np.nan

        if has_after:
            # Closest following thalamic IED
            nearest_after_latency_s = float(np.min(after_dts))
            nearest_after_thal_time_s = float(hpc_t + nearest_after_latency_s)

        nearest_before_latency_s = np.nan
        nearest_before_thal_time_s = np.nan

        if has_before:
            # Closest preceding thalamic IED
            nearest_before_latency_s = float(np.max(before_dts))
            nearest_before_thal_time_s = float(hpc_t + nearest_before_latency_s)

        nearest_zero_latency_s = np.nan
        nearest_zero_thal_time_s = np.nan

        if has_near_zero:
            nearest_zero_latency_s = float(
                near_zero_dts[np.argmin(np.abs(near_zero_dts))]
            )
            nearest_zero_thal_time_s = float(hpc_t + nearest_zero_latency_s)

        nearest_abs_latency_s = np.nan
        nearest_abs_thal_time_s = np.nan

        if len(any_dts) > 0:
            nearest_abs_latency_s = float(any_dts[np.argmin(np.abs(any_dts))])
            nearest_abs_thal_time_s = float(hpc_t + nearest_abs_latency_s)

        # --------------------------------------------------------
        # Primary class
        # --------------------------------------------------------
        if has_after:
            primary_event_class = "hpc_to_thal_recruited"
            selected_relation = "hpc_first"
            selected_latency_s = nearest_after_latency_s
            selected_thal_time_s = nearest_after_thal_time_s

        elif has_before:
            primary_event_class = "thal_to_hpc_preceded"
            selected_relation = "thal_first"
            selected_latency_s = nearest_before_latency_s
            selected_thal_time_s = nearest_before_thal_time_s

        elif has_near_zero:
            primary_event_class = "simultaneous_or_volume_conduction"
            selected_relation = "near_zero_lag"
            selected_latency_s = nearest_zero_latency_s
            selected_thal_time_s = nearest_zero_thal_time_s

        else:
            primary_event_class = "hpc_only"
            selected_relation = "none"
            selected_latency_s = np.nan
            selected_thal_time_s = np.nan

        # --------------------------------------------------------
        # Context class
        # --------------------------------------------------------
        if has_both_sides:
            context_class = "thal_before_and_after"
        elif has_after:
            context_class = "thal_after_only"
        elif has_before:
            context_class = "thal_before_only"
        elif has_near_zero:
            context_class = "near_zero_only"
        else:
            context_class = "no_thal_in_window"

        # Optional stricter event class for easy filtering
        if primary_event_class == "hpc_to_thal_recruited" and has_before:
            event_class = "hpc_to_thal_recruited_with_prior_thal"
        else:
            event_class = primary_event_class

        rows.append({
            "hpc_ied_index": i,
            "hpc_ied_time_s": float(hpc_t),

            "event_class": event_class,
            "primary_event_class": primary_event_class,
            "context_class": context_class,
            "selected_relation": selected_relation,

            # Main selected match
            "thal_ied_time_s": selected_thal_time_s,
            "selected_latency_ms": (
                selected_latency_s * 1000.0
                if np.isfinite(selected_latency_s)
                else np.nan
            ),
            
            "hpc_to_thal_latency_ms": (
                nearest_after_latency_s * 1000.0
                if np.isfinite(nearest_after_latency_s)
                else np.nan
            ),
            "thal_to_hpc_latency_ms": (
                nearest_before_latency_s * 1000.0
                if np.isfinite(nearest_before_latency_s)
                else np.nan
            ),

            # Hpc->Thal-specific latency
            "nearest_after_thal_time_s": nearest_after_thal_time_s,
            "nearest_after_thal_latency_ms": (
                nearest_after_latency_s * 1000.0
                if np.isfinite(nearest_after_latency_s)
                else np.nan
            ),

            # Thal->Hpc-specific latency
            "nearest_before_thal_time_s": nearest_before_thal_time_s,
            "nearest_before_thal_latency_ms": (
                nearest_before_latency_s * 1000.0
                if np.isfinite(nearest_before_latency_s)
                else np.nan
            ),

            # Near-zero-lag event
            "nearest_zero_thal_time_s": nearest_zero_thal_time_s,
            "nearest_zero_thal_latency_ms": (
                nearest_zero_latency_s * 1000.0
                if np.isfinite(nearest_zero_latency_s)
                else np.nan
            ),

            # Absolute nearest event, useful for diagnostics only
            "nearest_abs_thal_time_s": nearest_abs_thal_time_s,
            "nearest_abs_thal_latency_ms": (
                nearest_abs_latency_s * 1000.0
                if np.isfinite(nearest_abs_latency_s)
                else np.nan
            ),

            # Counts
            "n_thal_after_in_window": int(len(after_dts)),
            "n_thal_before_in_window": int(len(before_dts)),
            "n_thal_near_zero_lag": int(len(near_zero_dts)),
            "n_thal_total_in_pm_window": int(len(any_dts)),

            # Binary flags
            "has_thal_after": int(has_after),
            "has_thal_before": int(has_before),
            "has_thal_before_and_after": int(has_both_sides),
            "has_near_zero_lag": int(has_near_zero),

            "recruited_thalamus": int(has_after),
            "clean_hpc_to_thal_recruited": int(has_after and not has_before and not has_near_zero),
            "hpc_only": int(primary_event_class == "hpc_only"),
            "thal_first": int(primary_event_class == "thal_to_hpc_preceded"),
            "near_zero_lag": int(primary_event_class == "simultaneous_or_volume_conduction"),
        })

    return pd.DataFrame(rows)


def make_bidirectional_event_pairs(
    hpc_events_s: np.ndarray,
    thal_events_s: np.ndarray,
    min_abs_latency_s: float = 0.010,
    max_abs_latency_s: float = 0.500,
) -> pd.DataFrame:
    """
    Optional helper table.

    This returns every hippocampus-thalamus event pair within +/- max window,
    with near-zero events labeled separately.

    This is useful for making latency histograms.
    """
    hpc_events_s = np.asarray(hpc_events_s, dtype=float)
    thal_events_s = np.asarray(thal_events_s, dtype=float)

    rows = []

    for hpc_i, hpc_t in enumerate(hpc_events_s):
        dt = thal_events_s - hpc_t
        mask = np.abs(dt) <= max_abs_latency_s

        for thal_i, latency_s in zip(np.where(mask)[0], dt[mask]):
            if abs(latency_s) < min_abs_latency_s:
                direction = "near_zero_lag"
            elif latency_s > 0:
                direction = "hpc_first"
            else:
                direction = "thal_first"

            rows.append({
                "hpc_ied_index": int(hpc_i),
                "hpc_ied_time_s": float(hpc_t),
                "thal_ied_index": int(thal_i),
                "thal_ied_time_s": float(thal_events_s[thal_i]),
                "latency_ms": float(latency_s * 1000.0),
                "direction": direction,
            })

    return pd.DataFrame(rows)


# ============================================================
# Basic feature placeholders
# ============================================================

def add_simple_realtime_features(
    event_df: pd.DataFrame,
    hpc_signal: np.ndarray,
    fs: float,
    feature_window_s: tuple[float, float] = (-0.025, 0.075),
) -> pd.DataFrame:
    """
    Adds simple STM32-feasible hippocampal features around each hippocampal IED.

    These are intentionally simple:
        - peak absolute amplitude
        - peak-to-peak amplitude
        - RMS
        - line length
        - energy
        - max slope

    Later, you can replace or extend this using your existing feature code.
    """
    out = event_df.copy()
    y = np.asarray(hpc_signal, dtype=float)

    pre_s, post_s = feature_window_s
    n = len(y)

    feats = {
        "hpc_peak_abs": [],
        "hpc_peak_to_peak": [],
        "hpc_rms": [],
        "hpc_line_length": [],
        "hpc_energy": [],
        "hpc_max_abs_slope": [],
        "hpc_feature_n_samples": [],
    }

    for t0 in out["hpc_ied_time_s"].values:
        start_idx = max(0, int(round((t0 + pre_s) * fs)))
        end_idx = min(n, int(round((t0 + post_s) * fs)))

        if end_idx <= start_idx + 2:
            for key in feats:
                feats[key].append(np.nan)
            continue

        seg = y[start_idx:end_idx]
        dy = np.diff(seg)

        feats["hpc_peak_abs"].append(float(np.max(np.abs(seg))))
        feats["hpc_peak_to_peak"].append(float(np.max(seg) - np.min(seg)))
        feats["hpc_rms"].append(float(np.sqrt(np.mean(seg ** 2))))
        feats["hpc_line_length"].append(float(np.sum(np.abs(dy))))
        feats["hpc_energy"].append(float(np.sum(seg ** 2)))
        feats["hpc_max_abs_slope"].append(float(np.max(np.abs(dy)) * fs))
        feats["hpc_feature_n_samples"].append(int(len(seg)))

    for key, vals in feats.items():
        out[key] = vals

    return out


def add_recent_ied_features(
    event_df: pd.DataFrame,
    hpc_events_s: np.ndarray,
    rate_window_s: float = 10.0,
) -> pd.DataFrame:
    """
    Adds simple temporal context features:
        - time since previous hippocampal IED
        - number of hippocampal IEDs in the previous rate_window_s
        - recent hippocampal IED rate
    """
    out = event_df.copy()
    hpc_events_s = np.asarray(hpc_events_s, dtype=float)

    time_since_prev = []
    n_prev = []
    rate_prev = []

    for i, t0 in enumerate(out["hpc_ied_time_s"].values):
        previous = hpc_events_s[hpc_events_s < t0]

        if len(previous) == 0:
            time_since_prev.append(np.nan)
        else:
            time_since_prev.append(float(t0 - previous[-1]))

        in_window = previous[previous >= (t0 - rate_window_s)]
        n_prev.append(int(len(in_window)))
        rate_prev.append(float(len(in_window) / rate_window_s))

    out["hpc_time_since_prev_ied_s"] = time_since_prev
    out[f"hpc_n_ieds_prev_{rate_window_s:g}s"] = n_prev
    out[f"hpc_ied_rate_prev_{rate_window_s:g}s"] = rate_prev

    return out



# ============================================================
# Hippocampal IED-centered time-frequency analysis
# ============================================================

def bandpass_signal_for_tf(
    y: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase bandpass one signal segment for offline time-frequency analysis.

    This is intentionally separate from filter_signal_for_detector().
    The detector filter stays fixed at 50-85 Hz, while this function is used
    to scan many candidate frequencies around each hippocampal IED.
    """
    y = np.asarray(y, dtype=float)

    nyq = fs / 2.0
    low_hz = max(0.5, float(low_hz))
    high_hz = min(float(high_hz), nyq * 0.95)

    if low_hz >= high_hz:
        return np.full_like(y, np.nan, dtype=float)

    sos = butter(
        order,
        [low_hz, high_hz],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return sosfiltfilt(sos, y)


def extract_ied_epoch_for_tf(
    t_s: np.ndarray,
    y: np.ndarray,
    event_t_s: float,
    pre_s: float,
    post_s: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Extract a raw signal epoch around one hippocampal IED.

    Returned time is relative to the hippocampal IED time, so 0 s is the
    detected/refined hippocampal event.
    """
    t_s = np.asarray(t_s, dtype=float)
    y = np.asarray(y, dtype=float)

    start_t = event_t_s - pre_s
    end_t = event_t_s + post_s

    mask = (t_s >= start_t) & (t_s <= end_t)

    if mask.sum() < 10:
        return None, None

    epoch_t = t_s[mask] - event_t_s
    epoch_y = y[mask]

    return epoch_t, epoch_y


def compute_filterbank_hilbert_tf(
    epoch_y: np.ndarray,
    epoch_t: np.ndarray,
    fs: float,
    freqs_hz: np.ndarray,
    bandwidth_hz: float = 5.0,
    baseline_window_s: tuple[float, float] = (-0.450, -0.100),
    threshold_mode: str = "z",
) -> np.ndarray:
    """
    Compute a time-frequency matrix using bandpass filter-bank + Hilbert power.

    Parameters
    ----------
    threshold_mode:
        "z":
            Robust baseline z-score per frequency:
            (power - baseline median) / (1.4826 * baseline MAD)

        "ratio":
            Power divided by median baseline power per frequency.
            In this mode, 5.0 means 5x baseline power.

    Returns
    -------
    tf_matrix : np.ndarray
        Shape = [n_freqs, n_timepoints].
    """
    epoch_y = np.asarray(epoch_y, dtype=float)
    epoch_t = np.asarray(epoch_t, dtype=float)
    freqs_hz = np.asarray(freqs_hz, dtype=float)

    power = np.zeros((len(freqs_hz), len(epoch_y)), dtype=float)

    for i, f in enumerate(freqs_hz):
        low_hz = f - bandwidth_hz / 2.0
        high_hz = f + bandwidth_hz / 2.0

        filtered = bandpass_signal_for_tf(
            y=epoch_y,
            fs=fs,
            low_hz=low_hz,
            high_hz=high_hz,
            order=4,
        )

        analytic = hilbert(filtered)
        power[i, :] = np.abs(analytic) ** 2

    baseline_mask = (
        (epoch_t >= baseline_window_s[0]) &
        (epoch_t <= baseline_window_s[1])
    )

    min_baseline_samples = max(10, int(round(0.025 * fs)))
    if baseline_mask.sum() < min_baseline_samples:
        raise ValueError(
            "Time-frequency baseline window has too few samples. "
            "Increase TF_PRE_S or change TF_BASELINE_WINDOW_S."
        )

    baseline_power = power[:, baseline_mask]

    if threshold_mode == "z":
        baseline_med = np.nanmedian(baseline_power, axis=1, keepdims=True)
        baseline_mad = np.nanmedian(
            np.abs(baseline_power - baseline_med),
            axis=1,
            keepdims=True,
        )
        tf_matrix = (power - baseline_med) / (1.4826 * baseline_mad + 1e-12)

    elif threshold_mode == "ratio":
        baseline_med = np.nanmedian(baseline_power, axis=1, keepdims=True)
        tf_matrix = power / (baseline_med + 1e-12)

    else:
        raise ValueError("threshold_mode must be 'z' or 'ratio'.")

    return tf_matrix


def summarize_tf_threshold_crossing(
    tf_matrix: np.ndarray,
    epoch_t: np.ndarray,
    freqs_hz: np.ndarray,
    threshold: float,
    detector_band: tuple[float, float],
) -> dict:
    """
    Summarize where the time-frequency matrix crosses the selected threshold.

    This gives you a compact CSV summary alongside each plot:
        - peak time/frequency
        - frequency range above threshold
        - time range above threshold
        - fraction of threshold-crossing points inside 50-85 Hz
    """
    tf_matrix = np.asarray(tf_matrix, dtype=float)
    epoch_t = np.asarray(epoch_t, dtype=float)
    freqs_hz = np.asarray(freqs_hz, dtype=float)

    finite_mask = np.isfinite(tf_matrix)
    if not np.any(finite_mask):
        return {
            "tf_any_threshold_crossing": 0,
            "tf_peak_value": np.nan,
            "tf_peak_freq_hz": np.nan,
            "tf_peak_time_s": np.nan,
            "tf_threshold_min_freq_hz": np.nan,
            "tf_threshold_max_freq_hz": np.nan,
            "tf_threshold_start_time_s": np.nan,
            "tf_threshold_end_time_s": np.nan,
            "tf_n_threshold_points": 0,
            "tf_fraction_threshold_points_in_detector_band": np.nan,
        }

    peak_flat_idx = int(np.nanargmax(tf_matrix))
    peak_freq_idx, peak_time_idx = np.unravel_index(peak_flat_idx, tf_matrix.shape)

    threshold_mask = tf_matrix >= threshold
    any_crossing = bool(np.any(threshold_mask))

    if not any_crossing:
        return {
            "tf_any_threshold_crossing": 0,
            "tf_peak_value": float(tf_matrix[peak_freq_idx, peak_time_idx]),
            "tf_peak_freq_hz": float(freqs_hz[peak_freq_idx]),
            "tf_peak_time_s": float(epoch_t[peak_time_idx]),
            "tf_threshold_min_freq_hz": np.nan,
            "tf_threshold_max_freq_hz": np.nan,
            "tf_threshold_start_time_s": np.nan,
            "tf_threshold_end_time_s": np.nan,
            "tf_n_threshold_points": 0,
            "tf_fraction_threshold_points_in_detector_band": 0.0,
        }

    freq_indices, time_indices = np.where(threshold_mask)

    detector_freq_mask = (
        (freqs_hz[freq_indices] >= detector_band[0]) &
        (freqs_hz[freq_indices] <= detector_band[1])
    )

    return {
        "tf_any_threshold_crossing": 1,
        "tf_peak_value": float(tf_matrix[peak_freq_idx, peak_time_idx]),
        "tf_peak_freq_hz": float(freqs_hz[peak_freq_idx]),
        "tf_peak_time_s": float(epoch_t[peak_time_idx]),
        "tf_threshold_min_freq_hz": float(np.min(freqs_hz[freq_indices])),
        "tf_threshold_max_freq_hz": float(np.max(freqs_hz[freq_indices])),
        "tf_threshold_start_time_s": float(np.min(epoch_t[time_indices])),
        "tf_threshold_end_time_s": float(np.max(epoch_t[time_indices])),
        "tf_n_threshold_points": int(len(freq_indices)),
        "tf_fraction_threshold_points_in_detector_band": float(np.mean(detector_freq_mask)),
    }


def plot_hpc_ied_time_frequency(
    epoch_t: np.ndarray,
    epoch_y: np.ndarray,
    tf_matrix: np.ndarray,
    freqs_hz: np.ndarray,
    event_t_s: float,
    abf_stem: str,
    out_dir: Path,
    threshold: float = 5.0,
    threshold_mode: str = "z",
    detector_band: tuple[float, float] = (50.0, 85.0),
) -> Path:
    """
    Save one hippocampal IED-centered time-frequency plot.

    The contour marks where the baseline-normalized TF matrix crosses threshold.
    The horizontal dashed lines mark the embedded detector band.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    finite_values = tf_matrix[np.isfinite(tf_matrix)]
    if finite_values.size == 0:
        vmin, vmax = None, None
    else:
        lo, hi = TF_COLOR_PERCENTILE_CLIP
        vmin = float(np.nanpercentile(finite_values, lo))
        vmax = float(np.nanpercentile(finite_values, hi))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = None, None

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 3]},
    )

    axes[0].plot(epoch_t, epoch_y, linewidth=0.8)
    axes[0].axvline(0, linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Raw HC signal")
    axes[0].set_title(
        f"Hippocampal IED-centered TF: {abf_stem}, event {event_t_s:.3f} s"
    )

    im = axes[1].pcolormesh(
        epoch_t,
        freqs_hz,
        tf_matrix,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(im, ax=axes[1])

    if threshold_mode == "z":
        cbar.set_label("Robust power z-score vs baseline")
        threshold_label = f"power z ≥ {threshold:g}"
    else:
        cbar.set_label("Power / median baseline power")
        threshold_label = f"power ≥ {threshold:g}x baseline"

    threshold_mask = tf_matrix >= threshold

    if np.any(threshold_mask):
        axes[1].contour(
            epoch_t,
            freqs_hz,
            threshold_mask.astype(float),
            levels=[0.5],
            linewidths=1.0,
        )

    axes[1].axhline(detector_band[0], linestyle="--", linewidth=1.0)
    axes[1].axhline(detector_band[1], linestyle="--", linewidth=1.0)
    axes[1].axvline(0, linestyle="--", linewidth=1.0)

    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_xlabel("Time from hippocampal IED (s)")
    axes[1].set_title(
        f"Threshold contour: {threshold_label}; "
        f"detector band = {detector_band[0]:g}-{detector_band[1]:g} Hz"
    )

    fig.tight_layout()

    out_path = out_dir / f"{abf_stem}_hpc_ied_TF_{event_t_s:.3f}s_{threshold_mode}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return out_path


def plot_hpc_time_frequency_for_events(
    t_s: np.ndarray,
    hpc_raw: np.ndarray,
    hpc_events_s: np.ndarray,
    fs: float,
    abf_stem: str,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Run hippocampal IED-centered TF analysis for a subset of detected events.

    This should be called from process_abf_file() after hpc detection.
    """
    tf_out_dir = out_dir / "hpc_ied_time_frequency"
    tf_out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    hpc_events_s = np.asarray(hpc_events_s, dtype=float)

    events_to_plot = hpc_events_s[:TF_MAX_EVENTS_TO_PLOT]

    for event_i, event_t_s in enumerate(events_to_plot):
        epoch_t, epoch_y = extract_ied_epoch_for_tf(
            t_s=t_s,
            y=hpc_raw,
            event_t_s=float(event_t_s),
            pre_s=TF_PRE_S,
            post_s=TF_POST_S,
        )

        if epoch_t is None:
            rows.append({
                "tf_event_index": int(event_i),
                "hpc_ied_time_s": float(event_t_s),
                "tf_status": "skipped_too_close_to_edge",
            })
            continue

        try:
            tf_matrix = compute_filterbank_hilbert_tf(
                epoch_y=epoch_y,
                epoch_t=epoch_t,
                fs=fs,
                freqs_hz=TF_FREQS_HZ,
                bandwidth_hz=TF_BANDWIDTH_HZ,
                baseline_window_s=TF_BASELINE_WINDOW_S,
                threshold_mode=TF_THRESHOLD_MODE,
            )
        except ValueError as exc:
            rows.append({
                "tf_event_index": int(event_i),
                "hpc_ied_time_s": float(event_t_s),
                "tf_status": f"skipped_{exc}",
            })
            continue

        plot_path = plot_hpc_ied_time_frequency(
            epoch_t=epoch_t,
            epoch_y=epoch_y,
            tf_matrix=tf_matrix,
            freqs_hz=TF_FREQS_HZ,
            event_t_s=float(event_t_s),
            abf_stem=abf_stem,
            out_dir=tf_out_dir,
            threshold=TF_THRESHOLD,
            threshold_mode=TF_THRESHOLD_MODE,
            detector_band=(FILTER_LOW_HZ, FILTER_HIGH_HZ),
        )

        summary = summarize_tf_threshold_crossing(
            tf_matrix=tf_matrix,
            epoch_t=epoch_t,
            freqs_hz=TF_FREQS_HZ,
            threshold=TF_THRESHOLD,
            detector_band=(FILTER_LOW_HZ, FILTER_HIGH_HZ),
        )

        summary.update({
            "tf_event_index": int(event_i),
            "hpc_ied_time_s": float(event_t_s),
            "tf_status": "plotted",
            "tf_plot_path": str(plot_path),
            "tf_threshold_mode": TF_THRESHOLD_MODE,
            "tf_threshold": TF_THRESHOLD,
            "tf_baseline_start_s": TF_BASELINE_WINDOW_S[0],
            "tf_baseline_end_s": TF_BASELINE_WINDOW_S[1],
            "tf_bandwidth_hz": TF_BANDWIDTH_HZ,
            "tf_detector_low_hz": FILTER_LOW_HZ,
            "tf_detector_high_hz": FILTER_HIGH_HZ,
        })

        rows.append(summary)

    tf_summary_df = pd.DataFrame(rows)
    tf_summary_csv = tf_out_dir / f"{abf_stem}_hpc_ied_time_frequency_summary.csv"
    tf_summary_df.to_csv(tf_summary_csv, index=False)
    print("Saved:", tf_summary_csv)

    return tf_summary_df


# ============================================================
# Plotting
# ============================================================

def plot_two_channel_detection_summary(
    t_s: np.ndarray,
    hpc: ChannelDetectionResult,
    thal: ChannelDetectionResult,
    event_df: pd.DataFrame,
    abf_stem: str,
    out_dir: Path,
) -> None:
    """
    Diagnostic plot showing:
        1. Raw hippocampus + thalamus traces
        2. Filtered traces with all detected IEDs
        3. Event raster / classification summary

    This avoids duplicating the same filtered trace in subplot 2 and 3.
    """

    if PLOT_START_S is None:
        start_idx = 0
    else:
        start_idx = int(np.searchsorted(t_s, PLOT_START_S))

    if PLOT_END_S is None:
        end_idx = len(t_s)
    else:
        end_idx = int(np.searchsorted(t_s, PLOT_END_S))

    if end_idx <= start_idx:
        print("Plot window is empty. Check PLOT_START_S and PLOT_END_S.")
        return

    ts = t_s[start_idx:end_idx]
    t_min = ts[0]
    t_max = ts[-1]

    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)

    # ------------------------------------------------------------
    # Subplot 1: raw signals
    # ------------------------------------------------------------
    axes[0].plot(
        ts,
        hpc.raw[start_idx:end_idx],
        linewidth=0.8,
        label=f"{HPC_LABEL} raw",
    )
    axes[0].plot(
        ts,
        thal.raw[start_idx:end_idx],
        linewidth=0.8,
        alpha=0.7,
        label=f"{THAL_LABEL} raw",
    )
    axes[0].set_ylabel("Raw")
    axes[0].legend(loc="upper right")

    # ------------------------------------------------------------
    # Subplot 2: filtered signals + all detected events
    # ------------------------------------------------------------
    axes[1].plot(
        ts,
        hpc.filtered[start_idx:end_idx],
        linewidth=0.8,
        label=f"{HPC_LABEL} filtered",
    )
    axes[1].plot(
        ts,
        thal.filtered[start_idx:end_idx],
        linewidth=0.8,
        alpha=0.7,
        label=f"{THAL_LABEL} filtered",
    )

    for ev_t in hpc.refined_events_s:
        if t_min <= ev_t <= t_max:
            axes[1].axvline(ev_t, linestyle="--", linewidth=0.8)

    for ev_t in thal.refined_events_s:
        if t_min <= ev_t <= t_max:
            axes[1].axvline(ev_t, linestyle=":", linewidth=0.8)

    axes[1].set_ylabel("Filtered + detections")
    axes[1].legend(loc="upper right")

    # ------------------------------------------------------------
    # Subplot 3: event classification raster
    # ------------------------------------------------------------
    axes[2].set_ylabel("Event class")
    axes[2].set_xlabel("Time (s)")

    # Plot all detected hippocampal and thalamic IEDs as a raster
    hpc_events_in_window = [
        ev_t for ev_t in hpc.refined_events_s
        if t_min <= ev_t <= t_max
    ]

    thal_events_in_window = [
        ev_t for ev_t in thal.refined_events_s
        if t_min <= ev_t <= t_max
    ]

    if len(hpc_events_in_window) > 0:
        axes[2].scatter(
            hpc_events_in_window,
            np.ones(len(hpc_events_in_window)),
            marker="|",
            s=120,
            label=f"{HPC_LABEL} IEDs",
        )

    if len(thal_events_in_window) > 0:
        axes[2].scatter(
            thal_events_in_window,
            np.zeros(len(thal_events_in_window)),
            marker="|",
            s=120,
            label=f"{THAL_LABEL} IEDs",
        )

    # Draw matched/classified pairs from event_df
    for _, row in event_df.iterrows():
        hpc_t = row["hpc_ied_time_s"]

        if not (t_min <= hpc_t <= t_max):
            continue

        event_class = row["event_class"]

        # Hpc-only event: mark hippocampal event only
        if event_class == "hpc_only":
            axes[2].text(
                hpc_t,
                1.08,
                "H-only",
                rotation=90,
                fontsize=7,
                va="bottom",
                ha="center",
            )

        # Events with selected thalamic match
        if np.isfinite(row["thal_ied_time_s"]):
            thal_t = row["thal_ied_time_s"]

            if t_min <= thal_t <= t_max:
                # Connecting line from Hpc row to Thal row
                axes[2].plot(
                    [hpc_t, thal_t],
                    [1, 0],
                    linewidth=0.8,
                    alpha=0.8,
                )

                if event_class == "hpc_to_thal_recruited":
                    label = "H→T"
                elif event_class == "hpc_to_thal_recruited_with_prior_thal":
                    label = "H→T + prior T"
                elif event_class == "thal_to_hpc_preceded":
                    label = "T→H"
                elif event_class == "simultaneous_or_volume_conduction":
                    label = "0-lag"
                else:
                    label = "match"

                mid_t = (hpc_t + thal_t) / 2
                axes[2].text(
                    mid_t,
                    0.5,
                    label,
                    fontsize=7,
                    va="center",
                    ha="center",
                )

    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels([THAL_LABEL, HPC_LABEL])
    axes[2].set_ylim(-0.5, 1.5)
    axes[2].legend(loc="upper right")

    fig.suptitle(f"IED characterization: {abf_stem}")
    fig.tight_layout()

    out_path = out_dir / f"{abf_stem}_two_channel_ied_summary.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_latency_histogram(pairs_df: pd.DataFrame, abf_stem: str, out_dir: Path) -> None:
    """Plot histogram of thalamic event latencies relative to hippocampal IEDs."""
    if pairs_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pairs_df["latency_ms"].values, bins=60)
    ax.axvline(-MIN_ABS_LATENCY_S * 1000, linestyle="--", linewidth=1)
    ax.axvline(MIN_ABS_LATENCY_S * 1000, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="-", linewidth=1)

    ax.set_xlabel("Thalamic IED latency relative to hippocampal IED (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"Latency histogram: {abf_stem}")
    fig.tight_layout()

    out_path = out_dir / f"{abf_stem}_latency_histogram.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ============================================================
# Main processing
# ============================================================

def process_abf_file(abf_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process one ABF file and return event classification table + pair table."""
    print("\n" + "=" * 80)
    print("Processing:", abf_path)

    out_dir = OUTPUT_DIR / abf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    abf = pyabf.ABF(str(abf_path))

    print("adcNames:", getattr(abf, "adcNames", None))
    print("adcUnits:", getattr(abf, "adcUnits", None))
    print("dataRate:", abf.dataRate)
    print("sweepCount:", abf.sweepCount)

    t_hpc, y_hpc, fs_hpc = load_abf_channel(abf, SWEEP_INDEX, HPC_CHANNEL)
    t_thal, y_thal, fs_thal = load_abf_channel(abf, SWEEP_INDEX, THAL_CHANNEL)

    if abs(fs_hpc - fs_thal) > 1e-9:
        raise ValueError(f"Sampling rates differ: hpc={fs_hpc}, thal={fs_thal}")

    if len(y_hpc) != len(y_thal):
        raise ValueError(f"Channel lengths differ: hpc={len(y_hpc)}, thal={len(y_thal)}")

    fs = fs_hpc
    t_s = t_hpc

    hpc = detect_ieds_on_channel(y_hpc, fs, HPC_CHANNEL, HPC_LABEL)
    thal = detect_ieds_on_channel(y_thal, fs, THAL_CHANNEL, THAL_LABEL)

    print(f"{HPC_LABEL} detected events:", len(hpc.refined_events_s))
    print(f"{THAL_LABEL} detected events:", len(thal.refined_events_s))
    print(f"{HPC_LABEL} detector counts:", hpc.debug["counts"])
    print(f"{THAL_LABEL} detector counts:", thal.debug["counts"])

    event_df = classify_hpc_ieds_by_thalamic_timing(
        hpc_events_s=hpc.refined_events_s,
        thal_events_s=thal.refined_events_s,
        min_abs_latency_s=MIN_ABS_LATENCY_S,
        max_abs_latency_s=MAX_ABS_LATENCY_S,
    )

    pair_df = make_bidirectional_event_pairs(
        hpc_events_s=hpc.refined_events_s,
        thal_events_s=thal.refined_events_s,
        min_abs_latency_s=MIN_ABS_LATENCY_S,
        max_abs_latency_s=MAX_ABS_LATENCY_S,
    )

    # Add basic features now. Replace/extend later with your richer feature functions.
    event_df = add_simple_realtime_features(
        event_df=event_df,
        hpc_signal=hpc.filtered,
        fs=fs,
        feature_window_s=(-0.025, 0.075),
    )

    event_df = add_recent_ied_features(
        event_df=event_df,
        hpc_events_s=hpc.refined_events_s,
        rate_window_s=10.0,
    )

    # Add metadata columns
    event_df.insert(0, "file", abf_path.name)
    event_df.insert(1, "sweep_index", SWEEP_INDEX)
    event_df.insert(2, "hpc_channel", HPC_CHANNEL)
    event_df.insert(3, "thal_channel", THAL_CHANNEL)
    event_df.insert(4, "fs_hz", fs)
    event_df.insert(5, "min_abs_latency_s", MIN_ABS_LATENCY_S)
    event_df.insert(6, "max_abs_latency_s", MAX_ABS_LATENCY_S)

    if not pair_df.empty:
        pair_df.insert(0, "file", abf_path.name)
        pair_df.insert(1, "sweep_index", SWEEP_INDEX)
        pair_df.insert(2, "fs_hz", fs)

    event_csv = out_dir / f"{abf_path.stem}_hpc_ied_characterization.csv"
    pair_csv = out_dir / f"{abf_path.stem}_hpc_thal_event_pairs.csv"

    event_df.to_csv(event_csv, index=False)
    pair_df.to_csv(pair_csv, index=False)

    print("Saved:", event_csv)
    print("Saved:", pair_csv)

    if len(event_df) > 0:
        print("\nEvent class counts:")
        print(event_df["event_class"].value_counts())

    if MAKE_TIME_FREQUENCY_PLOTS:
        # Use hpc.raw here on purpose. Using hpc.filtered would bias the TF plot
        # toward the detector's imposed 50-85 Hz band.
        plot_hpc_time_frequency_for_events(
            t_s=t_s,
            hpc_raw=hpc.raw,
            hpc_events_s=hpc.refined_events_s,
            fs=fs,
            abf_stem=abf_path.stem,
            out_dir=out_dir,
        )

    if MAKE_PLOTS:
        plot_two_channel_detection_summary(t_s, hpc, thal, event_df, abf_path.stem, out_dir)
        plot_latency_histogram(pair_df, abf_path.stem, out_dir)

    return event_df, pair_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    abf_files = find_abf_files()
    print("Number of ABF files:", len(abf_files))

    all_events = []
    all_pairs = []

    for abf_path in abf_files:
        event_df, pair_df = process_abf_file(abf_path)
        all_events.append(event_df)
        all_pairs.append(pair_df)

    if all_events:
        all_events_df = pd.concat(all_events, ignore_index=True)
        out_csv = OUTPUT_DIR / "ALL_hpc_ied_characterization.csv"
        all_events_df.to_csv(out_csv, index=False)
        print("\nSaved combined event table:", out_csv)

    if all_pairs:
        all_pairs_df = pd.concat(all_pairs, ignore_index=True)
        out_csv = OUTPUT_DIR / "ALL_hpc_thal_event_pairs.csv"
        all_pairs_df.to_csv(out_csv, index=False)
        print("Saved combined pair table:", out_csv)


if __name__ == "__main__":
    main()
