# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:35:40 2026

@author: vijit
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pyabf
from scipy.signal import butter, sosfiltfilt, sosfilt


# ============================================================
# User settings
# ============================================================

ABF_DIR = Path.cwd().parent / "Data"

# Pick first ABF in folder. You can also set this manually:
ABF_PATH = None
# Example:
# ABF_PATH = r"C:\Users\vijit\Documents\seizureDetection\IEDData\your_file.abf"

SWEEP_INDEX = 0

# Your current intended STM32 sample rate
# This should match the ABF sampling rate, or the signal should be resampled.
SAMP_FREQ = 5000

# Python preprocessing filter, matching your current UART debug script
FILTER_LOW_HZ = 50.0
FILTER_HIGH_HZ = 85.0
FILTER_ORDER = 4

# If True, use zero-phase offline filtering for debug visualization.
# If False, use causal filtering, closer to embedded behavior.
USE_ZERO_PHASE_FILTER = True

# This should match your current STM32 debug condition:
#   float bp = x;
# meaning Python sends already filtered data and STM32 bypasses its own filter.
STREAM_FILTERED_SIGNAL = True

# Simulate the same scaling/quantization as your UART transfer:
# Python:
#   sample_i = int(round(sample_float * GAIN))
# STM32:
#   int32_t sample = reconstructed signed 24-bit
SIMULATE_UART_24BIT_SCALING = True
GAIN = 1000

# IED detector parameters, matching your current STM32 settings
ENV_K = 5.0
AMP_MIN_K = 10.0
AMP_ARTIFACT_K = 60.0
BASELINE_ALPHA = 0.00002
MIN_STD = 1.0
WARMUP_SECONDS = 0.5
REFRACTORY_SECONDS = 0.35
ENVELOPE_WINDOW_SAMPLES = 25  # 25 samples at 5 kHz = 5 ms

# For first debugging, I recommend disabling amplitude rejection
# to confirm envelope detections happen.
ENABLE_LOW_AMPLITUDE_REJECTION = True
ENABLE_ARTIFACT_REJECTION = True

# Plot range. Set to None to plot full file.
PLOT_START_S = None
PLOT_END_S = None


# ============================================================
# Detector implementation
# ============================================================

@dataclass
class IEDParams:
    fs_hz: float
    env_k: float
    amp_min_k: float
    amp_artifact_k: float
    baseline_alpha: float
    min_std: float
    warmup_samples: int
    refractory_samples: int
    envelope_window_samples: int


class IEDDetector:
    """
    Python equivalent of your STM32 IED detector.

    This version assumes the input x is already bandpassed if STREAM_FILTERED_SIGNAL=True.
    It mirrors:
        float bp = x;
        rectified = fabsf(bp);
        env = moving_average_envelope(rectified);
        amp = fabsf(x) or rectified;
        baseline mean/var updates
        threshold checks
    """

    IED_NO_EVENT = 0
    IED_DETECTED = 1
    IED_REJECTED_LOW_AMPLITUDE = 2
    IED_REJECTED_ARTIFACT = 3
    IED_IN_REFRACTORY = 4

    def __init__(self, params: IEDParams):
        self.p = params

        self.sample_count = 0
        self.refractory_until_sample = 0

        self.env_mean = 0.0
        self.env_var = 0.0

        self.amp_mean = 0.0
        self.amp_var = 0.0

        self.env_buf = np.zeros(self.p.envelope_window_samples, dtype=float)
        self.env_sum = 0.0
        self.env_index = 0
        self.env_count = 0

        # Debug values
        self.last_bandpass = 0.0
        self.last_envelope = 0.0
        self.last_amp = 0.0
        self.last_env_thresh = 0.0
        self.last_amp_min_thresh = 0.0
        self.last_amp_artifact_thresh = 0.0

    @staticmethod
    def _update_baseline(x, alpha, mean, var):
        delta = x - mean
        mean = mean + alpha * delta
        var = (1.0 - alpha) * (var + alpha * delta * delta)
        return mean, var

    def _moving_average_envelope(self, rectified):
        win = self.p.envelope_window_samples

        if self.env_count < win:
            self.env_buf[self.env_index] = rectified
            self.env_sum += rectified
            self.env_count += 1
        else:
            self.env_sum -= self.env_buf[self.env_index]
            self.env_buf[self.env_index] = rectified
            self.env_sum += rectified

        self.env_index += 1
        if self.env_index >= win:
            self.env_index = 0

        if self.env_count == 0:
            return 0.0

        return self.env_sum / self.env_count

    def process_sample(self, raw_sample):
        self.sample_count += 1
        sample_idx = self.sample_count - 1

        x = float(raw_sample)

        # Current STM32 debug mode:
        #   float bp = x;
        bp = x
        self.last_bandpass = bp

        rectified = abs(bp)

        env = self._moving_average_envelope(rectified)
        self.last_envelope = env

        # Current debug simplification:
        #   float hp = x;
        #   amp = fabsf(hp)
        # Since x is already filtered, amp = abs(x) is acceptable.
        amp = abs(x)
        self.last_amp = amp

        env_std = np.sqrt(max(self.env_var, self.p.min_std * self.p.min_std))
        amp_std = np.sqrt(max(self.amp_var, self.p.min_std * self.p.min_std))

        env_thresh = self.env_mean + self.p.env_k * env_std
        amp_min_thresh = self.amp_mean + self.p.amp_min_k * amp_std
        amp_artifact_thresh = self.amp_mean + self.p.amp_artifact_k * amp_std

        self.last_env_thresh = env_thresh
        self.last_amp_min_thresh = amp_min_thresh
        self.last_amp_artifact_thresh = amp_artifact_thresh

        # Warmup baseline phase
        if self.sample_count < self.p.warmup_samples:
            self.env_mean, self.env_var = self._update_baseline(
                env, self.p.baseline_alpha, self.env_mean, self.env_var
            )
            self.amp_mean, self.amp_var = self._update_baseline(
                amp, self.p.baseline_alpha, self.amp_mean, self.amp_var
            )
            return self.IED_NO_EVENT

        # Refractory gate
        if sample_idx < self.refractory_until_sample:
            return self.IED_IN_REFRACTORY

        # Candidate from envelope
        envelope_crossed = env >= env_thresh

        if not envelope_crossed:
            self.env_mean, self.env_var = self._update_baseline(
                env, self.p.baseline_alpha, self.env_mean, self.env_var
            )
            self.amp_mean, self.amp_var = self._update_baseline(
                amp, self.p.baseline_alpha, self.amp_mean, self.amp_var
            )
            return self.IED_NO_EVENT

        # Low-amplitude rejection
        if ENABLE_LOW_AMPLITUDE_REJECTION and amp < amp_min_thresh:
            return self.IED_REJECTED_LOW_AMPLITUDE

        # Artifact rejection
        if ENABLE_ARTIFACT_REJECTION and amp > amp_artifact_thresh:
            return self.IED_REJECTED_ARTIFACT

        # Valid IED
        self.refractory_until_sample = sample_idx + self.p.refractory_samples
        return self.IED_DETECTED


# ============================================================
# Utility functions
# ============================================================

def find_abf_file():
    if ABF_PATH is not None:
        return Path(ABF_PATH)

    abf_files = []
    for dirpath, _, filenames in os.walk(ABF_DIR):
        for filename in filenames:
            if filename.lower().endswith(".abf"):
                abf_files.append(Path(dirpath) / filename)

    if not abf_files:
        raise FileNotFoundError(f"No ABF files found in {ABF_DIR}")

    return abf_files[0]


def simulate_signed_24bit_transfer(signal_float, gain):
    """
    Simulates:
        Python float -> signed 24-bit integer -> STM32 int32_t reconstruction

    This helps check if scaling/quantization changes detection.
    """
    sample_i = np.round(signal_float * gain).astype(np.int64)

    sample_i = np.clip(sample_i, -8388608, 8388607)

    # This final int32 signal is what STM32 should reconstruct.
    return sample_i.astype(np.int32)


def run_detector(signal_in, fs):
    params = IEDParams(
        fs_hz=fs,
        env_k=ENV_K,
        amp_min_k=AMP_MIN_K,
        amp_artifact_k=AMP_ARTIFACT_K,
        baseline_alpha=BASELINE_ALPHA,
        min_std=MIN_STD,
        warmup_samples=int(WARMUP_SECONDS * fs),
        refractory_samples=int(REFRACTORY_SECONDS * fs),
        envelope_window_samples=ENVELOPE_WINDOW_SAMPLES,
    )

    det = IEDDetector(params)

    events = []
    env_trace = np.zeros(len(signal_in), dtype=float)
    env_thresh_trace = np.zeros(len(signal_in), dtype=float)
    amp_trace = np.zeros(len(signal_in), dtype=float)
    amp_min_thresh_trace = np.zeros(len(signal_in), dtype=float)
    amp_artifact_thresh_trace = np.zeros(len(signal_in), dtype=float)
    event_code_trace = np.zeros(len(signal_in), dtype=np.int16)

    counts = {
        "detected": 0,
        "low_amp": 0,
        "artifact": 0,
        "refractory": 0,
        "no_event": 0,
    }

    for i, sample in enumerate(signal_in):
        ev = det.process_sample(sample)

        env_trace[i] = det.last_envelope
        env_thresh_trace[i] = det.last_env_thresh
        amp_trace[i] = det.last_amp
        amp_min_thresh_trace[i] = det.last_amp_min_thresh
        amp_artifact_thresh_trace[i] = det.last_amp_artifact_thresh
        event_code_trace[i] = ev

        t = i / fs

        if ev == det.IED_DETECTED:
            counts["detected"] += 1
            events.append(t)
        elif ev == det.IED_REJECTED_LOW_AMPLITUDE:
            counts["low_amp"] += 1
        elif ev == det.IED_REJECTED_ARTIFACT:
            counts["artifact"] += 1
        elif ev == det.IED_IN_REFRACTORY:
            counts["refractory"] += 1
        else:
            counts["no_event"] += 1

    debug = {
        "env": env_trace,
        "env_thresh": env_thresh_trace,
        "amp": amp_trace,
        "amp_min_thresh": amp_min_thresh_trace,
        "amp_artifact_thresh": amp_artifact_thresh_trace,
        "event_code": event_code_trace,
        "counts": counts,
    }

    return events, debug


def plot_results(t, raw, filtered, stream_signal, events, debug, file_name):
    if PLOT_START_S is not None:
        start_idx = np.searchsorted(t, PLOT_START_S)
    else:
        start_idx = 0

    if PLOT_END_S is not None:
        end_idx = np.searchsorted(t, PLOT_END_S)
    else:
        end_idx = len(t)

    ts = t[start_idx:end_idx]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(ts, raw[start_idx:end_idx], linewidth=0.8, label="Raw ABF")
    axes[0].plot(ts, filtered[start_idx:end_idx], linewidth=0.8, label="Python 50-100 Hz filtered")
    axes[0].set_ylabel("Signal")
    axes[0].legend()

    axes[1].plot(ts, stream_signal[start_idx:end_idx], linewidth=0.8, label="Signal sent / detector input")
    axes[1].set_ylabel("Detector input")
    axes[1].legend()

    axes[2].plot(ts, debug["env"][start_idx:end_idx], linewidth=0.8, label="Envelope")
    axes[2].plot(ts, debug["env_thresh"][start_idx:end_idx], linewidth=0.8, label="Envelope threshold")
    axes[2].set_ylabel("Envelope")
    axes[2].legend()

    axes[3].plot(ts, debug["amp"][start_idx:end_idx], linewidth=0.8, label="Amplitude")
    axes[3].plot(ts, debug["amp_min_thresh"][start_idx:end_idx], linewidth=0.8, label="Amp min threshold")
    axes[3].plot(ts, debug["amp_artifact_thresh"][start_idx:end_idx], linewidth=0.8, label="Artifact threshold")
    axes[3].set_ylabel("Amplitude")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend()

    for ax in axes:
        for ev_t in events:
            if ts[0] <= ev_t <= ts[-1]:
                ax.axvline(ev_t, linestyle="--", linewidth=1.0)

    fig.suptitle(f"Offline STM32-like IED detector debug: {file_name}")
    fig.tight_layout()

    out_name = f"offline_ied_debug_{file_name}.png"
    fig.savefig(out_name, dpi=200)
    print(f"Saved figure: {out_name}")

    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    abf_path = find_abf_file()
    print("Using file:", abf_path)

    abf = pyabf.ABF(str(abf_path))
    abf.setSweep(SWEEP_INDEX)

    data_y = abf.sweepY.copy()
    data_x = abf.sweepX.copy()
    fs = float(abf.dataRate)

    file_name = abf_path.stem

    print("adcNames:", getattr(abf, "adcNames", None))
    print("adcUnits:", getattr(abf, "adcUnits", None))
    print("ABF sampling rate:", fs)
    print("Total samples:", len(data_y))
    print("Duration:", len(data_y) / fs, "s")

    if abs(fs - SAMP_FREQ) > 1e-6:
        print("\nWARNING:")
        print(f"  ABF fs = {fs}, but SAMP_FREQ = {SAMP_FREQ}.")
        print("  Your STM32 timing will not match unless you resample or change SAMP_FREQ.\n")

    sos = butter(
        FILTER_ORDER,
        [FILTER_LOW_HZ, FILTER_HIGH_HZ],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    if USE_ZERO_PHASE_FILTER:
        filtered_y = sosfiltfilt(sos, data_y)
    else:
        filtered_y = sosfilt(sos, data_y)

    if STREAM_FILTERED_SIGNAL:
        stream_signal_float = filtered_y
        print("Detector input: Python-filtered signal")
    else:
        stream_signal_float = data_y
        print("Detector input: raw ABF signal")

    print("\nSignal ranges before UART simulation:")
    print("Raw min/max:", np.min(data_y), np.max(data_y))
    print("Filtered min/max:", np.min(filtered_y), np.max(filtered_y))
    print("Stream float min/max:", np.min(stream_signal_float), np.max(stream_signal_float))

    if SIMULATE_UART_24BIT_SCALING:
        stream_signal = simulate_signed_24bit_transfer(stream_signal_float, GAIN)
        print("\nUART 24-bit scaling simulation enabled")
        print("GAIN:", GAIN)
        print("Stream int min/max:", np.min(stream_signal), np.max(stream_signal))
    else:
        stream_signal = stream_signal_float.astype(float)
        print("\nUART 24-bit scaling simulation disabled")

    events, debug = run_detector(stream_signal, fs)

    print("\nDetector parameters:")
    print("ENV_K:", ENV_K)
    print("AMP_MIN_K:", AMP_MIN_K)
    print("AMP_ARTIFACT_K:", AMP_ARTIFACT_K)
    print("BASELINE_ALPHA:", BASELINE_ALPHA)
    print("MIN_STD:", MIN_STD)
    print("WARMUP_SECONDS:", WARMUP_SECONDS)
    print("REFRACTORY_SECONDS:", REFRACTORY_SECONDS)
    print("ENVELOPE_WINDOW_SAMPLES:", ENVELOPE_WINDOW_SAMPLES)
    print("ENABLE_LOW_AMPLITUDE_REJECTION:", ENABLE_LOW_AMPLITUDE_REJECTION)
    print("ENABLE_ARTIFACT_REJECTION:", ENABLE_ARTIFACT_REJECTION)

    print("\nEvent counts:")
    for k, v in debug["counts"].items():
        print(f"{k}: {v}")

    print("\nIED detected times (s):")
    print(events)

    t = np.arange(len(data_y)) / fs
    plot_results(t, data_y, filtered_y, stream_signal, events, debug, file_name)


if __name__ == "__main__":
    main()