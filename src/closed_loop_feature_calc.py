"""Closed Loop Feature Calculations: Calculate features that can be used for distinguishing
closed loop """

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt

WINDOW_PRE = 0.4
WINDOW_POST = 0.2

REGION_EXCLUDED = -1
REGION_BASELINE = 0
REGION_IED = 1
REGION_REJECTED = 2

REGION_NAMES = {
    REGION_EXCLUDED: "excluded",
    REGION_BASELINE: "baseline",
    REGION_IED: "ied",
    REGION_REJECTED: "rejected",
}

@dataclass(frozen=True)
class ReplayConfig:
    filter_low_hz: float = 50.0
    filter_high_hz: float = 85.0
    filter_order: int = 4

    simulate_uart_scaling: bool = True
    uart_gain: float = 1000.0

    # Which signal should be used for the amplitude checks?
    # Options: "raw" or "bandpassed"
    amplitude_source: str = "raw"

    baseline_guard_s: float = 0.5
    baseline_hop_ms: float = 50.0
    random_seed: int = 42

@dataclass
class FeatureParams:
    fs_hz: float
    baseline_alpha: float = 0.00002
    min_std: float = 1.0
    warmup_samples: int = 2500
    envelope_window_samples: int = 25
    feature_window_samples: int = 125

class ClosedLoopFeatureAnalysis:
    """Calculate closed loop features on timestamps"""

    def __init__(self, timestamps, recording, rejected_timestamps=None, replay_config=None, feature_params=None, raw=None):
        self.timestamps = np.sort(np.asarray(timestamps, dtype=float))
        
        if rejected_timestamps is None:
            rejected_timestamps = []

        self.rejected_timestamps = np.sort(
            np.asarray(rejected_timestamps, dtype=float)
        )

        self.recording = recording

        if raw is None:
            raw = recording.raw

        self.raw = np.asarray(raw, dtype=float)
        self.fs = float(recording.fs)
        self.t = np.arange(len(self.raw), dtype=float) / self.fs

        self.config = replay_config or ReplayConfig()
        self.feature_params = feature_params or FeatureParams(fs_hz=self.fs)

        self.bandpassed = None
        self.detector_input = None
        self.amplitude_input = None
        self.feature_traces = None

        self.detected_times = None
        self.debug = None

        ## All the Features Calculated below ##
        self.feature_names = (
            "envelope",
            "env_mean",
            "env_std",
            "env_z",
            "amplitude",
            "amp_mean",
            "amp_std",
            "amp_z",
            "line_length",
            "line_length_mean",
            "line_length_std",
            "line_length_z",
            "energy",
            "energy_mean",
            "energy_std",
            "energy_z",
            "abs_slope",
        )

        self.region_code = None
        self.region_name = None
        self.event_id = None
        self.relative_time_s = None

    def _causal_bandpass(self, signal):
        """
        Causal filter. This is appropriate for STM32-transfer analysis.

        scipy.signal.sosfilt begins with zero filter state by default,
        similar to an embedded filter initialized to zero.
        """
        sos = butter(
            self.config.filter_order,
            [
                self.config.filter_low_hz,
                self.config.filter_high_hz,
            ],
            btype="bandpass",
            fs=self.fs,
            output="sos",
        )

        return sosfilt(sos, signal)

    @staticmethod
    def _simulate_signed_24bit_transfer(signal, gain):
        """
        Simulate:
            floating-point value
            -> scaled signed 24-bit integer
            -> reconstructed STM32 int32 value
        """
        scaled = np.round(signal * gain).astype(np.int64)
        scaled = np.clip(scaled, -8_388_608, 8_388_607)
        return scaled.astype(np.int32)
    
    def prepare_inputs(self):
        """Prepare causal bandpassed and amplitude signals."""

        self.bandpassed = self._causal_bandpass(self.raw)

        if self.config.simulate_uart_scaling:
            self.detector_input = self._simulate_signed_24bit_transfer(
                self.bandpassed,
                self.config.uart_gain,
            )

            if self.config.amplitude_source == "raw":
                self.amplitude_input = self._simulate_signed_24bit_transfer(
                    self.raw,
                    self.config.uart_gain,
                )
            elif self.config.amplitude_source == "bandpassed":
                self.amplitude_input = self.detector_input.copy()
            else:
                raise ValueError(
                    "amplitude_source must be 'raw' or 'bandpassed'."
                )

        else:
            self.detector_input = self.bandpassed.astype(float)

            if self.config.amplitude_source == "raw":
                self.amplitude_input = self.raw.astype(float)
            elif self.config.amplitude_source == "bandpassed":
                self.amplitude_input = self.bandpassed.astype(float)
            else:
                raise ValueError(
                    "amplitude_source must be 'raw' or 'bandpassed'."
                )
            
    def annotate_samples(self, ied_pre_s=0.100, ied_post_s=0.050, rejected_pre_s=0.100, rejected_post_s=0.050):
        """
        Label every sample as baseline, validated IED, rejected event,
        or excluded.

        Event timestamps are treated as reference landmarks, such as the
        manually identified IED trough.
        """

        n_samples = len(self.raw)

        region_code = np.full(
            n_samples,
            REGION_BASELINE,
            dtype=np.int8,
        )

        event_id = np.full(
            n_samples,
            -1,
            dtype=np.int32,
        )

        relative_time_s = np.full(
            n_samples,
            np.nan,
            dtype=np.float32,
        )

        # ---------------------------------------------------------
        # Exclude baseline initialization/warm-up
        # ---------------------------------------------------------
        warmup_end = min(
            self.feature_params.warmup_samples,
            n_samples,
        )

        region_code[:warmup_end] = REGION_EXCLUDED

        # ---------------------------------------------------------
        # Exclude guard regions surrounding every event
        # ---------------------------------------------------------
        guard_samples = int(
            round(self.config.baseline_guard_s * self.fs)
        )

        all_event_times = np.concatenate(
            [
                self.timestamps,
                self.rejected_timestamps,
            ]
        )

        for event_time_s in all_event_times:
            center_idx = int(round(event_time_s * self.fs))

            start_idx = max(
                0,
                center_idx - guard_samples,
            )
            end_idx = min(
                n_samples,
                center_idx + guard_samples + 1,
            )

            region_code[start_idx:end_idx] = REGION_EXCLUDED

        # ---------------------------------------------------------
        # Mark rejected-event windows
        # ---------------------------------------------------------
        rejected_pre_n = int(round(rejected_pre_s * self.fs))
        rejected_post_n = int(round(rejected_post_s * self.fs))

        for rejected_id, event_time_s in enumerate(
            self.rejected_timestamps
        ):
            center_idx = int(round(event_time_s * self.fs))

            start_idx = max(
                0,
                center_idx - rejected_pre_n,
            )
            end_idx = min(
                n_samples,
                center_idx + rejected_post_n + 1,
            )

            indices = np.arange(start_idx, end_idx)

            region_code[indices] = REGION_REJECTED
            event_id[indices] = rejected_id
            relative_time_s[indices] = (
                indices - center_idx
            ) / self.fs

        # ---------------------------------------------------------
        # Mark validated IED windows
        #
        # Do this after rejected events so validated IEDs take
        # priority if windows accidentally overlap.
        # ---------------------------------------------------------
        ied_pre_n = int(round(ied_pre_s * self.fs))
        ied_post_n = int(round(ied_post_s * self.fs))

        for ied_id, event_time_s in enumerate(self.timestamps):
            center_idx = int(round(event_time_s * self.fs))

            start_idx = max(
                0,
                center_idx - ied_pre_n,
            )
            end_idx = min(
                n_samples,
                center_idx + ied_post_n + 1,
            )

            indices = np.arange(start_idx, end_idx)

            region_code[indices] = REGION_IED
            event_id[indices] = ied_id
            relative_time_s[indices] = (
                indices - center_idx
            ) / self.fs

        self.region_code = region_code
        self.event_id = event_id
        self.relative_time_s = relative_time_s

        return region_code


    def calculate_feature_traces(self):
        """
        Calculate causal STM32-compatible features across the entire trace.
        """
        self.prepare_inputs()

        p = self.feature_params

        extractor_params = FeatureParams(
            fs_hz=self.fs,
            baseline_alpha=p.baseline_alpha,
            min_std=p.min_std,
            warmup_samples=p.warmup_samples,
            envelope_window_samples=p.envelope_window_samples,
            feature_window_samples=p.feature_window_samples,
        )

        extractor = RealTimeFeatureExtractor(extractor_params)

        self.feature_traces = {
            name: np.full(len(self.raw), np.nan, dtype=float)
            for name in self.feature_names
        }

        for i, (bandpassed_sample, amplitude_sample) in enumerate(
            zip(self.detector_input, self.amplitude_input)
        ):
            features = extractor.process_sample(
                bandpassed_sample=bandpassed_sample,
                amplitude_sample=amplitude_sample,
            )

            for name in self.feature_names:
                self.feature_traces[name][i] = features[name]

        return self.feature_traces

    def build_sample_dataframe(self, start_idx=0, end_idx=None):
        """
        Build a sample-level DataFrame for part or all of the recording.

        Using start_idx/end_idx allows a large recording to be written in
        manageable chunks.
        """
        if self.feature_traces is None:
            raise RuntimeError(
                "Call calculate_feature_traces() first."
            )

        if not hasattr(self, "region_code"):
            self.annotate_samples()

        if end_idx is None:
            end_idx = len(self.raw)

        start_idx = max(0, int(start_idx))
        end_idx = min(len(self.raw), int(end_idx))

        if end_idx <= start_idx:
            raise ValueError(
                "end_idx must be greater than start_idx."
            )

        sample_indices = np.arange(
            start_idx,
            end_idx,
            dtype=np.int64,
        )

        data = {
            "sample_index": sample_indices,
            "time_s": self.t[start_idx:end_idx],
            "raw": self.raw[start_idx:end_idx],
            "bandpassed": self.bandpassed[start_idx:end_idx],
            "detector_input": self.detector_input[start_idx:end_idx],
            "amplitude_input": self.amplitude_input[start_idx:end_idx],
            "region_code": self.region_code[start_idx:end_idx],
            "event_id": self.event_id[start_idx:end_idx],
            "relative_time_s": self.relative_time_s[start_idx:end_idx],
        }

        for feature_name, feature_trace in self.feature_traces.items():
            data[feature_name] = feature_trace[start_idx:end_idx]

        return pd.DataFrame(data)

    def extract_snippet(self, event_time_s, pre_s = WINDOW_PRE, post_s = WINDOW_POST):
        fs = self.recording.fs
        raw = self.recording.raw

        center_idx = int(round(event_time_s * fs))

        pre_n = int(round(fs * pre_s))
        post_n = int(round(fs * post_s))

        start_event_idx = center_idx - pre_n
        end_event_idx = center_idx + post_n

        if start_event_idx < 0 or end_event_idx > len(raw):
            return None

        event_samples = raw[start_event_idx:end_event_idx]
        return event_samples

  
class RealTimeFeatureExtractor:
    def __init__(self, params: FeatureParams):
        self.p = params
        self.sample_count = 0

        # Envelope rolling state
        self.env_buffer = np.zeros(
            self.p.envelope_window_samples,
            dtype=float,
        )
        self.env_sum = 0.0
        self.env_index = 0
        self.env_count = 0

        # General rolling feature state
        self.feature_buffer_length = self.p.feature_window_samples
        self.feature_index = 0
        self.feature_count = 0

        self.line_length_buffer = np.zeros(
            self.feature_buffer_length,
            dtype=float,
        )
        self.energy_buffer = np.zeros(
            self.feature_buffer_length,
            dtype=float,
        )

        self.line_length_sum = 0.0
        self.energy_sum = 0.0

        self.previous_bp = 0.0
        self.has_previous_bp = False

        # Adaptive feature baselines
        self.env_mean = 0.0
        self.env_var = 0.0

        self.amp_mean = 0.0
        self.amp_var = 0.0

        self.line_length_mean = 0.0
        self.line_length_var = 0.0

        self.energy_mean = 0.0
        self.energy_var = 0.0

    @staticmethod
    def _update_baseline(value, alpha, mean, variance):
        delta = value - mean
        mean = mean + alpha * delta
        variance = (1.0 - alpha) * (
            variance + alpha * delta * delta
        )
        return mean, variance

    def _update_envelope(self, rectified):
        if self.env_count < self.p.envelope_window_samples:
            self.env_count += 1
        else:
            self.env_sum -= self.env_buffer[self.env_index]

        self.env_buffer[self.env_index] = rectified
        self.env_sum += rectified

        self.env_index = (
            self.env_index + 1
        ) % self.p.envelope_window_samples

        return self.env_sum / max(self.env_count, 1)

    def _update_rolling_features(self, bp):
        if self.has_previous_bp:
            abs_difference = abs(bp - self.previous_bp)
        else:
            abs_difference = 0.0
            self.has_previous_bp = True

        self.previous_bp = bp
        squared_sample = bp * bp

        if self.feature_count < self.feature_buffer_length:
            self.feature_count += 1
        else:
            self.line_length_sum -= (
                self.line_length_buffer[self.feature_index]
            )
            self.energy_sum -= self.energy_buffer[self.feature_index]

        self.line_length_buffer[self.feature_index] = abs_difference
        self.energy_buffer[self.feature_index] = squared_sample

        self.line_length_sum += abs_difference
        self.energy_sum += squared_sample

        self.feature_index = (
            self.feature_index + 1
        ) % self.feature_buffer_length

        line_length = self.line_length_sum
        energy = self.energy_sum / max(self.feature_count, 1)

        return line_length, energy, abs_difference

    def process_sample(self, bandpassed_sample, amplitude_sample):
        self.sample_count += 1

        bp = float(bandpassed_sample)
        amp = abs(float(amplitude_sample))

        rectified = abs(bp)
        envelope = self._update_envelope(rectified)

        line_length, energy, abs_slope = (
            self._update_rolling_features(bp)
        )

        env_std = np.sqrt(
            max(self.env_var, self.p.min_std**2)
        )
        amp_std = np.sqrt(
            max(self.amp_var, self.p.min_std**2)
        )
        line_length_std = np.sqrt(
            max(self.line_length_var, self.p.min_std**2)
        )
        energy_std = np.sqrt(
            max(self.energy_var, self.p.min_std**2)
        )

        features = {
            "envelope": envelope,
            "env_mean": self.env_mean,
            "env_std": env_std,
            "env_z": (envelope - self.env_mean) / env_std,

            "amplitude": amp,
            "amp_mean": self.amp_mean,
            "amp_std": amp_std,
            "amp_z": (amp - self.amp_mean) / amp_std,

            "line_length": line_length,
            "line_length_mean": self.line_length_mean,
            "line_length_std": line_length_std,
            "line_length_z": (
                line_length - self.line_length_mean
            ) / line_length_std,

            "energy": energy,
            "energy_mean": self.energy_mean,
            "energy_std": energy_std,
            "energy_z": (
                energy - self.energy_mean
            ) / energy_std,

            "abs_slope": abs_slope,
        }

        # During exploratory analysis, update all feature baselines
        # continuously after calculating the current z-scores.
        self.env_mean, self.env_var = self._update_baseline(
            envelope,
            self.p.baseline_alpha,
            self.env_mean,
            self.env_var,
        )

        self.amp_mean, self.amp_var = self._update_baseline(
            amp,
            self.p.baseline_alpha,
            self.amp_mean,
            self.amp_var,
        )

        self.line_length_mean, self.line_length_var = (
            self._update_baseline(
                line_length,
                self.p.baseline_alpha,
                self.line_length_mean,
                self.line_length_var,
            )
        )

        self.energy_mean, self.energy_var = self._update_baseline(
            energy,
            self.p.baseline_alpha,
            self.energy_mean,
            self.energy_var,
        )

        return features