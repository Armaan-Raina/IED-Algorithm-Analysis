"""Filtering and event-snapping helpers."""

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

FILTER_LOWPASS_HZ = 300.0
FILTER_ORDER = 4

SNAP_HALF_WINDOW_S = 0.150


def bandpass_filter(signal: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase lowpass filter at 300Hz for IED detection."""
    sos = butter(
        FILTER_ORDER,
        FILTER_LOWPASS_HZ,
        btype="low",
        fs=fs,
        output="sos",
    )
    return sosfiltfilt(sos, signal)


def snap_to_nearest_peak(click_time_s: float, t: np.ndarray, filtered_signal: np.ndarray,
                          half_window_s: float = SNAP_HALF_WINDOW_S) -> float:
    """Snap a manually-placed flag to the argmin (most negative deflection) of the
    filtered signal within +/- half_window_s of the click time."""
    lo = np.searchsorted(t, click_time_s - half_window_s, side="left")
    hi = np.searchsorted(t, click_time_s + half_window_s, side="right")
    if hi <= lo:
        return click_time_s
    window = filtered_signal[lo:hi]
    offset = int(np.argmin(window))
    return float(t[lo + offset])


def find_preliminary_events(t: np.ndarray, filtered_signal: np.ndarray, fs: float,
                           min_height: float = None, min_distance_s: float = 0.050) -> np.ndarray:
    """Detect candidate IED events as peaks in the filtered signal.

    Args:
        t: Time array
        filtered_signal: Bandpass-filtered signal
        fs: Sampling frequency
        min_height: Minimum peak height (inverted peaks). If None, uses signal's std dev.
        min_distance_s: Minimum time between peaks in seconds

    Returns:
        Array of event times (seconds) for candidate peaks
    """
    if len(filtered_signal) == 0:
        return np.array([])

    if min_height is None:
        min_height = np.std(filtered_signal)

    min_distance_samples = int(min_distance_s * fs)

    peaks, _ = find_peaks(-filtered_signal, height=min_height, distance=min_distance_samples)

    if len(peaks) == 0:
        return np.array([])

    event_times = t[peaks]
    return np.sort(event_times)
