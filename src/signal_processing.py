"""Filtering and event-snapping helpers."""

import numpy as np
from scipy.signal import butter, sosfiltfilt

FILTER_LOW_HZ = 50.0
FILTER_HIGH_HZ = 100.0
FILTER_ORDER = 4

SNAP_HALF_WINDOW_S = 0.150


def bandpass_filter(signal: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase bandpass filter matching the STM32 detector's preprocessing band."""
    sos = butter(
        FILTER_ORDER,
        [FILTER_LOW_HZ, FILTER_HIGH_HZ],
        btype="bandpass",
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
