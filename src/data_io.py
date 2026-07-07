"""Loading of .abf recordings and algorithm-output .csv files."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyabf

CHANNEL_HIPPOCAMPUS = 0
CHANNEL_THALAMUS = 1

CHANNEL_NAMES = {
    CHANNEL_HIPPOCAMPUS: "hippocampus",
    CHANNEL_THALAMUS: "thalamus",
}

CSV_COLUMN_BY_CHANNEL = {
    CHANNEL_HIPPOCAMPUS: "hpc_ied_time_s",
    CHANNEL_THALAMUS: "thal_ied_time_s",
}


@dataclass
class Recording:
    file_stem: str
    fs: float
    t: np.ndarray
    raw: np.ndarray
    channel_index: int
    units: str = "mV"

    @property
    def channel_name(self) -> str:
        return CHANNEL_NAMES[self.channel_index]


def load_abf_channel(abf_path: str, channel_index: int) -> Recording:
    """Load a single channel from a single-sweep, 2-channel .abf file."""
    abf = pyabf.ABF(abf_path)

    if abf.channelCount < 2:
        raise ValueError(
            f"Expected 2 channels in {abf_path}, found {abf.channelCount}."
        )

    abf.setSweep(sweepNumber=0, channel=channel_index)
    raw = abf.sweepY.copy()
    t = abf.sweepX.copy()
    fs = float(abf.dataRate)

    # Get units and convert to millivolts if needed
    units = abf.sweepUnitsY if hasattr(abf, 'sweepUnitsY') else "mV"

    # Always convert to millivolts
    units = "mV"

    # Detect current units from ABF and convert accordingly
    if units and "pv" in units.lower():  # Picovolts
        raw = raw / 1e12 * 1000  # Convert pV to mV
    elif units and "nv" in units.lower():  # Nanovolts
        raw = raw / 1e9 * 1000  # Convert nV to mV
    elif units and "uv" in units.lower():  # Microvolts
        raw = raw / 1e6 * 1000  # Convert µV to mV
    elif units and "v" in units.lower() and "mv" not in units.lower():  # Volts (but not mV)
        raw = raw * 1000  # Convert V to mV
    else:
        # Check if data range suggests it's in volts (typical range 0.001-0.1)
        # while mV should be 1-100
        if np.max(np.abs(raw)) < 0.2:  # Likely in volts
            raw = raw * 1000  # Convert V to mV

    from pathlib import Path
    file_stem = Path(abf_path).stem

    return Recording(file_stem=file_stem, fs=fs, t=t, raw=raw, channel_index=channel_index, units=units)


def load_algo_events(csv_path: str, channel_index: int) -> np.ndarray:
    """Load algorithm-detected IED timestamps (seconds) for the given channel."""
    df = pd.read_csv(csv_path)
    col = CSV_COLUMN_BY_CHANNEL[channel_index]
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in {csv_path}. Columns present: {list(df.columns)}")
    values = df[col].dropna().to_numpy(dtype=float)
    return np.sort(values)
