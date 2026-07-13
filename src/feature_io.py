"""Master CSV that includes annotated features across every sample of the IED ABF file"""

from pathlib import Path
import pandas as pd
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

SUMMARY_SHEET_NAME = "Closed Loop Features"

CHANNEL_SUFFIX = {"hippocampus": "_hpc", "thalamus": "_thal"}

# @dataclass
# class FeatureFileResult:
#     abf_stem: str
#     channel_name: str
#     seizure_present: bool
#     seizure_onset_s: Optional[float]
#     num_gt_events: int
#     num_algo_events: int
#     f
#     """Input/output helpers for closed-loop feature traces."""

def save_sample_features_csv(
    path,
    analysis,
    chunk_rows=250_000,
):
    """
    Save complete sample-level feature traces to CSV in chunks.

    Parameters
    ----------
    path:
        Output CSV path.

    analysis:
        Completed ClosedLoopFeatureAnalysis instance.

    chunk_rows:
        Number of samples placed into each temporary DataFrame.
        This controls memory use but does not change the output.
    """
    output_path = Path(path)

    if output_path.suffix.lower() != ".csv":
        output_path = output_path.with_suffix(".csv")

    if analysis.feature_traces is None:
        raise RuntimeError(
            "Feature traces have not been calculated."
        )

    analysis.annotate_samples()

    n_samples = len(analysis.raw)
    first_chunk = True

    for start_idx in range(0, n_samples, chunk_rows):
        end_idx = min(
            start_idx + chunk_rows,
            n_samples,
        )

        chunk_df = analysis.build_sample_dataframe(
            start_idx=start_idx,
            end_idx=end_idx,
        )

        chunk_df.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            float_format="%.8g",
        )

        first_chunk = False

    return output_path