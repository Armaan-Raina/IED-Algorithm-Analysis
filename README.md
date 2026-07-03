# IED Detection GUI

Manual IED (interictal epileptiform discharge) labeling and algorithm-evaluation
tool. Loads a 2-channel, 1-sweep `.abf` recording alongside a `.csv` of
algorithm-detected IED timestamps, lets you mark the seizure onset (if any) and
manually flag ground-truth IEDs in the pre-seizure trace, then scores the
algorithm against your manual labels and appends the results to a master Excel
workbook.

## Setup

Requires Python 3.11 (PyQt5/pyabf/PyInstaller are not yet reliable on newer
Python versions). From this folder:

```
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Running

```
.venv\Scripts\python.exe main.py
```

You'll be prompted, in order, to:

1. Select the `.abf` recording (1 sweep, 2 channels).
2. Select the algorithm-output `.csv` (must contain `hpc_ied_time_s` and
   `thal_ied_time_s` columns; other columns are ignored).
3. Choose which channel to evaluate — Hippocampus (channel 1) or Thalamus
   (channel 2).

### Step 1 — Seizure marking

The full trace is shown (raw + a 50-100Hz filtered overlay). Click on the
trace at the point the seizure begins, or click **No Seizure Present** if
there isn't one. Only data before the seizure onset is used from this point
on — seizure and post-seizure data (and any algorithm-detected events in that
region) are excluded from everything downstream.

### Step 2 — Manual flagging

- **Scroll** to zoom in/out on the x-axis (centered on the cursor); zoom is
  capped at 50 ms of trace filling the screen. Use the matplotlib toolbar's
  pan tool to move around while zoomed in.
- Click **Flag Event**, then click (or click-drag) on an IED in the trace. On
  release, the flag snaps to the most extreme (most negative) point of the
  filtered signal within ±150 ms of where you clicked, and is marked with a
  red dashed line.
- Click directly on an existing red dashed line at any time to remove
  (un-flag) it.
- **Reset Perspective** returns the view to the full pre-seizure trace.
- Click **Done** when you've flagged everything. No further edits are
  possible after this.

### Analysis + export

After **Done**, the app matches your manually-flagged (ground truth) events
against the algorithm's pre-seizure timestamps for the selected channel,
using a ±100 ms match tolerance (one-to-one nearest-neighbor matching). It
reports:

- **TP** — ground truth event matched to an algorithm event
- **FN** — ground truth event with no matching algorithm event
- **FP** — algorithm event with no matching ground truth event
- **Sensitivity** — TP / (TP + FN)

Note: specificity/TN are not computed. IED detection is a point-event task
with no natural "negative" time window to count true negatives against, so
those two metrics are intentionally omitted from the output.

You'll then be asked to select or create a master `.xlsx` workbook. Results
for this `.abf` + channel are written to a sheet named
`<abf filename>_hpc` or `<abf filename>_thal` (re-running the same
abf/channel combination overwrites that sheet). Each file-level sheet
contains the summary metrics plus a table of TP events (ground-truth time,
matched algorithm time, and a blank "Features (TBD)" column to be filled in
once you decide which features to compute). The first sheet, `Summary`, is
recomputed from all other sheets on every save and holds the number of
recordings, aggregate TP/FP/FN/sensitivity across the whole workbook, and a
per-recording breakdown table.

## Tests

Headless smoke tests (no real `.abf` file needed — they use synthetic data
and Qt's offscreen platform):

```
.venv\Scripts\python.exe tests\test_core_logic.py
.venv\Scripts\python.exe tests\test_gui_smoke.py
```

## Building the standalone .exe

```
.venv\Scripts\pyinstaller.exe --noconfirm --name IED_Detection_GUI --windowed --onefile main.py
```

The built exe will be at `dist\IED_Detection_GUI.exe` — copy that single file
to any Windows machine to run it (no Python install needed there).

**Known issue:** Windows Defender's real-time protection frequently
quarantines PyInstaller's generic bootloader stub as soon as it's written to
`dist\`, causing the build to fail with a "file contains a virus" error before
it can embed the manifest/icon. This is a well-documented PyInstaller false
positive, not an actual virus. If you hit this, either:

- Temporarily exclude this project folder (or just `build\` and `dist\`) from
  Defender's real-time scanning while building:
  `Add-MpPreference -ExclusionPath "<path to this folder>"` (run as
  Administrator), then remove the exclusion afterward with
  `Remove-MpPreference -ExclusionPath "<path>"` if you don't want to keep it, or
- Build on a machine/CI runner without the same Defender policy, then copy
  the resulting `.exe` over.
