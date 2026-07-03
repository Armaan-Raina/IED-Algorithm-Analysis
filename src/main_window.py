"""Main application window: file/channel selection, seizure marking,
manual IED flagging, and analysis/export."""

import os

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QInputDialog, QDialog,
)

from src import data_io, signal_processing, scoring, workbook

MIN_ZOOM_SPAN_S = 0.050
LINE_PICK_PIXEL_TOLERANCE = 6

STATE_SEIZURE_MARKING = "seizure_marking"
STATE_FLAGGING = "flagging"
STATE_DONE = "done"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IED Detection GUI")
        self.resize(1400, 800)

        self.recording = None
        self.algo_events_all = None
        self.filtered_full = None

        self.t = None          # pre-seizure time array (or full if no seizure)
        self.raw = None
        self.filtered = None
        self.algo_events = None  # pre-seizure algorithm events
        self.seizure_onset_s = None
        self.seizure_present = False

        self.state = None
        self.flags = {}            # time_s -> Line2D
        self._full_xlim = None
        self._full_ylim = None

        self._dragging = False
        self._temp_line = None
        self._blit_bg = None

        self._build_ui()
        QTimer.singleShot(100, self.start_session)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)

        self.instruction_label = QLabel("Loading...")
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setStyleSheet("font-size: 13px; padding: 4px;")
        layout.addWidget(self.instruction_label)

        self.figure = Figure(figsize=(12, 6))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        btn_row = QHBoxLayout()
        self.no_seizure_btn = QPushButton("No Seizure Present")
        self.no_seizure_btn.clicked.connect(self.on_no_seizure_clicked)
        btn_row.addWidget(self.no_seizure_btn)

        self.flag_btn = QPushButton("Flag Event")
        self.flag_btn.setCheckable(True)
        self.flag_btn.toggled.connect(self.on_flag_mode_toggled)
        btn_row.addWidget(self.flag_btn)

        self.reset_btn = QPushButton("Reset Perspective")
        self.reset_btn.clicked.connect(self.on_reset_perspective)
        btn_row.addWidget(self.reset_btn)

        self.done_btn = QPushButton("Done")
        self.done_btn.clicked.connect(self.on_done_clicked)
        btn_row.addWidget(self.done_btn)

        layout.addLayout(btn_row)
        self.setCentralWidget(central)

        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.ax.callbacks.connect("xlim_changed", self.on_xlim_changed)
        self._clamping = False

        self._set_buttons_for_state(None)

    def _set_buttons_for_state(self, state):
        self.no_seizure_btn.setVisible(state == STATE_SEIZURE_MARKING)
        self.flag_btn.setVisible(state == STATE_FLAGGING)
        self.reset_btn.setVisible(state == STATE_FLAGGING)
        self.done_btn.setVisible(state == STATE_FLAGGING)

    # ------------------------------------------------------------- session

    def start_session(self):
        abf_path, _ = QFileDialog.getOpenFileName(self, "Select .abf recording", "", "Axon Binary Files (*.abf)")
        if not abf_path:
            self.close()
            return

        csv_path, _ = QFileDialog.getOpenFileName(self, "Select algorithm output .csv", "", "CSV Files (*.csv)")
        if not csv_path:
            self.close()
            return

        items = ["Hippocampus (Channel 1)", "Thalamus (Channel 2)"]
        choice, ok = QInputDialog.getItem(self, "Select Channel", "Which channel do you want to evaluate?",
                                           items, 0, False)
        if not ok:
            self.close()
            return
        channel_index = data_io.CHANNEL_HIPPOCAMPUS if choice.startswith("Hippocampus") else data_io.CHANNEL_THALAMUS

        try:
            self.recording = data_io.load_abf_channel(abf_path, channel_index)
            self.algo_events_all = data_io.load_algo_events(csv_path, channel_index)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load files", str(exc))
            self.close()
            return

        self.filtered_full = signal_processing.bandpass_filter(self.recording.raw, self.recording.fs)

        self._enter_seizure_marking()

    # ------------------------------------------------------- seizure stage

    def _enter_seizure_marking(self):
        self.state = STATE_SEIZURE_MARKING
        self._set_buttons_for_state(self.state)
        self.instruction_label.setText(
            "Step 1 of 2 — Does this trace contain a seizure? "
            "Click on the trace at the point the seizure begins, or click 'No Seizure Present' below."
        )
        self.ax.clear()
        t = self.recording.t
        self.ax.plot(t, self.recording.raw, linewidth=0.6, color="#4c72b0", label="Raw")
        self.ax.plot(t, self.filtered_full, linewidth=0.6, color="#c44e52", alpha=0.8, label="Filtered (50-100Hz)")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel(f"{self.recording.channel_name.capitalize()} signal")
        self.ax.legend(loc="upper right")
        self.ax.set_xlim(t[0], t[-1])
        self.canvas.draw_idle()

    def on_no_seizure_clicked(self):
        self.seizure_present = False
        self.seizure_onset_s = None
        self._finish_seizure_marking()

    def _confirm_seizure_click(self, x):
        reply = QMessageBox.question(
            self, "Confirm seizure onset",
            f"Mark seizure onset at t = {x:.3f} s?\n\nOnly data before this point will be used for IED labeling.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            self.seizure_present = True
            self.seizure_onset_s = float(x)
            self._finish_seizure_marking()

    def _finish_seizure_marking(self):
        t_full = self.recording.t
        if self.seizure_present:
            end_idx = int(np.searchsorted(t_full, self.seizure_onset_s, side="left"))
        else:
            end_idx = len(t_full)

        self.t = t_full[:end_idx]
        self.raw = self.recording.raw[:end_idx]
        self.filtered = self.filtered_full[:end_idx]

        if self.seizure_present:
            self.algo_events = self.algo_events_all[self.algo_events_all < self.seizure_onset_s]
        else:
            self.algo_events = self.algo_events_all

        self._enter_flagging()

    # ------------------------------------------------------- flagging stage

    def _enter_flagging(self):
        self.state = STATE_FLAGGING
        self.flags = {}
        self._set_buttons_for_state(self.state)
        self.instruction_label.setText(
            "Step 2 of 2 — Scroll to zoom, drag with the pan tool to move around (min zoom: 50 ms across). "
            "Click 'Flag Event', then click/drag on an IED to mark it (snaps to the nearest sharp deflection). "
            "Click an existing red dashed marker to remove it. Click 'Done' when finished."
        )
        self.ax.clear()
        self.ax.plot(self.t, self.raw, linewidth=0.6, color="#4c72b0", label="Raw")
        self.ax.plot(self.t, self.filtered, linewidth=0.6, color="#c44e52", alpha=0.8, label="Filtered (50-100Hz)")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel(f"{self.recording.channel_name.capitalize()} signal")
        self.ax.legend(loc="upper right")
        if len(self.t) > 0:
            self._full_xlim = (self.t[0], self.t[-1])
            self.ax.set_xlim(*self._full_xlim)
        self.canvas.draw_idle()
        self._full_ylim = self.ax.get_ylim()

    def on_flag_mode_toggled(self, checked):
        self.flag_btn.setStyleSheet("background-color: #ffb3b3;" if checked else "")

    def on_reset_perspective(self):
        if self._full_xlim is not None:
            self.ax.set_xlim(*self._full_xlim)
        if self._full_ylim is not None:
            self.ax.set_ylim(*self._full_ylim)
        self.canvas.draw_idle()

    def on_done_clicked(self):
        reply = QMessageBox.question(
            self, "Confirm done",
            f"You have flagged {len(self.flags)} event(s). Proceed to analysis? "
            "No further edits will be possible after this.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return
        self._run_analysis()

    # -------------------------------------------------------- mouse events

    def _pixel_x(self, data_x):
        return self.ax.transData.transform((data_x, 0))[0]

    def _find_flag_near_pixel(self, pixel_x):
        for t_flag in self.flags:
            if abs(self._pixel_x(t_flag) - pixel_x) <= LINE_PICK_PIXEL_TOLERANCE:
                return t_flag
        return None

    def on_press(self, event):
        if self.state not in (STATE_SEIZURE_MARKING, STATE_FLAGGING):
            return
        if event.inaxes != self.ax or event.xdata is None:
            return
        if self.toolbar.mode != "":
            return  # pan/zoom tool active; let matplotlib handle it

        if self.state == STATE_SEIZURE_MARKING:
            self._confirm_seizure_click(event.xdata)
            return

        # STATE_FLAGGING
        hit = self._find_flag_near_pixel(event.x)
        if hit is not None:
            self._remove_flag(hit)
            return

        if not self.flag_btn.isChecked():
            return

        self._dragging = True
        self._temp_line = self.ax.axvline(event.xdata, color="red", linestyle=":", linewidth=1.5, animated=True)
        self.canvas.draw()
        self._blit_bg = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self._temp_line)
        self.canvas.blit(self.ax.bbox)

    def on_motion(self, event):
        if not self._dragging or event.inaxes != self.ax or event.xdata is None:
            return
        self._temp_line.set_xdata([event.xdata, event.xdata])
        self.canvas.restore_region(self._blit_bg)
        self.ax.draw_artist(self._temp_line)
        self.canvas.blit(self.ax.bbox)

    def on_release(self, event):
        if not self._dragging:
            return
        self._dragging = False
        release_x = event.xdata if (event.inaxes == self.ax and event.xdata is not None) else self._temp_line.get_xdata()[0]

        self._temp_line.set_animated(False)
        self._temp_line.remove()
        self._temp_line = None
        self._blit_bg = None

        snapped_t = signal_processing.snap_to_nearest_peak(release_x, self.t, self.filtered)
        self._add_flag(snapped_t)
        self.canvas.draw_idle()

    def on_scroll(self, event):
        if self.state not in (STATE_SEIZURE_MARKING, STATE_FLAGGING):
            return
        if event.inaxes != self.ax or event.xdata is None:
            return
        xlim = self.ax.get_xlim()
        span = xlim[1] - xlim[0]
        factor = 0.8 if event.button == "up" else 1.25
        new_span = span * factor
        if self.state == STATE_FLAGGING:
            new_span = max(new_span, MIN_ZOOM_SPAN_S)
        if self._full_xlim is not None:
            new_span = min(new_span, self._full_xlim[1] - self._full_xlim[0])

        cursor = event.xdata
        left_frac = (cursor - xlim[0]) / span if span > 0 else 0.5
        new_left = cursor - left_frac * new_span
        new_right = new_left + new_span
        self.ax.set_xlim(new_left, new_right)
        self.canvas.draw_idle()

    def on_xlim_changed(self, ax):
        if self._clamping or self.state != STATE_FLAGGING:
            return
        xlim = ax.get_xlim()
        span = xlim[1] - xlim[0]
        if span < MIN_ZOOM_SPAN_S:
            center = (xlim[0] + xlim[1]) / 2
            self._clamping = True
            ax.set_xlim(center - MIN_ZOOM_SPAN_S / 2, center + MIN_ZOOM_SPAN_S / 2)
            self._clamping = False
            self.canvas.draw_idle()

    # ---------------------------------------------------------------- flags

    def _add_flag(self, t_flag):
        line = self.ax.axvline(t_flag, color="red", linestyle="--", linewidth=1.2)
        self.flags[t_flag] = line

    def _remove_flag(self, t_flag):
        line = self.flags.pop(t_flag)
        line.remove()
        self.canvas.draw_idle()

    # -------------------------------------------------------------- analysis

    def _run_analysis(self):
        self.state = STATE_DONE
        self._set_buttons_for_state(self.state)

        gt_times = sorted(self.flags.keys())
        result = scoring.score_events(gt_times, list(self.algo_events), tolerance_s=scoring.MATCH_TOLERANCE_S)

        sens_str = f"{result.sensitivity:.3f}" if result.sensitivity is not None else "N/A"
        self.instruction_label.setText(
            f"Analysis complete — TP: {result.tp}  FP: {result.fp}  FN: {result.fn}  Sensitivity: {sens_str}"
        )

        QMessageBox.information(
            self, "Analysis Results",
            f"Ground truth events: {len(gt_times)}\n"
            f"Algorithm events (pre-seizure): {len(self.algo_events)}\n\n"
            f"TP: {result.tp}\nFP: {result.fp}\nFN: {result.fn}\n"
            f"Sensitivity: {sens_str}",
        )

        self._save_to_workbook(result, gt_times)

    def _save_to_workbook(self, result, gt_times):
        dlg = QFileDialog(self, "Select or create master workbook", "", "Excel Workbook (*.xlsx)")
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.setAcceptMode(QFileDialog.AcceptOpen)
        if dlg.exec_() != QDialog.Accepted:
            return
        selected = dlg.selectedFiles()
        if not selected:
            return
        path = selected[0]
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"

        file_result = workbook.FileResult(
            abf_stem=self.recording.file_stem,
            channel_name=self.recording.channel_name,
            seizure_present=self.seizure_present,
            seizure_onset_s=self.seizure_onset_s,
            num_gt_events=len(gt_times),
            num_algo_events=len(self.algo_events),
            tp=result.tp,
            fp=result.fp,
            fn=result.fn,
            sensitivity=result.sensitivity,
            tp_pairs=result.tp_pairs,
        )

        try:
            workbook.save_result(path, file_result)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to save workbook", str(exc))
            return

        QMessageBox.information(self, "Saved", f"Results saved to:\n{path}\n\nSheet: {workbook.sheet_name_for(file_result)}")
