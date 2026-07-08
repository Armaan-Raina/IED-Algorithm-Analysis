"""Main application window: file/channel selection, seizure marking,
manual IED flagging, and analysis/export."""

import os

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QInputDialog, QDialog, QTextEdit,
    QScrollBar, QFrame,
)

from . import data_io, signal_processing, scoring, workbook
import json
from pathlib import Path

MIN_ZOOM_SPAN_S = 0.050
LINE_PICK_PIXEL_TOLERANCE = 6
EVENT_CONTEXT_SECONDS = 0.5  # Zoom to ±500ms around clicked event

STATE_IDLE = "idle"
STATE_SEIZURE_MARKING = "seizure_marking"
STATE_FLAGGING = "flagging"
STATE_DONE = "done"

PRELIMINARY_EVENT_COLOR = "#ff9500"   # Orange (candidates - pending validation)
VALIDATED_EVENT_COLOR = "#2ca02c"     # Green (all validated events)
REJECTED_EVENT_COLOR = "#d62728"      # Red (rejected events)
ALGORITHM_EVENT_COLOR = "#1f77b4"     # Blue (reference)
SELECTED_EVENT_COLOR = "#ffd700"      # Gold (highlighted when selected)


class CustomNavigationToolbar(NavigationToolbar2QT):
    """Custom toolbar with text labels instead of icons."""

    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self._remove_all_tools()
        self._add_text_tools()

    def _remove_all_tools(self):
        """Remove all default toolbar widgets."""
        for action in self.actions():
            self.removeAction(action)

    def _add_text_tools(self):
        """Add text-labeled tools."""
        self.addAction("Pan", self.pan)


class InstructionDialog(QDialog):
    """Large instruction dialog that doesn't overlay the plot."""

    def __init__(self, parent, title, instructions):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 600, 300)

        layout = QVBoxLayout(self)

        text_edit = QTextEdit()
        text_edit.setText(instructions)
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("font-size: 13px; padding: 10px;")
        layout.addWidget(text_edit)

        ok_btn = QPushButton("OK, Got It!")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setMinimumHeight(40)
        layout.addWidget(ok_btn)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IED Detection & Validation Tool")
        self.resize(1800, 1000)

        # File and data state
        self.recording = None
        self.algo_events_all = None
        self.filtered_full = None

        # Pre-seizure data (or full if no seizure)
        self.t = None
        self.raw = None
        self.filtered = None
        self.algo_events = None
        self.preliminary_events = None
        self.seizure_onset_s = None
        self.seizure_present = False

        # Event state
        self.state = STATE_IDLE
        self.validated_flags = {}          # time_s -> Line2D (all validated events)
        self.preliminary_flags = {}        # time_s -> Line2D (auto-detected candidates)
        self.rejected_flags = set()        # set of rejected preliminary event times
        self.undo_stack = []               # for undo functionality
        self.selected_event = None         # currently selected event for validation/rejection
        self.pool_view = "candidates"      # current pool view: "candidates", "accepted", or "rejected"

        # UI state
        self._full_xlim = None
        self._full_ylim = None
        self._full_ycenter = None           # Y center for signal centering
        self._dragging = False
        self._temp_line = None
        self._blit_bg = None
        self._clamping = False
        self._show_filtered = False         # Toggle for filtered signal
        self.filtered_line = None           # Line object for filtered signal
        self._current_candidate_idx = 0     # Current candidate being viewed
        self._candidate_list = []           # List of all candidates (preliminary + manual)

        self._build_ui()
        self.show_idle_screen()

    # ================================================================== UI SETUP

    def _build_ui(self):
        """Build the complete UI layout."""
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Top toolbar
        toolbar_layout = self._build_toolbar()
        main_layout.addLayout(toolbar_layout)

        # Main content: just the plot, no sidebar
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(2)

        self.figure = Figure(figsize=(14, 8))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = CustomNavigationToolbar(self.canvas, self)

        plot_layout.addWidget(self.toolbar)

        # Plot with scrollbars perfectly aligned
        plot_area_layout = QHBoxLayout()
        plot_area_layout.setContentsMargins(0, 0, 0, 0)
        plot_area_layout.setSpacing(0)

        # Plot canvas
        plot_area_layout.addWidget(self.canvas, stretch=1)

        # Y slider (vertical, to the right of plot, exact height match)
        self.y_slider = QScrollBar(Qt.Vertical)
        self.y_slider.setMaximum(1000)
        self.y_slider.setValue(500)
        self.y_slider.setFixedWidth(20)
        self.y_slider.sliderMoved.connect(self.on_y_slider)
        self.y_slider.setStyleSheet("""
            QScrollBar:vertical {
                border: 1px solid #ccc;
                background: #f0f0f0;
                width: 18px;
            }
            QScrollBar::handle:vertical {
                background: #0078d4;
                border-radius: 4px;
                min-height: 50px;
            }
            QScrollBar::handle:vertical:hover {
                background: #005a9e;
            }
        """)
        plot_area_layout.addWidget(self.y_slider)

        plot_layout.addLayout(plot_area_layout, stretch=1)

        # X slider (horizontal, below plot, exact width match)
        x_slider_layout = QHBoxLayout()
        x_slider_layout.setContentsMargins(0, 0, 0, 0)
        x_slider_layout.setSpacing(0)

        self.x_slider = QScrollBar(Qt.Horizontal)
        self.x_slider.setMaximum(1000)
        self.x_slider.setValue(500)
        self.x_slider.setFixedHeight(20)
        self.x_slider.sliderMoved.connect(self.on_x_slider)
        self.x_slider.setStyleSheet("""
            QScrollBar:horizontal {
                border: 1px solid #ccc;
                background: #f0f0f0;
                height: 18px;
            }
            QScrollBar::handle:horizontal {
                background: #0078d4;
                border-radius: 4px;
                min-width: 50px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #005a9e;
            }
        """)
        x_slider_layout.addWidget(self.x_slider, stretch=1)

        # Spacer for corner alignment with Y scrollbar
        spacer = QWidget()
        spacer.setFixedSize(20, 20)
        x_slider_layout.addWidget(spacer)

        plot_layout.addLayout(x_slider_layout)

        main_layout.addWidget(plot_widget, stretch=1)

        # Bottom: Action buttons (much taller)
        action_layout = self._build_action_buttons()
        main_layout.addLayout(action_layout)

        self.setCentralWidget(central)

        # Connect plot events
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.ax.callbacks.connect("xlim_changed", self.on_xlim_changed)

        # Connect pool view buttons (after instruction_label is created)
        self.pool_candidates_btn.clicked.connect(lambda: self.on_pool_view_change("candidates"))
        self.pool_accepted_btn.clicked.connect(lambda: self.on_pool_view_change("accepted"))
        self.pool_rejected_btn.clicked.connect(lambda: self.on_pool_view_change("rejected"))

    def _build_toolbar(self):
        """Build the top toolbar with browse and file info."""
        layout = QHBoxLayout()

        self.browse_btn = QPushButton("Browse Files")
        self.browse_btn.setMinimumHeight(45)
        self.browse_btn.clicked.connect(self.on_browse_files)
        layout.addWidget(self.browse_btn)

        layout.addSpacing(16)

        # File info labels
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.file_label)

        self.channel_label = QLabel("")
        self.channel_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.channel_label)

        layout.addStretch()

        return layout

    def _build_action_buttons(self):
        """Build the bottom action buttons - much larger with keyboard shortcuts."""
        layout = QHBoxLayout()
        layout.setSpacing(6)

        # Event navigation buttons (only visible during flagging)
        self.prev_candidate_btn = QPushButton("◀ Prev\n")
        self.prev_candidate_btn.setMinimumHeight(80)
        self.prev_candidate_btn.setMinimumWidth(100)
        self.prev_candidate_btn.clicked.connect(self.on_prev_candidate)
        self.prev_candidate_btn.setVisible(False)
        layout.addWidget(self.prev_candidate_btn)

        self.next_candidate_btn = QPushButton("Next\n ▶")
        self.next_candidate_btn.setMinimumHeight(80)
        self.next_candidate_btn.setMinimumWidth(100)
        self.next_candidate_btn.clicked.connect(self.on_next_candidate)
        self.next_candidate_btn.setVisible(False)
        layout.addWidget(self.next_candidate_btn)

        self.candidate_counter = QLabel("-- / --")
        self.candidate_counter.setStyleSheet("font-size: 12px; font-weight: bold; min-width: 60px;")
        self.candidate_counter.setAlignment(Qt.AlignCenter)
        self.candidate_counter.setVisible(False)
        layout.addWidget(self.candidate_counter)

        layout.addSpacing(12)

        # Pool view toggle buttons
        self.pool_candidates_btn = QPushButton("Candidates\n[1]")
        self.pool_candidates_btn.setMinimumHeight(80)
        self.pool_candidates_btn.setMinimumWidth(120)
        self.pool_candidates_btn.setCheckable(True)
        self.pool_candidates_btn.setChecked(True)
        self.pool_candidates_btn.setVisible(False)
        layout.addWidget(self.pool_candidates_btn)

        self.pool_accepted_btn = QPushButton("Validated\n[2]")
        self.pool_accepted_btn.setMinimumHeight(80)
        self.pool_accepted_btn.setMinimumWidth(120)
        self.pool_accepted_btn.setCheckable(True)
        self.pool_accepted_btn.setVisible(False)
        layout.addWidget(self.pool_accepted_btn)

        self.pool_rejected_btn = QPushButton("Rejected\n[3]")
        self.pool_rejected_btn.setMinimumHeight(80)
        self.pool_rejected_btn.setMinimumWidth(120)
        self.pool_rejected_btn.setCheckable(True)
        self.pool_rejected_btn.setVisible(False)
        layout.addWidget(self.pool_rejected_btn)

        layout.addSpacing(12)

        # Toggle filtered signal button
        self.toggle_filtered_btn = QPushButton("Show\nFiltered [T]")
        self.toggle_filtered_btn.setMinimumHeight(80)
        self.toggle_filtered_btn.setMinimumWidth(140)
        self.toggle_filtered_btn.setCheckable(True)
        self.toggle_filtered_btn.setChecked(False)
        self.toggle_filtered_btn.toggled.connect(self.on_toggle_filtered)
        self.toggle_filtered_btn.setVisible(False)
        layout.addWidget(self.toggle_filtered_btn)

        layout.addSpacing(12)

        self.no_seizure_btn = QPushButton("No Seizure\nPresent")
        self.no_seizure_btn.setMinimumHeight(80)
        self.no_seizure_btn.setMinimumWidth(140)
        self.no_seizure_btn.clicked.connect(self.on_no_seizure_clicked)
        self.no_seizure_btn.setVisible(False)
        layout.addWidget(self.no_seizure_btn)

        self.seizure_mark_btn = QPushButton("Mark Seizure\nOnset")
        self.seizure_mark_btn.setMinimumHeight(80)
        self.seizure_mark_btn.setMinimumWidth(140)
        self.seizure_mark_btn.setEnabled(False)
        self.seizure_mark_btn.clicked.connect(self.on_seizure_mark_click)
        self.seizure_mark_btn.setVisible(False)
        layout.addWidget(self.seizure_mark_btn)

        layout.addSpacing(12)

        self.flag_btn = QPushButton("Flag Event\n[F]")
        self.flag_btn.setMinimumHeight(80)
        self.flag_btn.setMinimumWidth(140)
        self.flag_btn.setCheckable(True)
        self.flag_btn.toggled.connect(self.on_flag_mode_toggled)
        self.flag_btn.setVisible(False)
        layout.addWidget(self.flag_btn)

        self.validate_btn = QPushButton("Validate\n[V]")
        self.validate_btn.setMinimumHeight(80)
        self.validate_btn.setMinimumWidth(140)
        self.validate_btn.clicked.connect(self.on_validate_selected)
        self.validate_btn.setVisible(False)
        layout.addWidget(self.validate_btn)

        self.reject_btn = QPushButton("Reject\n[R]")
        self.reject_btn.setMinimumHeight(80)
        self.reject_btn.setMinimumWidth(140)
        self.reject_btn.clicked.connect(self.on_reject_selected)
        self.reject_btn.setVisible(False)
        layout.addWidget(self.reject_btn)

        layout.addSpacing(12)

        self.reset_btn = QPushButton("Reset\n[A]")
        self.reset_btn.setMinimumHeight(80)
        self.reset_btn.setMinimumWidth(140)
        self.reset_btn.clicked.connect(self.on_reset_perspective)
        self.reset_btn.setVisible(False)
        layout.addWidget(self.reset_btn)

        self.undo_btn = QPushButton("Undo\n[U]")
        self.undo_btn.setMinimumHeight(80)
        self.undo_btn.setMinimumWidth(140)
        self.undo_btn.clicked.connect(self.on_undo)
        self.undo_btn.setVisible(False)
        layout.addWidget(self.undo_btn)

        layout.addSpacing(12)

        self.done_btn = QPushButton("Done\n[D]")
        self.done_btn.setMinimumHeight(80)
        self.done_btn.setMinimumWidth(140)
        self.done_btn.clicked.connect(self.on_done_clicked)
        self.done_btn.setVisible(False)
        layout.addWidget(self.done_btn)

        layout.addStretch()

        return layout

    # ================================================================ SCREEN STATES

    def show_idle_screen(self):
        """Show the idle screen prompting for file browse."""
        self.state = STATE_IDLE
        self.ax.clear()
        self.ax.text(0.5, 0.5, "Click 'Browse Files' to load ABF and CSV files",
                    ha="center", va="center", fontsize=16, transform=self.ax.transAxes)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.canvas.draw_idle()

    # ============================================================== FILE LOADING

    def on_browse_files(self):
        """Handle browse files button click."""
        abf_path, _ = QFileDialog.getOpenFileName(
            self, "Select ABF recording", "", "Axon Binary Files (*.abf)"
        )
        if not abf_path:
            return

        csv_path, _ = QFileDialog.getOpenFileName(
            self, "Select algorithm output CSV", "", "CSV Files (*.csv)"
        )
        if not csv_path:
            return

        items = ["Hippocampus (Channel 1)", "Thalamus (Channel 2)"]
        choice, ok = QInputDialog.getItem(
            self, "Select Channel", "Which channel do you want to evaluate?",
            items, 0, False
        )
        if not ok:
            return

        channel_index = (
            data_io.CHANNEL_HIPPOCAMPUS
            if choice.startswith("Hippocampus")
            else data_io.CHANNEL_THALAMUS
        )

        try:
            self.recording = data_io.load_abf_channel(abf_path, channel_index)
            self.algo_events_all = data_io.load_algo_events(csv_path, channel_index)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load files", str(exc))
            return

        self.file_label.setText(f"File: {self.recording.file_stem}")
        self.channel_label.setText(f"Channel: {self.recording.channel_name.capitalize()}")

        self.filtered_full = signal_processing.bandpass_filter(
            self.recording.raw, self.recording.fs
        )

        self._enter_seizure_marking()

    # ========================================================== SEIZURE MARKING STAGE

    def _enter_seizure_marking(self):
        """Enter the seizure marking stage."""
        self.state = STATE_SEIZURE_MARKING

        instructions = (
            "STEP 1: SEIZURE MARKING\n\n"
            "Review the full trace shown below.\n\n"
            "If a SEIZURE is present:\n"
            "  • Click on the trace at the exact point where the seizure BEGINS\n"
            "  • Then click the 'Mark Seizure Onset' button to confirm\n\n"
            "If NO SEIZURE is present:\n"
            "  • Click the 'No Seizure Present' button\n\n"
            "CONTROLS:\n"
            "  • Scroll wheel: Zoom in/out\n"
            "  • Use 'Home' button in toolbar to reset view\n"
            "  • Use X/Y scrollbars to navigate the trace"
        )

        dlg = InstructionDialog(self, "Seizure Marking Instructions", instructions)
        dlg.exec_()

        self.no_seizure_btn.setVisible(True)
        self.seizure_mark_btn.setVisible(True)
        self.flag_btn.setVisible(False)
        self.validate_btn.setVisible(False)
        self.reject_btn.setVisible(False)
        self.reset_btn.setVisible(False)
        self.undo_btn.setVisible(False)
        self.done_btn.setVisible(False)

        self.ax.clear()
        t = self.recording.t
        self.ax.plot(t, self.recording.raw, linewidth=0.6, color="#4c72b0", label="Raw")
        self.ax.set_xlabel("Time (s)", fontsize=12)
        self.ax.set_ylabel(f"{self.recording.channel_name.capitalize()} signal", fontsize=12)
        self.ax.legend(loc="upper right", fontsize=10)
        self.ax.set_xlim(t[0], t[-1])
        self.ax.set_ylabel(f"{self.recording.channel_name.capitalize()} signal ({self.recording.units})", fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

        self._full_xlim = (t[0], t[-1])
        self._full_ylim = self.ax.get_ylim()
        # Calculate Y center for signal centering
        self._full_ycenter = (self._full_ylim[0] + self._full_ylim[1]) / 2
        self._update_sliders()
        self._seizure_click_pending = False

    def on_no_seizure_clicked(self):
        """User clicked 'No Seizure Present'."""
        self.seizure_present = False
        self.seizure_onset_s = None
        self._finish_seizure_marking()

    def on_seizure_mark_click(self):
        """User clicked 'Mark Seizure Onset' button."""
        if self._seizure_click_pending:
            self._confirm_and_finish_seizure()

    def on_press(self, event):
        """Handle mouse press on the plot."""
        if self.state not in (STATE_SEIZURE_MARKING, STATE_FLAGGING):
            return
        if event.inaxes != self.ax or event.xdata is None:
            return
        if self.toolbar.mode != "":
            return

        if self.state == STATE_SEIZURE_MARKING:
            self._seizure_click_pending = True
            self.seizure_mark_btn.setEnabled(True)
            self.seizure_mark_btn.setText(f"Mark Seizure\nOnset @ {event.xdata:.3f}s")
            self._seizure_click_time = event.xdata

            # Remove any previous seizure marker line
            if self._temp_line is not None:
                self._temp_line.remove()

            # Add a tall vertical line at the click position for precision
            self._temp_line = self.ax.axvline(event.xdata, color="#ff0000", linestyle="-", linewidth=2, alpha=0.7)
            self.canvas.draw_idle()
            return

        # STATE_FLAGGING - check if clicking on event (select, don't auto-validate)
        hit = None
        if self.pool_view == "candidates":
            hit = self._find_preliminary_flag_near_pixel(event.x)
            if hit is not None:
                self._select_event(hit, "candidate")
        elif self.pool_view == "accepted":
            hit = self._find_validated_flag_near_pixel(event.x)
            if hit is not None:
                self._select_event(hit, "validated")
        elif self.pool_view == "rejected":
            # For rejected pool, check rejected_flags_lines
            for t_flag in self.rejected_flags_lines:
                if abs(self._pixel_x(t_flag) - event.x) <= LINE_PICK_PIXEL_TOLERANCE:
                    hit = t_flag
                    break
            if hit is not None:
                self._select_event(hit, "rejected")

        if hit is not None:
            return

        # Manual flagging (only in candidates pool)
        if not self.flag_btn.isChecked() or self.pool_view != "candidates":
            return

        self._dragging = True
        self._temp_line = self.ax.axvline(
            event.xdata, color=PRELIMINARY_EVENT_COLOR, linestyle=":", linewidth=1.5, animated=True
        )
        self.canvas.draw()
        self._blit_bg = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self._temp_line)
        self.canvas.blit(self.ax.bbox)

    def on_motion(self, event):
        """Handle mouse motion on the plot."""
        # Check if dragging a manually flagged event
        if hasattr(self, '_dragging_manual_flag') and self._dragging_manual_flag:
            if event.inaxes == self.ax and event.xdata is not None:
                self._update_dragged_manual_flag(event.xdata)
            return

        # Normal flag dragging
        if not self._dragging or event.inaxes != self.ax or event.xdata is None:
            return
        self._temp_line.set_xdata([event.xdata, event.xdata])
        self.canvas.restore_region(self._blit_bg)
        self.ax.draw_artist(self._temp_line)
        self.canvas.blit(self.ax.bbox)

    def on_release(self, event):
        """Handle mouse release on the plot."""
        # Check if finishing manual flag drag
        if hasattr(self, '_dragging_manual_flag') and self._dragging_manual_flag:
            release_x = event.xdata if (event.inaxes == self.ax and event.xdata is not None) else None
            self._finish_drag_manually_flagged(release_x)
            return

        if not self._dragging:
            return
        self._dragging = False
        release_x = (
            event.xdata
            if (event.inaxes == self.ax and event.xdata is not None)
            else self._temp_line.get_xdata()[0]
        )

        self._temp_line.set_animated(False)
        self._temp_line.remove()
        self._temp_line = None
        self._blit_bg = None

        snapped_t = signal_processing.snap_to_nearest_peak(
            release_x, self.t, self.filtered
        )
        self._add_manually_flagged(snapped_t)
        self._push_undo_state()
        self.canvas.draw_idle()

    def on_scroll(self, event):
        """Handle scroll/zoom on the plot."""
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
        self._update_sliders()

    def on_xlim_changed(self, ax):
        """Enforce minimum zoom span."""
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
        self._update_scrollbars()

    def on_x_slider(self, value):
        """Handle X slider movement - pan left/right."""
        if self._full_xlim is None or self.state not in (STATE_SEIZURE_MARKING, STATE_FLAGGING):
            return

        xlim = self.ax.get_xlim()
        span = xlim[1] - xlim[0]
        full_span = self._full_xlim[1] - self._full_xlim[0]

        # Map slider position (0-1000) to plot position
        fraction = value / 1000.0
        new_left = self._full_xlim[0] + fraction * (full_span - span)
        new_right = new_left + span

        self.ax.set_xlim(new_left, new_right)
        self.canvas.draw_idle()

    def on_y_slider(self, value):
        """Handle Y slider movement - scale around center."""
        if self._full_ycenter is None or self.state not in (STATE_SEIZURE_MARKING, STATE_FLAGGING):
            return

        # Map slider position (0-1000) to zoom level
        # 500 = 1x zoom (full view)
        # Lower = more zoom (tighter view)
        # Higher = less zoom (wider view, showing more noise)
        fraction = value / 500.0  # 0 to 2
        if fraction < 0.1:
            fraction = 0.1  # Min zoom

        # Full span
        full_span = self._full_ylim[1] - self._full_ylim[0]
        half_span = full_span / (2 * fraction)

        new_bottom = self._full_ycenter - half_span
        new_top = self._full_ycenter + half_span

        self.ax.set_ylim(new_bottom, new_top)
        self.canvas.draw_idle()

    def _update_sliders(self):
        """Update slider positions based on current plot limits."""
        if self._full_xlim is None or self._full_ylim is None:
            return

        # Update X slider
        xlim = self.ax.get_xlim()
        x_span = xlim[1] - xlim[0]
        x_full_span = self._full_xlim[1] - self._full_xlim[0]
        if x_full_span > x_span:
            x_fraction = (xlim[0] - self._full_xlim[0]) / (x_full_span - x_span)
            self.x_slider.blockSignals(True)
            self.x_slider.setValue(int(x_fraction * 1000))
            self.x_slider.blockSignals(False)

        # Update Y slider
        ylim = self.ax.get_ylim()
        y_span = ylim[1] - ylim[0]
        y_full_span = self._full_ylim[1] - self._full_ylim[0]
        if y_full_span > y_span:
            fraction = (y_full_span / (2 * y_span))
            self.y_slider.blockSignals(True)
            self.y_slider.setValue(int(fraction * 500))
            self.y_slider.blockSignals(False)

    def _confirm_and_finish_seizure(self):
        """Confirm and finish seizure marking."""
        reply = QMessageBox.question(
            self, "Confirm seizure onset",
            f"Mark seizure onset at t = {self._seizure_click_time:.3f} s?\n\n"
            "Only data before this point will be used for IED labeling.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            self.seizure_present = True
            self.seizure_onset_s = self._seizure_click_time
            self._seizure_click_pending = False
            # Remove seizure marker line
            if self._temp_line is not None:
                self._temp_line.remove()
                self._temp_line = None
            self._finish_seizure_marking()

    def _finish_seizure_marking(self):
        """Finish seizure marking and enter flagging stage."""
        t_full = self.recording.t
        if self.seizure_present:
            end_idx = int(np.searchsorted(t_full, self.seizure_onset_s, side="left"))
        else:
            end_idx = len(t_full)

        self.t = t_full[:end_idx]
        self.raw = self.recording.raw[:end_idx]
        self.filtered = self.filtered_full[:end_idx]

        if self.seizure_present:
            self.algo_events = self.algo_events_all[
                self.algo_events_all < self.seizure_onset_s
            ]
        else:
            self.algo_events = self.algo_events_all

        # Auto-detect preliminary events with stringent parameters
        min_height = 5.0 * np.std(self.filtered)  # 5 SD
        min_distance_s = 0.200  # 200ms

        self.preliminary_events = signal_processing.find_preliminary_events(
            self.t, self.filtered, self.recording.fs,
            min_height=min_height,
            min_distance_s=min_distance_s
        )

        self._enter_flagging()

    # ======================================================== FLAGGING/VALIDATION STAGE

    def _enter_flagging(self):
        """Enter the flagging and validation stage."""
        self.state = STATE_FLAGGING
        # State tracking (persistent across pool switches)
        self.validated_events = set()  # Times of validated events
        self.rejected_events = set()   # Times of rejected events
        self.manually_flagged_events = set()  # Track manually flagged event times

        # Line object storage (cleared when redrawing pool view)
        self.preliminary_flags = {}  # Line objects for preliminary candidates currently shown
        self.validated_flags = {}    # Line objects for validated events currently shown
        self.rejected_flags_lines = {}  # Line objects for rejected events currently shown

        self.undo_stack = []
        self._seizure_click_pending = False
        self.selected_event = None
        self.pool_view = "candidates"

        # Set up auto-save timer (save every 30 seconds)
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self._auto_save_state)
        self.auto_save_timer.start(30000)  # 30 seconds

        # Try to load previous state from JSON sidecar
        self._load_previous_state()

        instructions = (
            "STEP 2: EVENT VALIDATION\n\n"
            "Your task: Review and validate IED events\n\n"
            "Click H to see a list of hotkeys you might find useful during validation.\n\n"
        )

        dlg = InstructionDialog(self, "Event Validation Instructions", instructions)
        dlg.exec_()

        self.no_seizure_btn.setVisible(False)
        self.seizure_mark_btn.setVisible(False)
        self.flag_btn.setVisible(True)
        self.validate_btn.setVisible(True)
        self.reject_btn.setVisible(True)
        self.reset_btn.setVisible(True)
        self.undo_btn.setVisible(True)
        self.done_btn.setVisible(True)
        self.toggle_filtered_btn.setVisible(True)
        self.prev_candidate_btn.setVisible(True)
        self.next_candidate_btn.setVisible(True)
        self.candidate_counter.setVisible(True)
        self.pool_candidates_btn.setVisible(True)
        self.pool_accepted_btn.setVisible(True)
        self.pool_rejected_btn.setVisible(True)

        # Reset filtered signal toggle
        self._show_filtered = False
        self.toggle_filtered_btn.setChecked(False)

        # Draw the plot
        self.ax.clear()
        self.ax.plot(self.t, self.raw, linewidth=0.6, color="#4c72b0", label="Raw")

        # Store filtered line reference but only show if toggled
        self.filtered_line = self.ax.plot(
            self.t, self.filtered, linewidth=0.6, color="#c44e52", alpha=0.8,
            label="Filtered (0 - 200 Hz)"
        )[0]
        self.filtered_line.set_visible(self._show_filtered)

        # Plot algorithm events
        for event_time in self.algo_events:
            self.ax.axvline(event_time, color=ALGORITHM_EVENT_COLOR, linestyle="--", linewidth=0.8, alpha=0.6)

        # Build candidate list and store in preliminary_flags (for tracking)
        self._candidate_list = sorted(self.preliminary_events)
        self._current_candidate_idx = 0
        for event_time in self.preliminary_events:
            self.preliminary_flags[event_time] = None  # Placeholder - actual lines created in _redraw_pool_view

        if len(self.t) > 0:
            self._full_xlim = (self.t[0], self.t[-1])
        self._full_ylim = self.ax.get_ylim()
        self._full_ycenter = (self._full_ylim[0] + self._full_ylim[1]) / 2

        # Draw the pool view (candidates by default)
        self._redraw_pool_view()
        self._update_candidate_counter()
        self._update_pool_button_styles()
        self._update_button_states()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if self.state == STATE_FLAGGING:
            if event.key() == Qt.Key_V:
                # Validate current candidate (only in candidates pool)
                if self.pool_view == "candidates" and self._candidate_list and self._current_candidate_idx < len(self._candidate_list):
                    event_time = self._candidate_list[self._current_candidate_idx]
                    self._validate_preliminary_event(event_time)
                    self.on_next_candidate()
            elif event.key() == Qt.Key_R:
                # Reject current candidate (only in candidates pool)
                if self.pool_view == "candidates" and self._candidate_list and self._current_candidate_idx < len(self._candidate_list):
                    event_time = self._candidate_list[self._current_candidate_idx]
                    self._reject_preliminary_event(event_time)
                    self.on_next_candidate()
            elif event.key() == Qt.Key_Up or event.key() == Qt.Key_Down:
                # Navigate between candidates
                if event.key() == Qt.Key_Down:
                    self.on_next_candidate()
                else:
                    self.on_prev_candidate()
            elif event.key() == Qt.Key_Right:
                # Pan right
                self._pan_x(0.1)
            elif event.key() == Qt.Key_Left:
                # Pan left
                self._pan_x(-0.1)
            elif event.key() == Qt.Key_U:
                self.on_undo()
            elif event.key() == Qt.Key_W:
                # Withdraw (undo) validation/rejection in accepted/rejected pools
                if self.pool_view == "accepted":
                    self.on_validate_selected()
                elif self.pool_view == "rejected":
                    self.on_reject_selected()
            elif event.key() == Qt.Key_F:
                self.flag_btn.setChecked(not self.flag_btn.isChecked())
            elif event.key() == Qt.Key_D:
                self.on_done_clicked()
            elif event.key() == Qt.Key_1:
                # Switch to candidates pool
                self.on_pool_view_change("candidates")
            elif event.key() == Qt.Key_2:
                # Switch to accepted pool
                self.on_pool_view_change("accepted")
            elif event.key() == Qt.Key_3:
                # Switch to rejected pool
                self.on_pool_view_change("rejected")
            elif event.key() == Qt.Key_A:
                # Zoom fit all
                self._zoom_fit_all()
            elif event.key() == Qt.Key_S:
                # Zoom to single event (±500ms)
                if self._candidate_list and self._current_candidate_idx < len(self._candidate_list):
                    self._zoom_to_event(self._candidate_list[self._current_candidate_idx])
            elif event.key() == Qt.Key_T:
                # Toggle filtered signal
                self.toggle_filtered_btn.setChecked(not self.toggle_filtered_btn.isChecked())
            elif event.key() == Qt.Key_H:
                # Show keyboard shortcuts help
                self._show_keyboard_help()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def on_flag_mode_toggled(self, checked):
        """Update flag mode button style."""
        if checked:
            self.flag_btn.setStyleSheet("background-color: #ffe6e6;")
            # Disable scrollbars when in flag mode
            self.x_slider.setEnabled(False)
            self.y_slider.setEnabled(False)
        else:
            self.flag_btn.setStyleSheet("")
            # Re-enable scrollbars
            self.x_slider.setEnabled(True)
            self.y_slider.setEnabled(True)

    def on_reset_perspective(self):
        """Reset plot to full view."""
        if self._full_xlim is not None:
            self.ax.set_xlim(*self._full_xlim)
        if self._full_ylim is not None:
            self.ax.set_ylim(*self._full_ylim)
        self.canvas.draw_idle()
        self._update_sliders()

    def on_validate_selected(self):
        """Validate the selected event (or undo validation in accepted pool)."""
        if self.pool_view == "accepted":
            # In accepted pool, this button undoes validation
            if self.selected_event is None:
                QMessageBox.warning(self, "No Selection", "Click on an event to select it first.")
                return
            self._unvalidate_event(self.selected_event)
            self.on_next_candidate()
            self._redraw_pool_view()
        else:
            # In candidates pool, validate only if zoomed to single event
            if not self._can_validate_reject():
                QMessageBox.warning(self, "Zoom Required", "Zoom into a single event before validating.")
                return
            if self.selected_event is None:
                QMessageBox.warning(self, "No Selection", "Click on an event to select it first.")
                return
            self._validate_preliminary_event(self.selected_event)
            self.on_next_candidate()

    def on_reject_selected(self):
        """Reject the selected event (or undo rejection in rejected pool)."""
        if self.pool_view == "rejected":
            # In rejected pool, this button undoes rejection
            if self.selected_event is None:
                QMessageBox.warning(self, "No Selection", "Click on an event to select it first.")
                return
            self._unrejected_event(self.selected_event)
            self.on_next_candidate()
            self._redraw_pool_view()
        else:
            # In candidates pool, reject only if zoomed to single event
            if not self._can_validate_reject():
                QMessageBox.warning(self, "Zoom Required", "Zoom into a single event before rejecting.")
                return
            if self.selected_event is None:
                QMessageBox.warning(self, "No Selection", "Click on an event to select it first.")
                return
            self._reject_preliminary_event(self.selected_event)
            self.on_next_candidate()

    def _validate_preliminary_event(self, event_time):
        """Move a preliminary event to validated."""
        self._push_undo_state()

        # Change line color from orange to green if it exists
        if event_time in self.preliminary_flags:
            line = self.preliminary_flags[event_time]
            line.set_color(VALIDATED_EVENT_COLOR)

        self.rejected_events.discard(event_time)
        self.manually_flagged_events.discard(event_time)

        # Add to validated state
        self.validated_events.add(event_time)

        # Remove from candidate list and update index
        if event_time in self._candidate_list:
            self._candidate_list.remove(event_time)
            if self._current_candidate_idx >= len(self._candidate_list) and len(self._candidate_list) > 0:
                self._current_candidate_idx = len(self._candidate_list) - 1

        self.selected_event = None
        self.canvas.draw_idle()
        self._update_candidate_counter()

    def _reject_preliminary_event(self, event_time):
        """Reject a preliminary event."""
        self._push_undo_state()

        # Change line color from orange to red if it exists
        if event_time in self.preliminary_flags:
            line = self.preliminary_flags[event_time]
            line.set_color(REJECTED_EVENT_COLOR)

        self.manually_flagged_events.discard(event_time)

        # Add to rejected state
        self.rejected_events.add(event_time)

        # Remove from candidate list and update index
        if event_time in self._candidate_list:
            self._candidate_list.remove(event_time)
            if self._current_candidate_idx >= len(self._candidate_list) and len(self._candidate_list) > 0:
                self._current_candidate_idx = len(self._candidate_list) - 1

        self.selected_event = None
        self.canvas.draw_idle()
        self._update_candidate_counter()

    def _add_manually_flagged(self, t_flag):
        """Add a manually flagged event - automatically validated."""
        if t_flag not in self.validated_events and t_flag not in self.rejected_events:
            # Create line in green (validated) since manually flagged events are auto-validated
            line = self.ax.axvline(t_flag, color=VALIDATED_EVENT_COLOR, linestyle="--", linewidth=1.5)
            self.preliminary_flags[t_flag] = line
            self.manually_flagged_events.add(t_flag)
            # Automatically add to validated events
            self.validated_events.add(t_flag)
        self._push_undo_state()

    def _remove_validated_flag(self, t_flag):
        """Remove a validated flag."""
        self._push_undo_state()
        if t_flag in self.validated_flags:
            line = self.validated_flags.pop(t_flag)
            line.remove()
        self.canvas.draw_idle()

    def _find_preliminary_flag_near_pixel(self, pixel_x):
        """Find a preliminary flag near the given pixel."""
        for t_flag in self.preliminary_flags:
            if abs(self._pixel_x(t_flag) - pixel_x) <= LINE_PICK_PIXEL_TOLERANCE:
                return t_flag
        return None

    def _find_validated_flag_near_pixel(self, pixel_x):
        """Find a validated flag near the given pixel."""
        for t_flag in self.validated_flags:
            if abs(self._pixel_x(t_flag) - pixel_x) <= LINE_PICK_PIXEL_TOLERANCE:
                return t_flag
        return None

    def _find_manually_flagged_near_pixel(self, pixel_x):
        """Find a manually flagged event near the given pixel (for dragging)."""
        for t_flag in self.manually_flagged_flags:
            if abs(self._pixel_x(t_flag) - pixel_x) <= LINE_PICK_PIXEL_TOLERANCE * 2:
                return t_flag
        return None

    def _start_drag_manually_flagged(self, event_time, initial_xdata):
        """Start dragging a manually flagged event."""
        self._dragging_manual_flag = True
        self._dragged_flag_time = event_time
        self._drag_start_xdata = initial_xdata
        self.flag_btn.setText("🚩 Drag to Shift")

    def _update_dragged_manual_flag(self, current_xdata):
        """Update manually flagged event position while dragging."""
        if not hasattr(self, '_dragging_manual_flag') or not self._dragging_manual_flag or current_xdata is None:
            return
        if self._dragged_flag_time in self.manually_flagged_flags:
            line = self.manually_flagged_flags[self._dragged_flag_time]
            line.set_xdata([current_xdata, current_xdata])
            self.canvas.draw_idle()

    def _finish_drag_manually_flagged(self, final_xdata):
        """Finish dragging and update manually flagged event position."""
        if not hasattr(self, '_dragging_manual_flag') or not self._dragging_manual_flag:
            self._dragging_manual_flag = False
            self.flag_btn.setText("🚩 Flag Event\n(Manual) [F]")
            return

        if final_xdata is None:
            self._dragging_manual_flag = False
            self.flag_btn.setText("🚩 Flag Event\n(Manual) [F]")
            return

        self._push_undo_state()
        old_time = self._dragged_flag_time
        new_time = final_xdata

        # Clamp to data range
        if new_time < self.t[0]:
            new_time = self.t[0]
        elif new_time > self.t[-1]:
            new_time = self.t[-1]

        # Move event to new location
        if old_time in self.manually_flagged_flags:
            line = self.manually_flagged_flags.pop(old_time)
            line.set_xdata([new_time, new_time])
            self.manually_flagged_flags[new_time] = line

        self._dragging_manual_flag = False
        self.flag_btn.setText("🚩 Flag Event\n(Manual) [F]")
        self.canvas.draw_idle()

    def _pixel_x(self, data_x):
        """Convert data x-coordinate to pixel x-coordinate."""
        return self.ax.transData.transform((data_x, 0))[0]

    def _zoom_to_event(self, event_time):
        """Zoom to ±500ms around an event."""
        new_left = event_time - EVENT_CONTEXT_SECONDS
        new_right = event_time + EVENT_CONTEXT_SECONDS

        # Clamp to data limits
        if new_left < self._full_xlim[0]:
            new_left = self._full_xlim[0]
        if new_right > self._full_xlim[1]:
            new_right = self._full_xlim[1]

        self.ax.set_xlim(new_left, new_right)
        self.canvas.draw_idle()
        self._update_sliders()

    def _pan_x(self, fraction):
        """Pan the x-axis by a fraction of current view width."""
        xlim = self.ax.get_xlim()
        span = xlim[1] - xlim[0]
        pan_amount = span * fraction

        new_left = xlim[0] + pan_amount
        new_right = xlim[1] + pan_amount

        # Clamp to data limits
        if new_left < self._full_xlim[0]:
            new_left = self._full_xlim[0]
            new_right = new_left + span
        if new_right > self._full_xlim[1]:
            new_right = self._full_xlim[1]
            new_left = new_right - span

        self.ax.set_xlim(new_left, new_right)
        self.canvas.draw_idle()
        self._update_sliders()

    def _can_validate_reject(self):
        """Check if current view allows validation/rejection (single event visible)."""
        if self.pool_view != "candidates":
            return False

        xlim = self.ax.get_xlim()
        # Count events visible in current x range
        visible_candidates = 0
        for event_time in self._get_candidates():
            if xlim[0] <= event_time <= xlim[1]:
                visible_candidates += 1
                if visible_candidates > 1:
                    return False  # More than one visible, can't validate

        return visible_candidates == 1

    def _zoom_fit_all(self):
        """Zoom to fit entire recording."""
        if self._full_xlim is not None:
            self.ax.set_xlim(*self._full_xlim)
        if self._full_ylim is not None:
            self.ax.set_ylim(*self._full_ylim)
        self.canvas.draw_idle()
        self._update_sliders()

    def _zoom_to_span(self, span_seconds):
        """Zoom to a specific time span centered at current view."""
        xlim = self.ax.get_xlim()
        center = (xlim[0] + xlim[1]) / 2
        new_left = center - span_seconds / 2
        new_right = center + span_seconds / 2

        # Clamp to data limits
        if new_left < self._full_xlim[0]:
            new_left = self._full_xlim[0]
            new_right = new_left + span_seconds
        if new_right > self._full_xlim[1]:
            new_right = self._full_xlim[1]
            new_left = new_right - span_seconds

        self.ax.set_xlim(new_left, new_right)
        self.canvas.draw_idle()
        self._update_sliders()

    def _show_keyboard_help(self):
        """Show keyboard shortcuts help dialog."""
        help_text = """
KEYBOARD SHORTCUTS

Navigation:
  A . . . . . . . . . . . Reset view (fit all)
  S . . . . . . . . . . . Zoom Single Event (±500ms)
  T . . . . . . . . . . . Toggle Filtered Signal

Event Actions (Candidates Pool):
  V . . . . . . . . . . . Validate selected event
  R . . . . . . . . . . . Reject selected event
  U . . . . . . . . . . . Undo last action
  F . . . . . . . . . . . Toggle Flag (manual event mode)

Undo Validation/Rejection (Accepted/Rejected Pools):
  W . . . . . . . . . . . Withdraw (undo) validation/rejection

Other:
  D . . . . . . . . . . . Done & Analyze

Pool Switching:
  1 . . . . . . . . . . . View Candidates
  2 . . . . . . . . . . . View Accepted
  3 . . . . . . . . . . . View Rejected

Help:
  H . . . . . . . . . . . Show this help
        """
        dlg = InstructionDialog(self, "Keyboard Shortcuts", help_text)
        dlg.exec_()

    def _auto_save_state(self):
        """Auto-save current validation state to JSON sidecar file."""
        if self.recording is None:
            return

        # Create JSON sidecar filename (same name as ABF with .validation.json)
        sidecar_path = Path(self.recording.file_stem).parent / f"{self.recording.file_stem}.validation.json"

        # If recording path is just a filename, save in current directory
        if not sidecar_path.parent.exists() or sidecar_path.parent == Path():
            sidecar_path = Path(self.recording.file_stem + ".validation.json")

        state = {
            "file_stem": self.recording.file_stem,
            "validated_events": sorted(list(self.validated_events)),
            "rejected_events": sorted(list(self.rejected_events)),
            "manually_flagged_events": sorted(list(self.manually_flagged_events)),
        }

        try:
            with open(sidecar_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception:
            # Silently fail on auto-save errors
            pass

    def _load_previous_state(self):
        """Load previous validation state from JSON sidecar if it exists."""
        if self.recording is None:
            return

        # Look for JSON sidecar
        sidecar_path = Path(self.recording.file_stem + ".validation.json")

        if not sidecar_path.exists():
            return

        try:
            with open(sidecar_path, 'r') as f:
                state = json.load(f)

            # Restore state
            self.validated_events = set(state.get("validated_events", []))
            self.rejected_events = set(state.get("rejected_events", []))
            self.manually_flagged_events = set(state.get("manually_flagged_events", []))
        except Exception:
            # Silently fail on load errors
            pass

    def on_undo(self):
        """Undo the last action."""
        if not self.undo_stack:
            return

        undo_state = self.undo_stack.pop()

        # Restore event state from undo
        self.validated_events = undo_state["validated"].copy()
        self.rejected_events = undo_state["rejected"].copy()
        self.manually_flagged_events = undo_state["manually_flagged"].copy()

        # Update display
        self._update_candidate_list_for_pool(self.pool_view)
        self._redraw_pool_view()
        self.selected_event = None

    def _push_undo_state(self):
        """Save current state to undo stack."""
        state = {
            "validated": self.validated_events.copy(),
            "rejected": self.rejected_events.copy(),
            "manually_flagged": self.manually_flagged_events.copy(),
        }
        self.undo_stack.append(state)
        # Keep only last 20 actions
        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)

    def on_pool_view_change(self, pool_type):
        """Change the pool view (candidates, accepted, or rejected)."""
        self.pool_view = pool_type
        self.selected_event = None
        self.flag_btn.setChecked(False)  # Disable flag mode when switching pools

        # Update button states
        self.pool_candidates_btn.setChecked(pool_type == "candidates")
        self.pool_accepted_btn.setChecked(pool_type == "accepted")
        self.pool_rejected_btn.setChecked(pool_type == "rejected")

        # Update visual styling to highlight active pool
        self._update_pool_button_styles()
        self._update_button_states()

        # Update button visibility based on pool
        if pool_type == "candidates":
            # Full functionality for candidates pool
            self.flag_btn.setVisible(True)
            self.validate_btn.setText("Validate\nSelected [V]")
            self.validate_btn.setVisible(True)
            self.reject_btn.setText("Reject\nSelected [R]")
            self.reject_btn.setVisible(True)
            self.prev_candidate_btn.setVisible(True)
            self.next_candidate_btn.setVisible(True)
            self.candidate_counter.setVisible(True)
            self.x_slider.setEnabled(True)
            self.y_slider.setEnabled(True)
            self.toggle_filtered_btn.setVisible(True)
            self._update_candidate_list_for_pool("candidates")
        elif pool_type == "accepted":
            # Review mode for accepted events
            self.flag_btn.setChecked(False)  # Ensure flag mode is off
            self.flag_btn.setVisible(False)
            self.validate_btn.setText("Undo\nValidation [W]")
            self.validate_btn.setVisible(True)
            self.reject_btn.setVisible(False)
            self.prev_candidate_btn.setVisible(True)
            self.next_candidate_btn.setVisible(True)
            self.candidate_counter.setVisible(True)
            self.x_slider.setEnabled(False)
            self.y_slider.setEnabled(False)
            self.toggle_filtered_btn.setVisible(True)
            self._update_candidate_list_for_pool("accepted")
        elif pool_type == "rejected":
            # Review mode for rejected events
            self.flag_btn.setChecked(False)  # Ensure flag mode is off
            self.flag_btn.setVisible(False)
            self.validate_btn.setVisible(False)
            self.reject_btn.setText("Undo\nRejection [W]")
            self.reject_btn.setVisible(True)
            self.prev_candidate_btn.setVisible(True)
            self.next_candidate_btn.setVisible(True)
            self.candidate_counter.setVisible(True)
            self.x_slider.setEnabled(False)
            self.y_slider.setEnabled(False)
            self.toggle_filtered_btn.setVisible(True)
            self._update_candidate_list_for_pool("rejected")

        # Redraw plot with appropriate events and zoom to first event
        self._redraw_pool_view()

        # Zoom to first event in the new pool (or reset to full view if empty)
        if self._candidate_list:
            self._zoom_to_event(self._candidate_list[0])
        else:
            # Reset to full view if pool is empty
            if self._full_xlim is not None:
                self.ax.set_xlim(*self._full_xlim)
            if self._full_ylim is not None:
                self.ax.set_ylim(*self._full_ylim)
            self.canvas.draw_idle()
            self._update_sliders()

    def _update_candidate_list_for_pool(self, pool_type):
        """Update the candidate list for the given pool."""
        if pool_type == "candidates":
            self._candidate_list = sorted(self._get_candidates())
        elif pool_type == "accepted":
            self._candidate_list = sorted(self.validated_events)
        elif pool_type == "rejected":
            self._candidate_list = sorted(self.rejected_events)
        self._current_candidate_idx = 0
        self._update_candidate_counter()

    def _update_pool_button_styles(self):
        """Update pool button visual styling to show active pool."""
        active_style = "font-weight: bold; border: 2px solid #0078d4; background-color: #e6f0ff;"
        inactive_style = ""

        if self.pool_view == "candidates":
            self.pool_candidates_btn.setStyleSheet(active_style)
            self.pool_accepted_btn.setStyleSheet(inactive_style)
            self.pool_rejected_btn.setStyleSheet(inactive_style)
        elif self.pool_view == "accepted":
            self.pool_candidates_btn.setStyleSheet(inactive_style)
            self.pool_accepted_btn.setStyleSheet(active_style)
            self.pool_rejected_btn.setStyleSheet(inactive_style)
        elif self.pool_view == "rejected":
            self.pool_candidates_btn.setStyleSheet(inactive_style)
            self.pool_accepted_btn.setStyleSheet(inactive_style)
            self.pool_rejected_btn.setStyleSheet(active_style)

    def _update_button_states(self):
        """Update button enabled/disabled states based on context."""
        # Flag button only enabled in candidates pool
        self.flag_btn.setEnabled(self.pool_view == "candidates")

        # Validate button: enabled in candidates (for validation) and accepted (for undo)
        self.validate_btn.setEnabled(self.pool_view in ("candidates", "accepted"))

        # Reject button: enabled in candidates (for rejection) and rejected (for undo)
        self.reject_btn.setEnabled(self.pool_view in ("candidates", "rejected"))

        # Pool buttons: disable the current pool button
        self.pool_candidates_btn.setEnabled(self.pool_view != "candidates")
        self.pool_accepted_btn.setEnabled(self.pool_view != "accepted")
        self.pool_rejected_btn.setEnabled(self.pool_view != "rejected")

    def _redraw_pool_view(self):
        """Redraw event lines for current pool view."""
        # Remove old event lines (keep signal lines)
        lines_to_remove = []
        for line in self.ax.get_lines():
            xdata = line.get_xdata()
            # Event marker lines have exactly 2 identical x values (vertical line)
            if len(xdata) == 2 and xdata[0] == xdata[1]:
                lines_to_remove.append(line)

        for line in lines_to_remove:
            line.remove()

        # Clear line object dictionaries (but NOT event state sets)
        self.preliminary_flags = {}
        self.validated_flags = {}
        self.rejected_flags_lines = {}

        # Show events based on current pool
        if self.pool_view == "candidates":
            # In candidates pool, show ALL events (candidates + validated + rejected)
            # This allows seeing validated/rejected lines while navigating candidates
            all_events = set()

            # Add candidates (orange)
            for event_time in sorted(self._get_candidates()):
                line = self.ax.axvline(event_time, color=PRELIMINARY_EVENT_COLOR, linestyle="--", linewidth=1.5)
                self.preliminary_flags[event_time] = line
                all_events.add(event_time)

            # Add validated (green) - keep their lines visible
            for event_time in sorted(self.validated_events):
                line = self.ax.axvline(event_time, color=VALIDATED_EVENT_COLOR, linestyle="--", linewidth=1.5)
                self.preliminary_flags[event_time] = line
                all_events.add(event_time)

            # Add rejected (red) - keep their lines visible
            for event_time in sorted(self.rejected_events):
                line = self.ax.axvline(event_time, color=REJECTED_EVENT_COLOR, linestyle="--", linewidth=1.5)
                self.preliminary_flags[event_time] = line
                all_events.add(event_time)

        elif self.pool_view == "accepted":
            # Show ONLY validated events (green)
            for event_time in sorted(self.validated_events):
                line = self.ax.axvline(event_time, color=VALIDATED_EVENT_COLOR, linestyle="--", linewidth=1.5)
                self.validated_flags[event_time] = line

        elif self.pool_view == "rejected":
            # Show ONLY rejected events (red)
            for event_time in sorted(self.rejected_events):
                line = self.ax.axvline(event_time, color=REJECTED_EVENT_COLOR, linestyle="--", linewidth=1.5)
                self.rejected_flags_lines[event_time] = line

        # Autoscale y-axis to fit the data in the current view
        self.ax.autoscale(axis='y')

        self.canvas.draw_idle()
        if hasattr(self, '_update_sliders'):
            self._update_sliders()

    def _get_candidates(self):
        """Get all candidate events (not validated or rejected)."""
        # Include both auto-detected and manually flagged events
        auto_detected = set(self.preliminary_events) if self.preliminary_events is not None else set()
        manually_flagged = self.manually_flagged_events.copy()
        all_candidates = auto_detected | manually_flagged
        return all_candidates - self.validated_events - self.rejected_events

    def _get_validated(self):
        """Get all validated events."""
        return self.validated_events.copy()

    def _select_event(self, event_time, event_type):
        """Select an event for validation/rejection."""
        self.selected_event = event_time
        self.selected_event_type = event_type

        # Zoom to ±500ms around selected event
        self._zoom_to_event(event_time)

    def _find_rejected_flag_near_pixel(self, pixel_x):
        """Find a rejected event near the given pixel."""
        for t_flag in self.rejected_events:
            if abs(self._pixel_x(t_flag) - pixel_x) <= LINE_PICK_PIXEL_TOLERANCE:
                return t_flag
        return None

    def on_review_validated(self):
        """Review validated events."""
        if not self.validated_events:
            QMessageBox.information(self, "No Events", "No validated events to review.")
            return

        # Get all validated events
        validated_times = sorted(list(self.validated_events))

        if not validated_times:
            return

        # Show review dialog for validated events
        event_idx = self._show_review_dialog("Validated Events", validated_times, "undo_validation")

    def on_review_rejected(self):
        """Review rejected events."""
        if not self.rejected_events:
            QMessageBox.information(self, "No Events", "No rejected events to review.")
            return

        rejected_times = sorted(self.rejected_events)
        if not rejected_times:
            return

        # Show review dialog for rejected events
        event_idx = self._show_review_dialog("Rejected Events", rejected_times, "undo_rejection")

    def _show_review_dialog(self, title, event_times, action_type):
        """Show a dialog for reviewing events with ±500ms context."""
        # Use a mutable container to allow modification in nested functions
        state = {"idx": 0}

        def show_event(idx):
            if idx < 0 or idx >= len(event_times):
                return
            state["idx"] = idx
            event_time = event_times[idx]
            self._zoom_to_event(event_time)
            dlg.setWindowTitle(f"{title} - Event {idx + 1} / {len(event_times)} @ {event_time:.3f}s")

        dlg = QDialog(self)
        dlg.setWindowTitle(f"{title} - Event 1 / {len(event_times)}")
        dlg.setGeometry(100, 100, 300, 200)
        layout = QVBoxLayout(dlg)

        info_label = QLabel(f"Reviewing {len(event_times)} events\nClick buttons to navigate and undo")
        layout.addWidget(info_label)

        # Navigation buttons
        nav_layout = QHBoxLayout()

        def go_prev():
            show_event(max(0, state["idx"] - 1))

        def go_next():
            show_event(min(len(event_times) - 1, state["idx"] + 1))

        prev_btn = QPushButton("Previous")
        prev_btn.clicked.connect(go_prev)
        nav_layout.addWidget(prev_btn)

        next_btn = QPushButton("Next")
        next_btn.clicked.connect(go_next)
        nav_layout.addWidget(next_btn)

        layout.addLayout(nav_layout)

        # Action button
        def undo_current():
            event_time = event_times[state["idx"]]
            if action_type == "undo_validation":
                self._unvalidate_event(event_time)
            elif action_type == "undo_rejection":
                self._unrejected_event(event_time)
            dlg.accept()

        action_btn = QPushButton(f"Undo {'Validation' if action_type == 'undo_validation' else 'Rejection'}")
        action_btn.clicked.connect(undo_current)
        layout.addWidget(action_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.reject)
        layout.addWidget(close_btn)

        show_event(0)
        dlg.exec_()

    def _unvalidate_event(self, event_time):
        """Unvalidate a previously validated event."""
        self._push_undo_state()
        # Remove from validated state
        self.validated_events.discard(event_time)

        # Change line color back to orange if it exists
        if event_time in self.preliminary_flags:
            line = self.preliminary_flags[event_time]
            line.set_color(PRELIMINARY_EVENT_COLOR)
            # Add back to candidate list
            if event_time not in self._candidate_list:
                self._candidate_list.append(event_time)
                self._candidate_list.sort()

        # Update counter and redraw
        self._update_candidate_counter()
        self.canvas.draw_idle()

    def _unrejected_event(self, event_time):
        """Unrejected a previously rejected event."""
        self._push_undo_state()
        # Remove from rejected state
        self.rejected_events.discard(event_time)

        # Change line color back to orange if it exists
        if event_time in self.preliminary_flags:
            line = self.preliminary_flags[event_time]
            line.set_color(PRELIMINARY_EVENT_COLOR)
            # Add back to candidate list
            if event_time not in self._candidate_list:
                self._candidate_list.append(event_time)
                self._candidate_list.sort()

        # Update counter and redraw
        self._update_candidate_counter()
        self.canvas.draw_idle()

    def on_toggle_filtered(self, checked):
        """Toggle filtered signal overlay without changing zoom."""
        self._show_filtered = checked

        # Find and toggle signal lines by color
        for line in self.ax.get_lines():
            color = line.get_color()
            # Raw signal is blue
            if color == "#4c72b0":
                line.set_visible(not self._show_filtered)
            # Filtered signal is red
            elif color == "#c44e52":
                line.set_visible(self._show_filtered)

        # Update button label with keyboard shortcut preserved
        if self._show_filtered:
            self.toggle_filtered_btn.setText("Hide\nFiltered [T]")
        else:
            self.toggle_filtered_btn.setText("Show\nFiltered [T]")

        self.canvas.draw_idle()

    def on_next_candidate(self):
        """Jump to next preliminary candidate."""
        if not self._candidate_list:
            return
        self._current_candidate_idx = (self._current_candidate_idx + 1) % len(self._candidate_list)
        event_time = self._candidate_list[self._current_candidate_idx]
        self._zoom_to_event(event_time)
        self._highlight_current_candidate()
        self._update_candidate_counter()

    def on_prev_candidate(self):
        """Jump to previous preliminary candidate."""
        if not self._candidate_list:
            return
        self._current_candidate_idx = (self._current_candidate_idx - 1) % len(self._candidate_list)
        event_time = self._candidate_list[self._current_candidate_idx]
        self._zoom_to_event(event_time)
        self._highlight_current_candidate()
        self._update_candidate_counter()

    def _highlight_current_candidate(self):
        """Highlight the current candidate line visually."""
        if not self._candidate_list or self._current_candidate_idx >= len(self._candidate_list):
            return

        current_time = self._candidate_list[self._current_candidate_idx]

        # Update all preliminary event line widths
        for event_time, line in self.preliminary_flags.items():
            if event_time == current_time:
                line.set_linewidth(2.5)  # Thicker for current
                line.set_alpha(1.0)      # Full opacity
            else:
                line.set_linewidth(1.5)  # Normal for others
                line.set_alpha(0.7)      # Slightly faded

        self.canvas.draw_idle()

    def _update_candidate_counter(self):
        """Update the candidate counter display."""
        if not self._candidate_list:
            self.candidate_counter.setText("-- / --")
        else:
            self.candidate_counter.setText(f"{self._current_candidate_idx + 1} / {len(self._candidate_list)}")

    def on_done_clicked(self):
        """Done with flagging, run analysis."""
        total_validated = len(self.validated_events)
        reply = QMessageBox.question(
            self, "Confirm done",
            f"You have validated {total_validated} event(s). "
            "Proceed to analysis? No further edits will be possible after this.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return
        self._run_analysis()

    # ============================================================== ANALYSIS & EXPORT

    def _run_analysis(self):
        """Run analysis and show results."""
        self.state = STATE_DONE
        self.flag_btn.setVisible(False)
        self.validate_btn.setVisible(False)
        self.reject_btn.setVisible(False)
        self.reset_btn.setVisible(False)
        self.undo_btn.setVisible(False)
        self.done_btn.setVisible(False)

        gt_times = sorted(self.validated_flags.keys())
        result = scoring.score_events(
            gt_times, list(self.algo_events), tolerance_s=scoring.MATCH_TOLERANCE_S
        )

        sens_str = f"{result.sensitivity:.3f}" if result.sensitivity is not None else "N/A"

        QMessageBox.information(
            self, "Analysis Results",
            f"Ground truth events: {len(gt_times)}\n"
            f"Algorithm events (pre-seizure): {len(self.algo_events)}\n\n"
            f"TP: {result.tp}\nFP: {result.fp}\nFN: {result.fn}\n"
            f"Sensitivity: {sens_str}",
        )

        self._save_to_workbook(result, gt_times)

    def _save_to_workbook(self, result, gt_times):
        """Save results to Excel workbook."""
        dlg = QFileDialog(
            self, "Select or create master workbook", "", "Excel Workbook (*.xlsx)"
        )
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

        QMessageBox.information(
            self, "Saved",
            f"Results saved to:\n{path}\n\nSheet: {workbook.sheet_name_for(file_result)}",
        )
