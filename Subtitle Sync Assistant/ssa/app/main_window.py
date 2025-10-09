# SPDX-License-Identifier: GPL-3.0-or-later
# Subtitle Sync Assistant
# This file is part of Subtitle Sync Assistant.
#
# Copyright (C) 2025 Subtitle Sync Assistant contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# -*- coding: utf-8 -*-
"""Main application window extracted from the monolith.

Wires together UI widgets,dialogs and background workers. Logic preserved.
Some helper code deduplicated via utils and services modules.

Responsibilities:
- File selection (media + subtitle) and output target
- Spawning analysis worker and presenting results (plots + tables)
- Lightweight waveform playback for selected subtitle rows
- Manual time shift/edit operations and export to SRT
- Optional audio-offset-finder integration for finding per-row offsets
"""
from __future__ import annotations

import os
from typing import List, Tuple, Optional

import numpy as np
import pysrt
from pydub import AudioSegment

from PyQt5.QtCore import Qt, QByteArray, QBuffer, QThread, pyqtSignal, QTimer  # Added QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtMultimedia import QAudioFormat, QAudioOutput
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QSizePolicy,
    QFrame,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFileDialog,
    QMessageBox,
    QSlider,
    QAbstractItemView,
    QMenu,
    QAction,
    QInputDialog,
)

from ..ui.plot_widget import MatplotlibPlotWidget
from ..ui.dialogs import BusyDialog, RangeSelectDialog, EditSubtitleDialog
from ..ui.preview import PreviewImagesWindow
from ..workers.analyze import AnalyzeWorker
from ..workers.offset import OffsetWorker, SlidingOffsetWorker
from ..services.audio import segment_to_float_array
from ..utils.timecodes import format_seconds

try:
    from audio_offset_finder.audio_offset_finder import find_offset_between_buffers  # noqa: F401
except Exception:
    find_offset_between_buffers = None


class MainWindow(QWidget):
    """Main window integrating plots, tables, and workflow controls."""

    def __init__(self):
        """Initialize the UI, state caches and threads/workers placeholders."""
        super().__init__()
        self.setWindowTitle("Subtitle Sync Assistant")
        self.resize(1200, 600)
        self._build_ui()

        # Audio playback shared state for table-row preview
        self.ref_audio_segment: Optional[AudioSegment] = None
        self.new_audio_segment: Optional[AudioSegment] = None
        self.audio_output: Optional[QAudioOutput] = None
        self.audio_buffer: Optional[QBuffer] = None
        self.audio_data: Optional[QByteArray] = None

        # Analysis worker/thread state
        self._analyze_thread: Optional[QThread] = None
        self._analyze_worker: Optional[AnalyzeWorker] = None
        self._busy_dialog: Optional[BusyDialog] = None

        # Offset worker/thread + audio caches for offset computations
        self._offset_thread: Optional[QThread] = None
        self._offset_worker: Optional[OffsetWorker] = None
        self._ref_mono_cache: Optional[np.ndarray] = None
        self._ref_sr_cache: Optional[int] = None
        self._new_mono_cache: Optional[np.ndarray] = None
        self._new_sr_cache: Optional[int] = None
        self._busy_offset: Optional[BusyDialog] = None
        self._shift_sel_color = QColor(255, 225, 160)  # selected rows shifted
        self._shift_all_color = QColor(255, 240, 200)  # all rows shifted

        # Preview window (created lazily)
        self._preview_win: Optional[PreviewImagesWindow] = None

        # --- Manual edit handling for start/end times -> waveform update ---
        # Suppress flag prevents expensive refresh spam during bulk ops
        self._suppress_item_changed: bool = False
        # Debounce timer groups rapid edits into a single interval refresh
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._debounced_interval_refresh)
        # Connect after tables exist
        self.referencetable.itemChanged.connect(self._on_table_item_changed)
        self.synctable.itemChanged.connect(self._on_table_item_changed)

    # ---------- UI ----------
    def _build_ui(self):
        """Create the full UI layout, plots, and tables."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(4)

        def add_row(btn_text, slot):
            """Helper to add a file row with a button and read-only line edit."""
            h = QHBoxLayout()
            btn = QPushButton(btn_text); btn.setFixedWidth(220)
            le = QLineEdit(); le.setReadOnly(True); le.setPlaceholderText("No file selected")
            h.addWidget(btn); h.addWidget(le); main_layout.addLayout(h)
            btn.clicked.connect(slot)
            return btn, le

        # File in/out rows
        self.btn1, self.le1 = add_row("Load reference media file", self.select_media_file_btn1)
        self.btn2, self.le2 = add_row("Load new media file", self.select_media_file_btn2)
        self.btn3, self.le3 = add_row("Load reference subtitle", self.select_subtitle_file_btn3)
        self.btn4, self.le4 = add_row("Save subtitle under...", self.save_subtitle_file_btn4)

        # Analyze button row
        row5 = QHBoxLayout(); self.btn5 = QPushButton("Analyze..."); row5.addWidget(self.btn5); main_layout.addLayout(row5)
        self.btn5.clicked.connect(self.on_analyze)

        # Separator
        main_layout.addSpacing(8)
        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setFrameShadow(QFrame.Sunken); sep.setLineWidth(1)
        main_layout.addWidget(sep)
        main_layout.addSpacing(8)

        # Two plots stacked vertically; each manages its own playback state
        self.plot1 = MatplotlibPlotWidget("Reference Audio Waveform")
        self.plot2 = MatplotlibPlotWidget("New Audio Waveform")
        # Ensure only one plot plays at a time
        self.plot1.playingChanged.connect(lambda on: on and self.plot2.stop_playback_external())
        self.plot2.playingChanged.connect(lambda on: on and self.plot1.stop_playback_external())
        main_layout.addWidget(self.plot1)
        main_layout.addWidget(self.plot2)
        main_layout.addSpacing(8)

        # Tables (reference vs sync target)
        tables_row = QHBoxLayout()
        self.referencetable = QTableWidget(0, 3)
        self.referencetable.setHorizontalHeaderLabels(["Start time", "End time", "Text"])
        self._init_table_column_sizing(self.referencetable, [0, 1], [2])
        self.referencetable.setMinimumHeight(300)
        self.referencetable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.referencetable.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.referencetable.setContextMenuPolicy(Qt.CustomContextMenu)
        self.referencetable.customContextMenuRequested.connect(self.show_referencetable_context_menu)

        # Sync table (right)
        self.synctable = QTableWidget(0, 6)  # Changed from 5 to 6 columns
        self.synctable.setHorizontalHeaderLabels(["Start time", "End time", "Text", "Found offset", "Score", "Total shift"])  # Added "Score"
        self._init_table_column_sizing(self.synctable, [0, 1, 3, 4, 5], [2])  # Updated column indices
        # Header tooltip for "Total shift"
        hdrTotalshift = self.synctable.horizontalHeaderItem(5)
        if hdrTotalshift:
            hdrTotalshift.setToolTip(
                "Cumulative time shift that was applied to a line."
            )
        hdrFoundOffset = self.synctable.horizontalHeaderItem(3)
        if hdrFoundOffset:
            hdrFoundOffset.setToolTip(
                "Offset that was found when searching for a line in the new media. "
                "BBC-offset-finder will always find a match, which may be incorrect."
            )
        self.synctable.setMinimumHeight(300)
        self.synctable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.synctable.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.synctable.setContextMenuPolicy(Qt.CustomContextMenu)
        self.synctable.customContextMenuRequested.connect(self.show_synctable_context_menu)

        # Add tables to layout with stretch factors (7:10)
        tables_row.addWidget(self.referencetable, 7)
        tables_row.addWidget(self.synctable, 10)
        tables_row.setContentsMargins(0, 0, 0, 0)
        tables_container = QWidget(); tables_container.setLayout(tables_row); tables_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(tables_container)

        # Column alignment and selection -> plot highlight wiring
        self.align_table_columns_left(self.referencetable)
        self.align_table_columns_left(self.synctable)
        self.referencetable.selectionModel().selectionChanged.connect(self.on_reference_table_selection)
        self.synctable.selectionModel().selectionChanged.connect(self.on_sync_table_selection)

        self.referencetable.setSortingEnabled(False)
        self.synctable.setSortingEnabled(False)

    def _init_table_column_sizing(self, table: QTableWidget, fixed_cols: List[int], stretch_cols: List[int]):
        """Apply a mix of fixed and stretch sizing to specific columns."""
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for c in fixed_cols:
            table.horizontalHeader().setSectionResizeMode(c, QHeaderView.Fixed)
            table.setColumnWidth(c, 100)
        for c in stretch_cols:
            table.horizontalHeader().setSectionResizeMode(c, QHeaderView.Stretch)

    def _preview_window(self) -> PreviewImagesWindow:
        """Return the singleton preview window, creating it on first use.

        The preview is created lazily to avoid extra widget/thread overhead
        during startup. It is parented to the main window so it closes with it.
        """
        # Lazily instantiate and reuse a single PreviewImagesWindow instance
        if self._preview_win is None:
            self._preview_win = PreviewImagesWindow(self)
        return self._preview_win

    def _selected_row_start(self, table: QTableWidget) -> Optional[float]:
        """Return start time (seconds) from the first selected row in a table.

        Parameters
        ----------
        table: QTableWidget
            Either the reference or sync table. Column 0 holds the start time.

        Returns
        -------
        Optional[float]
            Parsed seconds value, or None if no selection/invalid cell.
        """
        # Get current row selection; we use the first selected row if any
        sel = table.selectionModel().selectedRows()
        if not sel:
            return None
        # Start time is stored in column 0 as an SRT time string
        itm = table.item(sel[0].row(), 0)
        if not itm:
            return None
        # Reuse helper to parse HH:MM:SS,mmm -> seconds (float)
        return self._parse_time_to_seconds(itm.text())

    def _selected_row_index(self, table: QTableWidget) -> Optional[int]:
        """Return the first selected row index (0-based) or None."""
        sel = table.selectionModel().selectedRows()
        return sel[0].row() if sel else None

    def _update_preview_images_from_selection(self):
        """Open/refresh the preview using the current table selections.

        Collects start times from both tables and the media file paths from
        the line edits, then forwards these to the preview window so it can
        extract and display frames side by side.
        """
        # Read the selected start times (seconds) from both tables
        ref_time = self._selected_row_start(self.referencetable)
        new_time = self._selected_row_start(self.synctable)
        ref_idx = self._selected_row_index(self.referencetable)
        new_idx = self._selected_row_index(self.synctable)
        # Read the media file paths from the top inputs (empty -> None)
        ref_path = self.le1.text().strip() if self.le1.text().strip() else None
        new_path = self.le2.text().strip() if self.le2.text().strip() else None
        # Ask the preview to show/update with the provided selection
        self._preview_window().show_for_selection(
            ref_path, ref_time,
            new_path, new_time,
            ref_line=ref_idx, new_line=new_idx
        )

    def align_table_columns_left(self, table: QTableWidget):
        """Left-align header and cell text for all columns/rows."""
        for col in range(table.columnCount()):
            header_item = table.horizontalHeaderItem(col)
            if header_item:
                header_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            for row in range(table.rowCount()):
                cell = table.item(row, col)
                if cell:
                    cell.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    def _add_preview_to_menu(self, menu: QMenu):
        """Append a 'Show preview images' action to a given context menu.

        The action opens or refreshes the Preview Images window based on the
        current selection in the tables.
        """
        # Create the action and wire it to the update handler
        act_preview = QAction("Show preview images", self)
        act_preview.triggered.connect(self._update_preview_images_from_selection)
        # Add to the provided menu at the current insertion point
        menu.addAction(act_preview)

    # ---------- NEW: Manual time edit handling ----------
    def _on_table_item_changed(self, item: QTableWidgetItem):
        """React to user edits of start/end time cells by refreshing waveform intervals.

        Debounced to avoid performance issues while the user types or during
        multi-cell edits (e.g. paste). Suppressed during bulk programmatic
        updates (analysis load, shifting operations) using _suppress_item_changed.
        """
        if self._suppress_item_changed:
            return
        if not item:
            return
        table = item.tableWidget()
        if table not in (self.referencetable, self.synctable):
            return
        col = item.column()
        if col not in (0, 1):  # Only care about start/end time edits
            return
        row = item.row()
        # Sanitize the edited row's time values (keep consistent formatting)
        start_item = table.item(row, 0)
        end_item = table.item(row, 1)
        if not start_item or not end_item:
            return
        try:
            s = self._parse_time_to_seconds(start_item.text())
            e = self._parse_time_to_seconds(end_item.text())
            if e < s:
                e = s  # Clamp end to start if reversed
            # Reformat (avoid recursion by suppressing temporarily)
            self._suppress_item_changed = True
            start_item.setText(self._format_seconds_to_time(s))
            end_item.setText(self._format_seconds_to_time(e))
        finally:
            self._suppress_item_changed = False
        # Debounce interval refresh (120ms)
        self._debounce_timer.start(120)

    def _debounced_interval_refresh(self):
        """Apply updated interval overlays after debounced manual edits."""
        self.plot1.set_subtitle_intervals(self._collect_referencetable_intervals())
        self.plot2.set_subtitle_intervals(self._collect_synctable_intervals())
        # Keep preview in sync if it is visible
        if self._preview_win and self._preview_win.isVisible():
            self._update_preview_images_from_selection()

    # ---------- Selection -> plot highlight ----------
    def on_reference_table_selection(self):
        """Highlight selected reference rows on the reference plot."""
        indices = [idx.row() for idx in self.referencetable.selectionModel().selectedRows()]
        self.plot1.set_selected_subtitle_indices(indices)
        # If preview window is open, update its images
        if self._preview_win and self._preview_win.isVisible():
            self._update_preview_images_from_selection()

    def on_sync_table_selection(self):
        """Highlight selected sync rows on the new plot."""
        indices = [idx.row() for idx in self.synctable.selectionModel().selectedRows()]
        self.plot2.set_selected_subtitle_indices(indices)
        # If preview window is open, update its images
        if self._preview_win and self._preview_win.isVisible():
            self._update_preview_images_from_selection()

    # ---------- Context Menus ----------
    def show_synctable_context_menu(self, pos):
        """Show context menu for the sync table with row operations."""
        menu = QMenu(self)
        act_play = QAction("Play", self)
        act_jump = QAction("Jump to", self)
        act_edit = QAction("Edit...", self)
        act_delete = QAction("Delete line(s)", self)
        act_shift_sel = QAction("Shift times for selected line(s)...", self)
        act_shift_all = QAction("Shift all times...", self)
        act_shift_by_offset = QAction("Shift times by found offset for selected line(s)", self)  # NEW
        act_undo_shift = QAction("Undo shift for selected line(s)", self)
        act_find_offsets = QAction("Find Offset(s) (BBC-offset-finder)", self)
        act_find_offsets_range = QAction("Find Offset(s) in range (BBC-offset-finder)", self)
        act_export = QAction("Export subtitle", self)
        act_preview = QAction("Show preview images", self)

        menu.addAction(act_play); menu.addAction(act_jump); menu.addAction(act_edit); menu.addSeparator()
        menu.addAction(act_delete); menu.addSeparator(); menu.addAction(act_shift_sel); menu.addAction(act_shift_all); menu.addAction(act_shift_by_offset); menu.addAction(act_undo_shift); menu.addSeparator()  # MODIFIED LINE
        menu.addAction(act_find_offsets); menu.addAction(act_find_offsets_range); menu.addSeparator(); menu.addAction(act_export)
        menu.addSeparator(); menu.addAction(act_preview)

        act_play.triggered.connect(self.synctable_play_selected)
        act_jump.triggered.connect(self.synctable_jump_to_selected)
        act_edit.triggered.connect(self.edit_selected_subtitle)
        act_delete.triggered.connect(self.synctable_delete_selected)
        act_shift_sel.triggered.connect(lambda: self.shift_times(selected_only=True))
        act_shift_all.triggered.connect(lambda: self.shift_times(selected_only=False))
        act_shift_by_offset.triggered.connect(self.shift_times_by_found_offset)  # NEW
        act_undo_shift.triggered.connect(self.undo_total_shift_for_selected)
        act_find_offsets.triggered.connect(self.find_offsets_for_selected)
        act_find_offsets_range.triggered.connect(self.find_offsets_for_selected_in_range)
        act_export.triggered.connect(self.export_synctable_as_srt)
        act_preview.triggered.connect(self._update_preview_images_from_selection)

        menu.exec_(self.synctable.viewport().mapToGlobal(pos))

    def show_referencetable_context_menu(self, pos):
        """Show minimal context menu for the reference table."""
        menu = QMenu(self)
        act_play = QAction("Play", self)
        act_jump = QAction("Jump to", self)
        act_preview = QAction("Show preview images", self)
        menu.addAction(act_play); menu.addAction(act_jump); menu.addSeparator(); menu.addAction(act_preview)
        act_play.triggered.connect(self.referencetable_play_selected)
        act_jump.triggered.connect(self.referencetable_jump_to_selected)
        act_preview.triggered.connect(self._update_preview_images_from_selection)
        menu.exec_(self.referencetable.viewport().mapToGlobal(pos))

    def edit_selected_subtitle(self):
        """Open edit dialog for the first selected sync row and apply changes."""
        self.synctable.setSortingEnabled(False)
        sel = self.synctable.selectionModel().selectedRows()
        if not sel:
            return
        row = sel[0].row()
        s_item = self.synctable.item(row, 0)
        e_item = self.synctable.item(row, 1)
        t_item = self.synctable.item(row, 2)
        if not (s_item and e_item and t_item):
            return
        start_orig = s_item.text(); end_orig = e_item.text(); text_orig = t_item.text()
        new_start, new_end, new_text, ok = EditSubtitleDialog.edit(self, start_orig, end_orig, text_orig)
        if not ok:
            return

        # Compute start-time delta to accumulate into "Total shift" (col 4)
        old_start_sec = self._parse_time_to_seconds(start_orig)
        new_start_sec = self._parse_time_to_seconds(new_start)
        delta = new_start_sec - old_start_sec

        # Apply edits (suppress handler to avoid duplicate refresh)
        self._suppress_item_changed = True
        try:
            s_item.setText(new_start); e_item.setText(new_end); t_item.setText(new_text)
        finally:
            self._suppress_item_changed = False

        # Update cumulative Total shift
        ts_item = self.synctable.item(row, 5)
        if ts_item is None:
            ts_item = QTableWidgetItem("+0.000")
            self.synctable.setItem(row, 5, ts_item)
        try:
            current_total = float(ts_item.text().replace(",", "."))
        except ValueError:
            current_total = 0.0
        ts_item.setText(f"{(current_total + delta):+.3f}")

        # Visual + plot refresh
        self._mark_rows_shifted([row], all_mode=False)
        self.plot2.set_subtitle_intervals(self._collect_synctable_intervals())
        # If preview window is open, refresh images to reflect edited times
        if self._preview_win and self._preview_win.isVisible():
            self._update_preview_images_from_selection()

    # ---------- Playback (table rows) ----------
    def referencetable_play_selected(self):
        """Play the audio for the first selected reference row interval."""
        selected = self.referencetable.selectionModel().selectedRows()
        if not selected:
            return
        row = selected[0].row()
        # Lazy-load source if needed (keeps startup light)
        if self.ref_audio_segment is None:
            path = self.le1.text().strip()
            if not path or not os.path.exists(path):
                QMessageBox.warning(self, "Play", "Reference media file not loaded.")
                return
            try:
                self.ref_audio_segment = AudioSegment.from_file(path)
            except Exception as e:
                QMessageBox.critical(self, "Play", str(e))
                return
        s_item = self.referencetable.item(row, 0); e_item = self.referencetable.item(row, 1)
        if not s_item or not e_item:
            return
        s = self._parse_time_to_seconds(s_item.text()); e = self._parse_time_to_seconds(e_item.text())
        if e > s:
            self._play_audio_segment(self.ref_audio_segment, s, e)

    def synctable_play_selected(self):
        """Play the audio for the first selected sync row interval."""
        selected = self.synctable.selectionModel().selectedRows()
        if not selected:
            return
        row = selected[0].row()
        # Lazy-load target if needed
        if self.new_audio_segment is None:
            path = self.le2.text().strip()
            if not path or not os.path.exists(path):
                QMessageBox.warning(self, "Play", "New media file not loaded.")
                return
            try:
                self.new_audio_segment = AudioSegment.from_file(path)
            except Exception as e:
                QMessageBox.critical(self, "Play", str(e))
                return
        s_item = self.synctable.item(row, 0); e_item = self.synctable.item(row, 1)
        if not s_item or not e_item:
            return
        s = self._parse_time_to_seconds(s_item.text()); e = self._parse_time_to_seconds(e_item.text())
        if e > s:
            self._play_audio_segment(self.new_audio_segment, s, e)

    def _play_audio_segment(self, segment: AudioSegment, start_sec: float, end_sec: float):
        """Play a subsection of an AudioSegment via QAudioOutput."""
        if not segment:
            return
        total_dur = len(segment) / 1000.0
        start_sec = max(0.0, start_sec)
        end_sec = min(total_dur, end_sec)
        if end_sec <= start_sec:
            return
        # pydub slicing uses milliseconds
        part = segment[int(start_sec * 1000):int(end_sec * 1000)]
        if part.sample_width != 2:
            part = part.set_sample_width(2)
        fmt = QAudioFormat(); fmt.setSampleRate(part.frame_rate); fmt.setChannelCount(part.channels); fmt.setSampleSize(part.sample_width * 8)
        fmt.setCodec("audio/pcm"); fmt.setByteOrder(QAudioFormat.LittleEndian); fmt.setSampleType(QAudioFormat.SignedInt)
        # Dispose any previous output/buffer before creating a new one
        if self.audio_output:
            self.audio_output.stop(); self.audio_output.deleteLater(); self.audio_output = None
        if self.audio_buffer:
            self.audio_buffer.close(); self.audio_buffer.deleteLater(); self.audio_buffer = None
        self.audio_data = QByteArray(part.raw_data)
        self.audio_buffer = QBuffer(); self.audio_buffer.setData(self.audio_data); self.audio_buffer.open(QBuffer.ReadOnly)
        self.audio_output = QAudioOutput(fmt, self); self.audio_output.start(self.audio_buffer)

    # ---------- Jump ----------
    def referencetable_jump_to_selected(self):
        """Move reference plot to the earliest selected start time."""
        selected = self.referencetable.selectionModel().selectedRows()
        if not selected:
            return
        starts = []
        for idx in selected:
            itm = self.referencetable.item(idx.row(), 0)
            if itm:
                starts.append(self._parse_time_to_seconds(itm.text()))
        if starts:
            self.plot1.jump_to_time(min(starts), center=True)

    def synctable_jump_to_selected(self):
        """Move new plot to the earliest selected start time."""
        selected = self.synctable.selectionModel().selectedRows()
        if not selected:
            return
        starts = []
        for idx in selected:
            itm = self.synctable.item(idx.row(), 0)
            if itm:
                starts.append(self._parse_time_to_seconds(itm.text()))
        if starts:
            self.plot2.jump_to_time(min(starts), center=True)

    # ---------- Table Editing ----------
    def _collect_synctable_intervals(self) -> List[Tuple[float, float]]:
        """Return [(start_sec, end_sec)] for all rows in the sync table."""
        intervals = []
        for r in range(self.synctable.rowCount()):
            s_item = self.synctable.item(r, 0); e_item = self.synctable.item(r, 1)
            if not s_item or not e_item:
                continue
            s = self._parse_time_to_seconds(s_item.text()); e = self._parse_time_to_seconds(e_item.text())
            intervals.append((s, e))
        return intervals

    def _collect_referencetable_intervals(self) -> List[Tuple[float, float]]:
        """Return [(start_sec, end_sec)] for all rows in the reference table."""
        intervals = []
        for r in range(self.referencetable.rowCount()):
            s_item = self.referencetable.item(r, 0)
            e_item = self.referencetable.item(r, 1)
            if not s_item or not e_item:
                continue
            s = self._parse_time_to_seconds(s_item.text())
            e = self._parse_time_to_seconds(e_item.text())
            intervals.append((s, e))
        return intervals

    def synctable_delete_selected(self):
        """Delete selected rows in the sync table and mirror deletion in reference."""
        selected = self.synctable.selectionModel().selectedRows()
        if not selected:
            return

        rows_to_delete = sorted({idx.row() for idx in selected})
        if not rows_to_delete:
            return

        # Delete in both tables using the same row indices (keeps them aligned)
        self._suppress_item_changed = True
        try:
            self._delete_rows_from_table(self.synctable, rows_to_delete)
            self._delete_rows_from_table(self.referencetable, rows_to_delete)
        finally:
            self._suppress_item_changed = False

        # Refresh plots based on current table contents
        self.plot2.set_subtitle_intervals(self._collect_synctable_intervals())
        self.plot2.set_selected_subtitle_indices([])

        self.plot1.set_subtitle_intervals(self._collect_referencetable_intervals())
        self.plot1.set_selected_subtitle_indices([])

        # Keep text alignment consistent
        self.align_table_columns_left(self.referencetable)
        self.align_table_columns_left(self.synctable)

    def _delete_rows_from_table(self, table: QTableWidget, rows: List[int]):
        """Delete specified rows from the table, adjusting selection and keeping headers."""
        if not rows:
            return
        table.setSortingEnabled(False)
        # Adjust selection to avoid removing entire rows in selection
        sel_model = table.selectionModel()
        if sel_model and sel_model.hasSelection():
            new_sel = [idx.row() for idx in sel_model.selectedRows() if idx.row() not in rows]
            table.clearSelection()
            for r in new_sel:
                table.selectRow(r)
        # Delete rows in reverse order to not mess up row indices
        for r in reversed(rows):
            if 0 <= r < table.rowCount():
                table.removeRow(r)
        # Keep current visual order; do not re-enable sorting or sort by any column

    def shift_times(self, selected_only: bool):
        """Shift start/end times for selected or all rows by user-provided delta."""
        self.synctable.setSortingEnabled(False)
        if self.synctable.rowCount() == 0:
            QMessageBox.information(self, "Shift Times", "No rows to shift.")
            return
        if selected_only:
            sel = self.synctable.selectionModel().selectedRows()
            target = [i.row() for i in sel]
            if not target:
                QMessageBox.information(self, "Shift Times", "No rows selected.")
                return
            prompt = "Shift selected line(s) by seconds (e.g. -1.250 or 2.5):"
            prefill = "0.000"
            # Prefill with first numeric Found offset if available
            for r in target:
                cell = self.synctable.item(r, 3)
                if not cell:
                    continue
                raw = cell.text().strip()
                try:
                    if raw and (raw[0].isdigit() or raw[0] in "+-"):
                        val = float(raw)
                        prefill = f"{val:+.3f}"
                        break
                except ValueError:
                    continue
        else:
            target = list(range(self.synctable.rowCount()))
            prompt = "Shift ALL lines by seconds (e.g. -1.250 or 2.5):"
            prefill = "0.000"
        val_str, ok = QInputDialog.getText(self, "Shift Times", prompt, text=prefill)
        if not ok or not val_str.strip():
            return
        try:
            delta = float(val_str.replace(",", "."))
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Could not parse shift value.")
            return
        self._suppress_item_changed = True
        try:
            for r in target:
                s_item = self.synctable.item(r, 0); e_item = self.synctable.item(r, 1)
                if not s_item or not e_item:
                    continue
                s = self._parse_time_to_seconds(s_item.text()) + delta
                e = self._parse_time_to_seconds(e_item.text()) + delta
                s = max(0.0, s); e = max(s, e)
                s_item.setText(self._format_seconds_to_time(s))
                e_item.setText(self._format_seconds_to_time(e))

                # Update cumulative shift column (index 4)
                ts_item = self.synctable.item(r, 5)
                if ts_item is None:
                    ts_item = QTableWidgetItem("+0.000")
                    self.synctable.setItem(r, 5, ts_item)
                try:
                    current = float(ts_item.text().replace(",", "."))
                except ValueError:
                    current = 0.0
                new_total = current + delta
                ts_item.setText(f"{new_total:+.3f}")
        finally:
            self._suppress_item_changed = False
        self._mark_rows_shifted(target, all_mode=not selected_only)
        self.plot2.set_subtitle_intervals(self._collect_synctable_intervals())
        # If preview window is open, refresh images to reflect shifted times
        if self._preview_win and self._preview_win.isVisible():
            self._update_preview_images_from_selection()

    def undo_total_shift_for_selected(self):
        """Undo cumulative shifts for selected rows using the 'Total shift' column."""
        sel = self.synctable.selectionModel().selectedRows()
        if not sel:
            return

        any_changed = False
        self._suppress_item_changed = True
        try:
            for idx in sel:
                r = idx.row()
                s_item = self.synctable.item(r, 0)
                e_item = self.synctable.item(r, 1)
                ts_item = self.synctable.item(r, 5)  # "Total shift" (moved from column 4 to 5)
                if not (s_item and e_item and ts_item):
                    continue

                # Parse cumulative shift; skip when zero/invalid
                try:
                    total_shift = float(ts_item.text().replace(",", "."))
                except Exception:
                    total_shift = 0.0
                if abs(total_shift) < 1e-9:
                    continue

                # Apply inverse shift to start/end
                delta = -total_shift
                s = max(0.0, self._parse_time_to_seconds(s_item.text()) + delta)
                e = max(s, self._parse_time_to_seconds(e_item.text()) + delta)
                s_item.setText(self._format_seconds_to_time(s))
                e_item.setText(self._format_seconds_to_time(e))

                # Reset cumulative shift
                ts_item.setText("+0.000")

                # Restore alternating background for the row
                bg = QColor(245, 245, 245) if r % 2 == 0 else QColor(230, 230, 230)
                for c in range(self.synctable.columnCount()):
                    itm = self.synctable.item(r, c)
                    if itm:
                        itm.setBackground(bg)

                any_changed = True
        finally:
            self._suppress_item_changed = False

        if any_changed:
            # Refresh plot and preview (if open)
            self.plot2.set_subtitle_intervals(self._collect_synctable_intervals())
            if self._preview_win and self._preview_win.isVisible():
                self._update_preview_images_from_selection()

    def shift_times_by_found_offset(self):
        """Apply the 'Found offset' value from column 3 to each selected row.
    
        Only rows with valid numeric offsets are shifted. Rows with non-numeric
        values (e.g., 'range-too-short', empty, or error messages) are skipped
        without changing their color. Successfully shifted rows are colored with
        _shift_sel_color to indicate manual adjustment.
        """
        sel = self.synctable.selectionModel().selectedRows()
        if not sel:
            return

        shifted_rows = []  # Track rows that were actually shifted
        self._suppress_item_changed = True
        try:
            for idx in sel:
                r = idx.row()
                s_item = self.synctable.item(r, 0)
                e_item = self.synctable.item(r, 1)
                offset_item = self.synctable.item(r, 3)  # "Found offset" column
            
                if not (s_item and e_item and offset_item):
                    continue
            
                # Try to parse the offset value
                offset_text = offset_item.text().strip()
                if not offset_text:
                    continue  # Skip empty offsets
            
                try:
                    # Parse offset - must start with + or - and be a valid float
                    if not (offset_text[0] in "+-" and len(offset_text) > 1):
                        continue  # Skip non-numeric values like "range-too-short", "err:...", etc.
                
                    offset_value = float(offset_text.replace(",", "."))
                except (ValueError, IndexError):
                    continue  # Skip invalid values
            
                # Apply the offset to start and end times
                s = self._parse_time_to_seconds(s_item.text()) + offset_value
                e = self._parse_time_to_seconds(e_item.text()) + offset_value
                s = max(0.0, s)
                e = max(s, e)
                s_item.setText(self._format_seconds_to_time(s))
                e_item.setText(self._format_seconds_to_time(e))
            
                # Update cumulative shift column (index 4)
                ts_item = self.synctable.item(r, 5)
                if ts_item is None:
                    ts_item = QTableWidgetItem("+0.000")
                    self.synctable.setItem(r, 5, ts_item)
                try:
                    current = float(ts_item.text().replace(",", "."))
                except ValueError:
                    current = 0.0
                new_total = current + offset_value
                ts_item.setText(f"{new_total:+.3f}")
            
                # Mark this row as shifted
                shifted_rows.append(r)
        finally:
            self._suppress_item_changed = False
    
        # Color only the rows that were actually shifted
        if shifted_rows:
            self._mark_rows_shifted(shifted_rows, all_mode=False)
            self.plot2.set_subtitle_intervals(self._collect_synctable_intervals())
            # If preview window is open, refresh images to reflect shifted times
            if self._preview_win and self._preview_win.isVisible():
                self._update_preview_images_from_selection()

    def _mark_rows_shifted(self, rows: List[int], all_mode: bool):
        """Color shifted rows to visually distinguish edits.

        all_mode controls whether a different color is used for mass edits.
        """
        color = self._shift_all_color if all_mode else self._shift_sel_color
        for r in rows:
            for c in range(self.synctable.columnCount()):
                item = self.synctable.item(r, c)
                if item:
                    item.setBackground(color)

    def export_synctable_as_srt(self):
        """Export the sync table as an SRT file to the path in the save field."""
        row_count = self.synctable.rowCount()
        if row_count == 0:
            QMessageBox.information(self, "Export Subtitle", "No rows to export.")
            return
        target_path = self.le4.text().strip()
        if not target_path:
            QMessageBox.warning(self, "Export Subtitle", "Please specify a save path (Save subtitle under...).")
            return
        if not target_path.lower().endswith(".srt"):
            target_path += ".srt"
        out_dir = os.path.dirname(target_path) or "."
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Export Subtitle", f"Cannot create directory:\n{e}")
            return
        subs = pysrt.SubRipFile()

        def parse_time(ts: str):
            """Parse a HH:MM:SS,mmm string into SubRipTime or None."""
            try:
                hms, ms = ts.split(","); h, m, s = hms.split(":")
                return pysrt.SubRipTime(hours=int(h), minutes=int(m), seconds=int(s), milliseconds=int(ms))
            except Exception:
                return None

        invalid_rows = []
        for i in range(row_count):
            s_item = self.synctable.item(i, 0); e_item = self.synctable.item(i, 1); t_item = self.synctable.item(i, 2)
            if not (s_item and e_item and t_item):
                invalid_rows.append(i + 1); continue
            start_time = parse_time(s_item.text()); end_time = parse_time(e_item.text())
            if not start_time or not end_time or (end_time.to_time() <= start_time.to_time()):
                invalid_rows.append(i + 1); continue
            text = t_item.text()
            subs.append(pysrt.SubRipItem(index=len(subs) + 1, start=start_time, end=end_time, text=text))
        if not subs:
            QMessageBox.warning(self, "Export Subtitle", "No valid rows to export.")
            return
        try:
            subs.clean_indexes(); subs.save(target_path, encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Export Subtitle", f"Failed to save SRT:\n{e}")
            return
        msg = f"Exported {len(subs)} subtitle lines to:\n{target_path}"
        if invalid_rows:
            msg += f"\n\nSkipped invalid rows: {', '.join(map(str, invalid_rows))}"
        QMessageBox.information(self, "Export Subtitle", msg)

    # ---------- File Selection ----------
    def select_media_file_btn1(self):
        """Browse for reference media and populate the line edit."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Reference Media", "",
                                              "Media files (*.avi *.mkv *.mp4 *.mov *.mpg *.mpeg *.wmv *.flv *.webm);;All files (*.*)")
        if path:
            self.le1.setText(path)

    def select_media_file_btn2(self):
        """Browse for new media and populate the line edit."""
        path, _ = QFileDialog.getOpenFileName(self, "Select New Media", "",
                                              "Media files (*.avi *.mkv *.mp4 *.mov *.mpg *.mpeg *.wmv *.flv *.webm);;All files (*.*)")
        if path:
            self.le2.setText(path)
            # Auto-fill save path to match NEW media file with .srt extension
            base, _ = os.path.splitext(path)
            self.le4.setText(base + ".srt")

    def select_subtitle_file_btn3(self):
        """Browse for subtitle file and suggest an output file name."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Reference Subtitle", "",
                                              "Subtitle files (*.srt);;All files (*.*)")
        if path:
            self.le3.setText(path)
            # Do not auto-fill le4 based on reference subtitle anymore

    def save_subtitle_file_btn4(self):
        """Browse for output SRT file and populate the save path field."""
        path, _ = QFileDialog.getSaveFileName(self, "Save Subtitle As", "",
                                              "Subtitle files (*.srt);;All files (*.*)")
        if path:
            if not os.path.splitext(path)[1]:
                path += ".srt"
            self.le4.setText(path)

    # ---------- Analysis ----------
    def sanity_check_files(self) -> bool:
        """Verify file paths are present and valid; warn for any missing items."""
        missing = []
        if not self.le1.text() or not os.path.exists(self.le1.text()):
            missing.append("Reference media file")
        if not self.le2.text() or not os.path.exists(self.le2.text()):
            missing.append("New media file")
        if not self.le3.text() or not os.path.exists(self.le3.text()):
            missing.append("Reference subtitle file")
        if not self.le4.text():
            missing.append("Subtitle save path")
        if missing:
            QMessageBox.warning(self, "Missing Files", "Missing or invalid:\n\n" + "\n".join(missing))
            return False
        return True

    def on_analyze(self):
        """Start the analysis worker in a background thread with a BusyDialog."""
        if self._analyze_thread is not None:
            QMessageBox.information(self, "Analyze", "Analysis already running.")
            return
        if not self.sanity_check_files():
            return
        self._busy_dialog = BusyDialog(self, title="Analyzing", message="Starting analysis ...", cancellable=True)
        self._busy_dialog.cancel_requested.connect(lambda: self._analyze_worker and self._analyze_worker.abort())
        self._busy_dialog.show()

        thread = QThread(); worker = AnalyzeWorker(self.le1.text().strip(), self.le2.text().strip(), self.le3.text().strip())
        self._analyze_worker = worker; self._analyze_thread = thread; worker.moveToThread(thread)

        worker.progress.connect(lambda _v, msg: self._busy_dialog and self._busy_dialog.set_message(msg))

        def finished(result):
            """Handle successful analysis result and populate UI."""
            self._teardown_analysis()
            try:
                self._apply_analysis_result(result)
            except Exception as e:
                QMessageBox.critical(self, "Result Error", str(e))

        def failed(msg):
            """Handle failure from worker and dismiss UI state."""
            self._teardown_analysis(); QMessageBox.critical(self, "Analyze Failed", msg)

        def cancelled():
            """Handle user cancellation and dismiss UI state."""
            self._teardown_analysis(); QMessageBox.information(self, "Analyze", "Analysis cancelled.")

        def cleanup():
            """Ensure thread is stopped and worker pointers cleared."""
            thread.quit(); thread.wait(); self._analyze_worker = None; self._analyze_thread = None

        worker.finished.connect(finished)
        worker.failed.connect(failed)
        worker.cancelled.connect(cancelled)
        worker.finished.connect(cleanup)
        worker.failed.connect(cleanup)
        worker.cancelled.connect(cleanup)
        thread.started.connect(worker.run)

        self.btn5.setEnabled(False)
        thread.start()

    def _teardown_analysis(self):
        """Restore UI state after analyze worker finishes/fails/cancels."""
        self.btn5.setEnabled(True)
        if self._busy_dialog:
            try:
                self._busy_dialog.finish()
            except Exception:
                pass
            self._busy_dialog = None

    def _apply_analysis_result(self, result: dict):
        """Receive data from worker and populate plots and both tables."""
        self.ref_audio_segment = result["ref_full"]
        self.new_audio_segment = result["new_full"]
        self.plot1.plot_waveform(result["ref_display"], result["ref_rate"])
        self.plot2.plot_waveform(result["new_display"], result["new_rate"])
        self.plot1.set_audio_segment(self.ref_audio_segment)
        self.plot2.set_audio_segment(self.new_audio_segment)
        
        rows = result["rows"]
        fmt = lambda t: f"{t.hours:02}:{t.minutes:02}:{t.seconds:02},{t.milliseconds:03}"
        # Suppress per-cell itemChanged while bulk inserting
        self._suppress_item_changed = True
        try:
            self.referencetable.setRowCount(len(rows))
            self.synctable.setRowCount(len(rows))
            for i, r in enumerate(rows):
                self.referencetable.setItem(i, 0, QTableWidgetItem(fmt(r["start"])))
                self.referencetable.setItem(i, 1, QTableWidgetItem(fmt(r["end"])))
                self.referencetable.setItem(i, 2, QTableWidgetItem(r["text"]))
                self.synctable.setItem(i, 0, QTableWidgetItem(fmt(r["start"])))
                self.synctable.setItem(i, 1, QTableWidgetItem(fmt(r["end"])))
                self.synctable.setItem(i, 2, QTableWidgetItem(r["text"]))
                self.synctable.setItem(i, 3, QTableWidgetItem(""))  # Found offset
                self.synctable.setItem(i, 4, QTableWidgetItem(""))  # Score (NEW)
                self.synctable.setItem(i, 5, QTableWidgetItem("+0.000"))  # Total shift (moved from 4 to 5)
                bg = QColor(245, 245, 245) if i % 2 == 0 else QColor(230, 230, 230)
                for c in range(3):
                    self.referencetable.item(i, c).setBackground(bg)
                for c in range(6):  # Changed from 5 to 6
                    self.synctable.item(i, c).setBackground(bg)
        finally:
            self._suppress_item_changed = False
        self.align_table_columns_left(self.referencetable)
        self.align_table_columns_left(self.synctable)
        self.plot1.set_subtitle_intervals(result["intervals"])
        self.plot2.set_subtitle_intervals(result["intervals"])

    # ---------- Offset Support ----------
    def _ensure_audio_caches_for_offsets(self) -> bool:
        """Prepare mono float caches for ref/new audio and check dependency.

        Returns False if audio-offset-finder is not installed or source media
        can't be loaded; otherwise True.
        """
        try:
            from audio_offset_finder.audio_offset_finder import find_offset_between_buffers as _tmp  # noqa: F401
        except Exception:
            QMessageBox.warning(self, "Offset Finder", "audio-offset-finder not installed.")
            return False

        def load(seg_attr, path_line_edit):
            """Lazy-load an AudioSegment attribute from the provided path edit."""
            if getattr(self, seg_attr) is None:
                p = path_line_edit.text().strip()
                if not p or not os.path.exists(p):
                    return False
                try:
                    setattr(self, seg_attr, AudioSegment.from_file(p))
                except Exception as e:
                    QMessageBox.critical(self, "Offset Finder", str(e)); return False
            return True

        if not load("ref_audio_segment", self.le1):
            return False
        if not load("new_audio_segment", self.le2):
            return False

        # Ensure caches are already float32 to avoid duplication in workers
        if self._ref_mono_cache is None:
            cache, sr = segment_to_float_array(self.ref_audio_segment)  # type: ignore[arg-type]
            # Pre-convert to float32 to avoid repeated conversions
            self._ref_mono_cache = cache if cache.dtype == np.float32 else cache.astype(np.float32)
            self._ref_sr_cache = sr
        if self._new_mono_cache is None:
            cache, sr = segment_to_float_array(self.new_audio_segment)  # type: ignore[arg-type]
            # Pre-convert to float32 to avoid repeated conversions
            self._new_mono_cache = cache if cache.dtype == np.float32 else cache.astype(np.float32)
            self._new_sr_cache = sr
        return True

    def cancel_offset_worker(self):
        """Signal the running offset worker to abort and update BusyDialog."""
        if self._offset_worker:
            self._offset_worker.abort()
            if self._busy_offset:
                self._busy_offset.set_message("Cancelling ...")
        # If a thread exists but is no longer running, clear it proactively
        if self._offset_thread and not self._offset_thread.isRunning():
            try:
                self._offset_thread.quit()
                self._offset_thread.wait()
            except Exception:
                pass
            finally:
                self._offset_thread = None
                self._offset_worker = None

    def find_offsets_for_selected(self):
        """Compute offsets for selected rows by matching reference lines inside the NEW audio."""
        if self.synctable.rowCount() == 0:
            QMessageBox.information(self, "Offset Finder", "No rows available.")
            return
        sel = self.synctable.selectionModel().selectedRows()
        if not sel:
            QMessageBox.information(self, "Offset Finder", "No rows selected.")
            return
        if not self._ensure_audio_caches_for_offsets():
            return
        # Only block if a thread exists AND is running
        if self._offset_thread is not None and self._offset_thread.isRunning():
            QMessageBox.information(self, "Offset Finder", "Offset computation already running.")
            return

        # Use the corresponding reference table times as the snippet to search for
        rows = []
        for idx in sel:
            r = idx.row()
            s_item = self.referencetable.item(r, 0)
            e_item = self.referencetable.item(r, 1)
            if not s_item or not e_item:
                continue
            s = self._parse_time_to_seconds(s_item.text())
            e = self._parse_time_to_seconds(e_item.text())
            rows.append((r, s, e))
        if not rows:
            QMessageBox.information(self, "Offset Finder", "Selected rows have no valid times.")
            return

        self._busy_offset = BusyDialog(self, title="Finding Offsets", message="Searching ...", cancellable=True)
        self._busy_offset.cancel_requested.connect(self.cancel_offset_worker)
        self._busy_offset.show()

        # Swap buffers: search reference-line snippets inside the NEW audio
        thread = QThread()
        worker = OffsetWorker(
            self._new_mono_cache, self._new_sr_cache,   # search corpus (NEW)
            self._ref_mono_cache, self._ref_sr_cache,   # snippets from (REFERENCE)
            rows, 1.0, ref_offset_sec=0.0               # offset base for the first buffer (NEW) window
        )  # type: ignore[arg-type]
        self._offset_thread = thread
        self._offset_worker = worker
        worker.moveToThread(thread)

        def on_result(row_index: int, delta_val, status: str, score, _worker=worker):
            """Update the Found offset cell per result and color by status."""
            if _worker is not self._offset_worker:
                return
            cell = self.synctable.item(row_index, 3)
            if cell is None:
                self.synctable.setItem(row_index, 3, QTableWidgetItem(""))
                cell = self.synctable.item(row_index, 3)
            score_cell = self.synctable.item(row_index, 4)
            if score_cell is None:
                self.synctable.setItem(row_index, 4, QTableWidgetItem(""))
                score_cell = self.synctable.item(row_index, 4)
            if delta_val is None:
                cell.setText(status)
                cell.setBackground(QColor(240, 240, 200) if not status.startswith("err") else QColor(255, 210, 210))
                score_cell.setText("")
                score_cell.setBackground(QColor(240, 240, 200) if not status.startswith("err") else QColor(255, 210, 210))
            else:
                adj = -delta_val
                cell.setText(f"{adj:+.3f}")
                if score is not None and float(score) < 7.0:
                    offset_color = QColor(255, 210, 210)
                    score_color = QColor(255, 210, 210)
                elif status == "ok":
                    offset_color = QColor(210, 245, 210)
                    score_color = QColor(210, 245, 210)
                else:
                    offset_color = QColor(255, 210, 210)
                    score_color = QColor(255, 210, 210)
                cell.setBackground(offset_color)
                if score is not None:
                    score_cell.setText(f"{float(score):.2f}")
                    score_cell.setBackground(score_color)
                else:
                    score_cell.setText("")
                    score_cell.setBackground(QColor(240, 240, 200))

        def on_progress(row_index: int, msg: str, _worker=worker):
            if _worker is not self._offset_worker:
                return
            if self._busy_offset:
                self._busy_offset.set_message(f"Processed {msg}")

        def on_finished(_worker=worker):
            self._finish_offset_worker()

        def on_failed(err: str, _worker=worker):
            if err != "Aborted":
                QMessageBox.critical(self, "Offset Finder", err)
            self._finish_offset_worker()

        def on_cancelled(_worker=worker):
            self._finish_offset_worker()

        def cleanup(_worker=worker):
            # Always stop and clear regardless of running state to avoid stale pointers after cancel
            try:
                thread.quit()
                thread.wait(1500)
            except Exception:
                pass
            finally:
                if self._offset_thread is thread:
                    self._offset_thread = None
                if self._offset_worker is worker:
                    self._offset_worker = None

        worker.result.connect(on_result)
        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.failed.connect(on_failed)
        worker.cancelled.connect(on_cancelled)
        # Ensure cleanup runs for all terminal signals
        worker.finished.connect(cleanup)
        worker.failed.connect(cleanup)
        worker.cancelled.connect(cleanup)
        thread.started.connect(worker.run)
        thread.start()

    def _finish_offset_worker(self):
        """Hide BusyDialog and fully tear down any active offset worker/thread."""
        if self._busy_offset:
            try:
                self._busy_offset.finish()
            except Exception:
                pass
            self._busy_offset = None
        # Always attempt to stop thread if it exists (even if still running after cancel)
        if self._offset_thread:
            try:
                if self._offset_thread.isRunning():
                    self._offset_thread.quit()
                    self._offset_thread.wait(1500)
            except Exception:
                pass
        # Clear pointers unconditionally
        self._offset_thread = None
        self._offset_worker = None

    def find_offsets_for_selected_in_range(self):
        """Compute offsets by matching reference lines inside the NEW audio.

        Behavior:
        - Ask the user for a single search range LENGTH in minutes (floats allowed).
        - Optional sign handling:
          +L -> NEW slice [ref_s, ref_e + L]
          -L -> NEW slice [ref_s - L, ref_e]
          no sign -> NEW slice [ref_s - L, ref_e + L]
        - All slices are clamped to NEW audio duration.
        """
        if self.synctable.rowCount() == 0:
            QMessageBox.information(self, "Offset Finder", "No rows available.")
            return
        sel = self.synctable.selectionModel().selectedRows()
        if not sel:
            QMessageBox.information(self, "Offset Finder", "No rows selected.")
            return
        if not self._ensure_audio_caches_for_offsets():
            return
        # Only block if a thread exists AND is running
        if self._offset_thread is not None and self._offset_thread.isRunning():
            QMessageBox.information(self, "Offset Finder", "Offset computation already running.")
            return

        # Gather selected rows in ascending order and collect their REFERENCE times
        indices = sorted([i.row() for i in sel])
        rows: List[Tuple[int, float, float]] = []
        for r in indices:
            rs_item = self.referencetable.item(r, 0)
            re_item = self.referencetable.item(r, 1)
            if not rs_item or not re_item:
                continue
            ref_s = self._parse_time_to_seconds(rs_item.text())
            ref_e = self._parse_time_to_seconds(re_item.text())
            rows.append((r, ref_s, ref_e))
        if not rows:
            QMessageBox.information(self, "Offset Finder", "Selected rows have no valid times.")
            return
        if self.new_audio_segment is None:
            QMessageBox.warning(self, "Offset Finder", "New audio not loaded.")
            return

        # Ask for range length (minutes -> seconds) with optional leading sign (+/-)
        val_str, ok = QInputDialog.getText(
            self,
            "Search Range",
            "Enter search range length in minutes (e.g. 0.5, 1.5, +1.0, -0.5):",
            text="1.0",
        )
        if not ok or not val_str.strip():
            return

        raw = val_str.strip().replace(",", ".")
        sign_char: Optional[str] = None
        if raw and raw[0] in "+-":
            sign_char = raw[0]
            raw = raw[1:].strip()

        try:
            minutes = float(raw)
        except ValueError:
            QMessageBox.warning(self, "Search Range", "Invalid number.")
            return
        if minutes <= 0:
            QMessageBox.warning(self, "Search Range", "Range must be greater than 0.")
            return
        L_sec = minutes * 60.0

        new_total_sec = len(self.new_audio_segment) / 1000.0

        # Local cancellation flag (set when user presses Cancel)
        cancelled = {"flag": False}

        # Busy dialog
        self._busy_offset = BusyDialog(
            self,
            title="Finding Offsets (± range around reference)",
            message="Preparing ...",
            cancellable=True,
        )

        def _request_cancel():
            cancelled["flag"] = True
            self.cancel_offset_worker()
            if self._busy_offset:
                self._busy_offset.set_message("Cancelling ...")

        self._busy_offset.cancel_requested.connect(_request_cancel)
        self._busy_offset.show()

        self._offset_thread = None
        self._offset_worker = None

        def run_task(task_index: int):
            if cancelled["flag"] or task_index >= len(rows):
                finish_all()
                return

            row_idx, ref_s, ref_e = rows[task_index]

            # Build NEW slice based on sign: +L, -L or both directions (no sign)
            if sign_char == "+":
                slice_start = max(0.0, ref_s)
                slice_end = min(new_total_sec, ref_e + L_sec)
            elif sign_char == "-":
                slice_start = max(0.0, ref_s - L_sec)
                slice_end = min(new_total_sec, ref_e)
            else:
                slice_start = max(0.0, ref_s - L_sec)
                slice_end = min(new_total_sec, ref_e + L_sec)

            if slice_end <= slice_start:
                update_cell(row_idx, None, "range-too-short", None)  # Added score parameter
                run_task(task_index + 1)
                return

            # Convert to sample indices and extract NEW slice
            start_i = int(slice_start * self._new_sr_cache)  # type: ignore[operator]
            end_i = min(int(slice_end * self._new_sr_cache), self._new_mono_cache.shape[0])  # type: ignore[operator]
            if end_i - start_i < 100:
                update_cell(row_idx, None, "range-too-short", None)  # Added score parameter
                run_task(task_index + 1)
                return

            if cancelled["flag"]:
                finish_all()
                return

            # Use array view - OffsetWorker will handle conversion if needed
            new_slice = self._new_mono_cache[start_i:end_i]  # type: ignore[index]
            
            # Only make contiguous if needed (audio_offset_finder may require it)
            # This avoids unnecessary copying when the slice is already contiguous
            if not new_slice.flags['C_CONTIGUOUS']:
                new_slice = np.ascontiguousarray(new_slice)

            if self._busy_offset and not cancelled["flag"]:
                self._busy_offset.set_message(
                    f"Line {task_index + 1}/{len(rows)}: searching {slice_start:.3f}s–{slice_end:.3f}s"
                )

            thread = QThread()
            worker = OffsetWorker(
                new_slice, self._new_sr_cache,              # search corpus: NEW slice
                self._ref_mono_cache, self._ref_sr_cache,   # snippets: full REFERENCE
                [(row_idx, ref_s, ref_e)], 1.0,             # single row task
                ref_offset_sec=slice_start                   # offsets relative to NEW slice start
            )  # type: ignore[arg-type]

            self._offset_thread = thread
            self._offset_worker = worker
            worker.moveToThread(thread)

            def on_result(r_i: int, delta_val, status: str, score, _worker=worker):  # Added score parameter
                if _worker is not self._offset_worker:
                    return
                if delta_val is None:
                    update_cell(r_i, None, status, None)  # Added score parameter
                else:
                    # Negate to keep "reference in NEW" UI convention
                    update_cell(r_i, -delta_val, status, score)  # Added score parameter

            def on_progress(_row_index: int, msg: str, _worker=worker):
                if _worker is not self._offset_worker:
                    return
                if self._busy_offset and not cancelled["flag"]:
                    self._busy_offset.set_message(f"Line {task_index + 1}/{len(rows)}: {msg}")

            def on_finished(_worker=worker):
                cleanup_current()
                if cancelled["flag"]:
                    finish_all()
                else:
                    run_task(task_index + 1)

            def on_failed(err: str, _worker=worker):
                cleanup_current()
                if err == "Aborted" or cancelled["flag"]:
                    cancelled["flag"] = True
                    finish_all()
                else:
                    QMessageBox.critical(self, "Offset Finder", err)
                    run_task(task_index + 1)

            def on_cancelled(_worker=worker):
                cancelled["flag"] = True
                cleanup_current()
                finish_all()

            def cleanup_current():
                try:
                    thread.quit()
                    thread.wait()
                except Exception:
                    pass
                finally:
                    if self._offset_worker is worker:
                        self._offset_worker = None
                    if self._offset_thread is thread:
                        self._offset_thread = None
                    
                    # Explicitly delete worker and thread to release memory
                    try:
                        worker.deleteLater()
                    except Exception:
                        pass
                    try:
                        thread.deleteLater()
                    except Exception:
                        pass
                    
                    # Force garbage collection to free memory immediately
                    import gc
                    gc.collect()

            worker.result.connect(on_result)
            worker.progress.connect(on_progress)
            worker.finished.connect(on_finished)
            worker.failed.connect(on_failed)
            worker.cancelled.connect(on_cancelled)
            thread.started.connect(worker.run)
            thread.start()

        def finish_all():
            if self._busy_offset:
                try:
                    self._busy_offset.finish()
                except Exception:
                    pass
                self._busy_offset = None

        def update_cell(row_index: int, delta_val, status: str, score):  # Added score parameter
            # Found offset cell (column 3)
            cell = self.synctable.item(row_index, 3)
            if cell is None:
                self.synctable.setItem(row_index, 3, QTableWidgetItem(""))
                cell = self.synctable.item(row_index, 3)
            
            # Score cell (column 4) - NEW
            score_cell = self.synctable.item(row_index, 4)
            if score_cell is None:
                self.synctable.setItem(row_index, 4, QTableWidgetItem(""))
                score_cell = self.synctable.item(row_index, 4)
            
            if delta_val is None:
                cell.setText(status)
                cell.setBackground(QColor(240, 240, 200) if not status or not status.startswith("err") else QColor(255, 210, 210))
                score_cell.setText("")
                score_cell.setBackground(QColor(240, 240, 200) if not status or not status.startswith("err") else QColor(255, 210, 210))
            else:
                sign = "+" if delta_val >= 0 else "-"
                cell.setText(f"{sign}{abs(float(delta_val)):.3f}")
                
                # Determine color based on score value
                if score is not None and float(score) < 7.0:
                    # Red for low scores (below 7)
                    offset_color = QColor(255, 210, 210)
                    score_color = QColor(255, 210, 210)
                elif status == "ok":
                    # Green for good scores
                    offset_color = QColor(210, 245, 210)
                    score_color = QColor(210, 245, 210)
                else:
                    # Default red for errors
                    offset_color = QColor(255, 210, 210)
                    score_color = QColor(255, 210, 210)
                
                cell.setBackground(offset_color)
                
                # Display score
                if score is not None:
                    score_cell.setText(f"{float(score):.2f}")
                    score_cell.setBackground(score_color)
                else:
                    score_cell.setText("")
                    score_cell.setBackground(QColor(240, 240, 200))

        # Start processing
        run_task(0)

    # ---------- Time Helpers ----------
    def _parse_time_to_seconds(self, text: str) -> float:
        """Parse HH:MM:SS,mmm string into seconds (returns 0.0 on failure)."""
        try:
            h, m, rest = text.split(":"); s, ms = rest.split(",")
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
        except Exception:
            return 0.0

    def _format_seconds_to_time(self, seconds: float) -> str:
        """Format seconds into SRT time string HH:MM:SS,mmm."""
        return format_seconds(seconds)

    def _parse_user_time_any(self, text: str) -> Optional[float]:
        """Parse either an SRT timecode or a float seconds value from free text."""
        if not text:
            return None
        t = text.strip()
        try:
            cand = t
            if ',' not in cand and cand.count(':') >= 2:
                if '.' in cand and cand.rsplit('.', 1)[1].isdigit():
                    cand = cand.replace('.', ',', 1)
                else:
                    cand += ",000"
            if cand.count(':') >= 2 and ',' in cand:
                return self._parse_time_to_seconds(cand)
        except Exception:
            pass
        try:
            return float(t.replace(',', '.'))
        except ValueError:
            return None

    def stop_playback_external(self):
        """Stop shared QAudioOutput if still active (called by plot widgets)."""
        if self.audio_output is not None:
            try:
                # This method existed in the monolith but was unused for MainWindow
                self.audio_output.stop()
            except Exception:
                pass
    
    def closeEvent(self, event):
        """Ensure all background threads and audio are stopped before exit."""
        # Stop plot-local playback
        try:
            self.plot1.stop_playback_external()
        except Exception:
            pass
        try:
            self.plot2.stop_playback_external()
        except Exception:
            pass

        # Stop table row preview audio
        try:
            if self.audio_output:
                self.audio_output.stop()
                self.audio_output.deleteLater()
                self.audio_output = None
        except Exception:
            pass
        try:
            if self.audio_buffer:
                self.audio_buffer.close()
                self.audio_buffer.deleteLater()
                self.audio_buffer = None
        except Exception:
            pass

        # Stop analysis worker/thread
        try:
            if self._analyze_worker:
                try:
                    self._analyze_worker.abort()
                except Exception:
                    pass
            if self._analyze_thread:
                self._analyze_thread.quit()
                self._analyze_thread.wait(1500)
        except Exception:
            pass
        finally:
            self._analyze_worker = None
            self._analyze_thread = None

        # Stop offset worker/thread
        try:
            if self._offset_worker:
                try:
                    self._offset_worker.abort()
                except Exception:
                    pass
            if self._offset_thread:
                self._offset_thread.quit()
                self._offset_thread.wait(1500)
        except Exception:
            pass
        finally:
            self._offset_worker = None
            self._offset_thread = None

        # Close preview window (cancels its threads in its closeEvent)
        try:
            if self._preview_win and self._preview_win.isVisible():
                self._preview_win.close()
        except Exception:
            pass

        # Dismiss busy dialogs
        try:
            if self._busy_dialog:
                self._busy_dialog.finish()
        except Exception:
            pass
        try:
            if self._busy_offset:
                self._busy_offset.finish()
        except Exception:
            pass

        super().closeEvent(event)
