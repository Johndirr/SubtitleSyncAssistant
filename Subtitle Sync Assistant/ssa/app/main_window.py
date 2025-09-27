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

Wires together UI widgets, dialogs and background workers. Logic preserved.
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

from PyQt5.QtCore import Qt, QByteArray, QBuffer, QThread, pyqtSignal
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
        self.resize(1024, 600)
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

        self.synctable = QTableWidget(0, 5)
        self.synctable.setHorizontalHeaderLabels(["Start time", "End time", "Text", "Found offset", "Total shift"])
        self._init_table_column_sizing(self.synctable, [0, 1, 3, 4], [2])
        # Header tooltip for "Total shift"
        hdrTotalshift = self.synctable.horizontalHeaderItem(4)
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

    def _update_preview_images_from_selection(self):
        """Open/refresh the preview using the current table selections.

        Collects start times from both tables and the media file paths from
        the line edits, then forwards these to the preview window so it can
        extract and display frames side by side.
        """
        # Read the selected start times (seconds) from both tables
        ref_time = self._selected_row_start(self.referencetable)
        new_time = self._selected_row_start(self.synctable)
        # Read the media file paths from the top inputs (empty -> None)
        ref_path = self.le1.text().strip() if self.le1.text().strip() else None
        new_path = self.le2.text().strip() if self.le2.text().strip() else None
        # Ask the preview to show/update with the provided selection
        self._preview_window().show_for_selection(ref_path, ref_time, new_path, new_time)

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
        act_shift_sel = QAction("Shift times for selected line(s)", self)
        act_shift_all = QAction("Shift all times", self)
        act_undo_shift = QAction("Undo shift for selected line(s)", self)
        act_find_offsets = QAction("Find Offset(s) (BBC-offset-finder)", self)
        act_find_offsets_range = QAction("Find Offset(s) in range (BBC-offset-finder)", self)
        act_export = QAction("Export subtitle", self)
        act_preview = QAction("Show preview images", self)

        menu.addAction(act_play); menu.addAction(act_jump); menu.addAction(act_edit); menu.addSeparator()
        menu.addAction(act_delete); menu.addSeparator(); menu.addAction(act_shift_sel); menu.addAction(act_shift_all); menu.addAction(act_undo_shift); menu.addSeparator()
        menu.addAction(act_find_offsets); menu.addAction(act_find_offsets_range); menu.addSeparator(); menu.addAction(act_export)
        menu.addSeparator(); menu.addAction(act_preview)

        act_play.triggered.connect(self.synctable_play_selected)
        act_jump.triggered.connect(self.synctable_jump_to_selected)
        act_edit.triggered.connect(self.edit_selected_subtitle)
        act_delete.triggered.connect(self.synctable_delete_selected)
        act_shift_sel.triggered.connect(lambda: self.shift_times(selected_only=True))
        act_shift_all.triggered.connect(lambda: self.shift_times(selected_only=False))
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

        # Apply edits
        s_item.setText(new_start); e_item.setText(new_end); t_item.setText(new_text)

        # Update cumulative Total shift
        ts_item = self.synctable.item(row, 4)
        if ts_item is None:
            ts_item = QTableWidgetItem("+0.000")
            self.synctable.setItem(row, 4, ts_item)
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
        self._delete_rows_from_table(self.synctable, rows_to_delete)
        self._delete_rows_from_table(self.referencetable, rows_to_delete)

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
        table.setSortingEnabled(True)
        table.sortByColumn(0, Qt.AscendingOrder)

    def shift_times(self, selected_only: bool):
        """Shift start/end times for selected or all rows by user-provided delta."""
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
            ts_item = self.synctable.item(r, 4)
            if ts_item is None:
                ts_item = QTableWidgetItem("+0.000")
                self.synctable.setItem(r, 4, ts_item)
            try:
                current = float(ts_item.text().replace(",", "."))
            except ValueError:
                current = 0.0
            new_total = current + delta
            ts_item.setText(f"{new_total:+.3f}")
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
        for idx in sel:
            r = idx.row()
            s_item = self.synctable.item(r, 0)
            e_item = self.synctable.item(r, 1)
            ts_item = self.synctable.item(r, 4)  # "Total shift"
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

        if any_changed:
            # Refresh plot and preview (if open)
            self.plot2.set_subtitle_intervals(self._collect_synctable_intervals())
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

    def select_subtitle_file_btn3(self):
        """Browse for subtitle file and suggest an output file name."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Reference Subtitle", "",
                                              "Subtitle files (*.srt);;All files (*.*)")
        if path:
            self.le3.setText(path)
            base, ext = os.path.splitext(path)
            self.le4.setText(f"{base}_resync{ext}")

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
        self.ref_audio_segment = result["ref_full"]; self.new_audio_segment = result["new_full"]
        self.plot1.plot_waveform(result["ref_display"], result["ref_rate"]) ; self.plot2.plot_waveform(result["new_display"], result["new_rate"])
        self.plot1.set_audio_segment(self.ref_audio_segment); self.plot2.set_audio_segment(self.new_audio_segment)
        rows = result["rows"]
        fmt = lambda t: f"{t.hours:02}:{t.minutes:02}:{t.seconds:02},{t.milliseconds:03}"
        self.referencetable.setRowCount(len(rows)); self.synctable.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self.referencetable.setItem(i, 0, QTableWidgetItem(fmt(r["start"]))); self.referencetable.setItem(i, 1, QTableWidgetItem(fmt(r["end"]))); self.referencetable.setItem(i, 2, QTableWidgetItem(r["text"]))
            self.synctable.setItem(i, 0, QTableWidgetItem(fmt(r["start"]))); self.synctable.setItem(i, 1, QTableWidgetItem(fmt(r["end"]))); self.synctable.setItem(i, 2, QTableWidgetItem(r["text"]))
            self.synctable.setItem(i, 3, QTableWidgetItem("")); self.synctable.setItem(i, 4, QTableWidgetItem("+0.000"))
            bg = QColor(245, 245, 245) if i % 2 == 0 else QColor(230, 230, 230)
            for c in range(3):
                self.referencetable.item(i, c).setBackground(bg)
            for c in range(5):
                self.synctable.item(i, c).setBackground(bg)
        self.align_table_columns_left(self.referencetable); self.align_table_columns_left(self.synctable)
        self.plot1.set_subtitle_intervals(result["intervals"]); self.plot2.set_subtitle_intervals(result["intervals"])

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

        if self._ref_mono_cache is None:
            self._ref_mono_cache, self._ref_sr_cache = segment_to_float_array(self.ref_audio_segment)  # type: ignore[arg-type]
        if self._new_mono_cache is None:
            self._new_mono_cache, self._new_sr_cache = segment_to_float_array(self.new_audio_segment)  # type: ignore[arg-type]
        return True

    def cancel_offset_worker(self):
        """Signal the running offset worker to abort and update BusyDialog."""
        if self._offset_worker:
            self._offset_worker.abort()
            if self._busy_offset:
                self._busy_offset.set_message("Cancelling ...")

    def find_offsets_for_selected(self):
        """Compute offsets for selected rows using the full reference."""
        if self.synctable.rowCount() == 0:
            QMessageBox.information(self, "Offset Finder", "No rows available.")
            return
        sel = self.synctable.selectionModel().selectedRows()
        if not sel:
            QMessageBox.information(self, "Offset Finder", "No rows selected.")
            return
        if not self._ensure_audio_caches_for_offsets():
            return
        if self._offset_thread is not None:
            QMessageBox.information(self, "Offset Finder", "Offset computation already running.")
            return
        rows = []
        for idx in sel:
            r = idx.row(); s_item = self.synctable.item(r, 0); e_item = self.synctable.item(r, 1)
            if not s_item or not e_item:
                continue
            s = self._parse_time_to_seconds(s_item.text()); e = self._parse_time_to_seconds(e_item.text())
            rows.append((r, s, e))
        if not rows:
            QMessageBox.information(self, "Offset Finder", "Selected rows have no valid times.")
            return
        self._busy_offset = BusyDialog(self, title="Finding Offsets", message="Searching ...", cancellable=True)
        self._busy_offset.cancel_requested.connect(self.cancel_offset_worker)
        self._busy_offset.show()
        thread = QThread(); worker = OffsetWorker(self._ref_mono_cache, self._ref_sr_cache, self._new_mono_cache, self._new_sr_cache, rows, 1.0, ref_offset_sec=0.0)  # type: ignore[arg-type]
        self._offset_thread = thread; self._offset_worker = worker; worker.moveToThread(thread)

        def on_result(row_index: int, delta_val, status: str, _worker=worker):
            """Update the Found offset cell per result and color by status."""
            if _worker is not self._offset_worker:
                return
            cell = self.synctable.item(row_index, 3)
            if cell is None:
                self.synctable.setItem(row_index, 3, QTableWidgetItem(""))
                cell = self.synctable.item(row_index, 3)
            if delta_val is None:
                cell.setText(status); cell.setBackground(QColor(240, 240, 200) if not status.startswith("err") else QColor(255, 210, 210))
            else:
                sign = "+" if delta_val >= 0 else "-"
                cell.setText(f"{sign}{abs(delta_val):.3f}")
                cell.setBackground(QColor(210, 245, 210) if status == "ok" else QColor(255, 210, 210))

        def on_progress(row_index: int, msg: str, _worker=worker):
            """Update BusyDialog with progress text from the worker."""
            if _worker is not self._offset_worker:
                return
            if self._busy_offset:
                self._busy_offset.set_message(f"Processed {msg}")

        def on_finished(_worker=worker):
            """Tear down BusyDialog when worker completes."""
            if _worker is not self._offset_worker:
                return
            self._finish_offset_worker()

        def on_failed(err: str, _worker=worker):
            """Notify on failure (unless user aborted) and tear down."""
            if _worker is not self._offset_worker:
                return
            if err != "Aborted":
                QMessageBox.critical(self, "Offset Finder", err)
            self._finish_offset_worker()

        def on_cancelled(_worker=worker):
            """Handle cancellation and tear down."""
            if _worker is not self._offset_worker:
                return
            self._finish_offset_worker()

        def cleanup(_worker=worker):
            """Stop thread and clear worker pointers after finish/cancel/fail."""
            if _worker is self._offset_worker:
                thread.quit(); thread.wait(); self._offset_thread = None; self._offset_worker = None

        worker.result.connect(on_result)
        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.failed.connect(on_failed)
        worker.cancelled.connect(on_cancelled)
        worker.finished.connect(cleanup)
        worker.failed.connect(cleanup)
        worker.cancelled.connect(cleanup)
        thread.started.connect(worker.run)
        thread.start()

    def find_offsets_for_selected_in_range(self):
        """Compute offsets for selected rows using a chosen reference range.

        Single selection uses a direct range; multiple rows use a sliding
        window aligned to the earliest selected row to keep windows similar.
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
        if self._offset_thread is not None:
            QMessageBox.information(self, "Offset Finder", "Offset computation already running.")
            return
        rows = []
        earliest = float('inf'); latest = 0.0
        for idx in sel:
            r = idx.row(); s_item = self.synctable.item(r, 0); e_item = self.synctable.item(r, 1)
            if not s_item or not e_item:
                continue
            s = self._parse_time_to_seconds(s_item.text()); e = self._parse_time_to_seconds(e_item.text())
            rows.append((r, s, e)); earliest = min(earliest, s); latest = max(latest, e)
        if not rows:
            QMessageBox.information(self, "Offset Finder", "Selected rows have no valid times.")
            return
        if self.ref_audio_segment is None:
            QMessageBox.warning(self, "Offset Finder", "Reference audio not loaded.")
            return
        ref_total_sec = len(self.ref_audio_segment) / 1000.0
        pad = 5.0; default_start = max(0.0, earliest - pad); default_end = min(ref_total_sec, latest + pad)
        ref_start, ref_end, ok = RangeSelectDialog.get_range(self, default_start, default_end)
        if not ok:
            return
        if ref_start < 0 or ref_end > ref_total_sec:
            QMessageBox.warning(self, "Reference Range", "Range outside reference duration.")
            return
        window_len = ref_end - ref_start
        if window_len <= 0:
            QMessageBox.warning(self, "Reference Range", "Invalid range length.")
            return
        multiple = len(rows) > 1
        if not multiple:
            # Single window direct search
            start_i = int(ref_start * self._ref_sr_cache); end_i = min(int(ref_end * self._ref_sr_cache), self._ref_mono_cache.shape[0])  # type: ignore[operator]
            if end_i - start_i < 100:
                QMessageBox.warning(self, "Reference Range", "Range too short.")
                return
            ref_slice = self._ref_mono_cache[start_i:end_i]  # type: ignore[index]
            self._busy_offset = BusyDialog(self, title="Finding Offsets (Range)", message=f"Searching {ref_start:.3f}s-{ref_end:.3f}s ...", cancellable=True)
            self._busy_offset.cancel_requested.connect(self.cancel_offset_worker); self._busy_offset.show()
            thread = QThread(); worker = OffsetWorker(ref_slice, self._ref_sr_cache, self._new_mono_cache, self._new_sr_cache, rows, 1.0, ref_offset_sec=ref_start)  # type: ignore[arg-type]
        else:
            # Build windows for each row by shifting the base window from the first row
            rows_sorted = sorted(rows, key=lambda x: x[1])
            anchor_start = rows_sorted[0][1]
            ref_windows: List[Tuple[np.ndarray, int, float]] = []
            for (row_idx, row_start, _row_end) in rows_sorted:
                shift = row_start - anchor_start
                win_start = ref_start + shift
                if win_start < 0:
                    win_start = 0.0
                win_end = win_start + window_len
                if win_end > ref_total_sec:
                    win_start = max(0.0, ref_total_sec - window_len)
                    win_end = ref_total_sec
                start_i = int(win_start * self._ref_sr_cache); end_i = min(int(win_end * self._ref_sr_cache), self._ref_mono_cache.shape[0])  # type: ignore[operator]
                if end_i - start_i < 100:
                    ref_slice = np.array([], dtype=np.float32)
                else:
                    ref_slice = self._ref_mono_cache[start_i:end_i]  # type: ignore[index]
                ref_windows.append((ref_slice, self._ref_sr_cache, win_start))  # type: ignore[arg-type]
            self._busy_offset = BusyDialog(self, title="Finding Offsets (Sliding Range)", message=f"Sliding window {window_len:.3f}s over {len(rows_sorted)} lines ...", cancellable=True)
            self._busy_offset.cancel_requested.connect(self.cancel_offset_worker); self._busy_offset.show()
            thread = QThread(); worker = SlidingOffsetWorker(self._new_mono_cache, self._new_sr_cache, rows_sorted, ref_windows, 1.0)  # type: ignore[arg-type]
        self._offset_thread = thread; self._offset_worker = worker; worker.moveToThread(thread)

        def on_result(row_index: int, delta_val, status: str, _worker=worker):
            """Update per-row result for sliding-range mode."""
            if _worker is not self._offset_worker:
                return
            cell = self.synctable.item(row_index, 3)
            if cell is None:
                self.synctable.setItem(row_index, 3, QTableWidgetItem(""))
                cell = self.synctable.item(row_index, 3)
            if delta_val is None:
                cell.setText(status); cell.setBackground(QColor(240, 240, 200) if not status.startswith("err") else QColor(255, 210, 210))
            else:
                sign = "+" if delta_val >= 0 else "-"
                cell.setText(f"{sign}{abs(delta_val):.3f}")
                cell.setBackground(QColor(210, 245, 210) if status == "ok" else QColor(255, 210, 210))

        def on_progress(row_index: int, msg: str, _worker=worker):
            """Propagate progress messages to the BusyDialog."""
            if _worker is not self._offset_worker:
                return
            if self._busy_offset:
                self._busy_offset.set_message(f"Processed {msg}")

        def on_finished(_worker=worker):
            """Tear down after sliding-range search completes."""
            if _worker is not self._offset_worker:
                return
            self._finish_offset_worker()

        def on_failed(err: str, _worker=worker):
            """Notify on failure (unless aborted) then tear down."""
            if _worker is not self._offset_worker:
                return
            if err != "Aborted":
                QMessageBox.critical(self, "Offset Finder", err)
            self._finish_offset_worker()

        def on_cancelled(_worker=worker):
            """Handle user cancellation then tear down."""
            if _worker is not self._offset_worker:
                return
            self._finish_offset_worker()

        def cleanup(_worker=worker):
            """Stop thread and clear worker pointers after finish/cancel/fail."""
            if _worker is self._offset_worker:
                thread.quit(); thread.wait(); self._offset_thread = None; self._offset_worker = None

        worker.result.connect(on_result)
        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.failed.connect(on_failed)
        worker.cancelled.connect(on_cancelled)
        worker.finished.connect(cleanup)
        worker.failed.connect(cleanup)
        worker.cancelled.connect(cleanup)
        thread.started.connect(worker.run)
        thread.start()

    def _finish_offset_worker(self):
        """Hide BusyDialog and clear pointer after any offset worker outcome."""
        if self._busy_offset:
            try:
                self._busy_offset.finish()
            except Exception:
                pass
            self._busy_offset = None

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
