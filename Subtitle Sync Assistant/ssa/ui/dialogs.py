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
"""Qt dialogs extracted from the monolith.

Contains BusyDialog, RangeSelectDialog, EditSubtitleDialog.
Logic is preserved; only moved and lightly documented.
"""
from __future__ import annotations

from typing import Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QDialogButtonBox,
    QHBoxLayout,
    QSpinBox,
    QMessageBox,
    QLineEdit,
    QTextEdit,
)


class BusyDialog(QDialog):
    """Modal indeterminate progress dialog with optional Cancel.

    Emits cancel_requested once when the user cancels. The dialog keeps
    showing until the owner calls finish(), so background operations can
    shut down safely.
    """

    cancel_requested = pyqtSignal()

    def __init__(self, parent=None, title="Working", message="Please wait...", cancellable: bool = False):
        """Create the busy dialog.

        cancellable toggles the presence of a Cancel button.
        """
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(360)
        self._cancellable = cancellable
        self._force_close = False
        self._cancel_sent = False
        layout = QVBoxLayout(self)
        self.label = QLabel(message, self)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self.bar = QProgressBar(self)
        self.bar.setRange(0, 0)
        self.bar.setTextVisible(False)
        layout.addWidget(self.bar)
        if cancellable:
            btns = QDialogButtonBox(QDialogButtonBox.Cancel)
            btns.rejected.connect(self._on_cancel)
            layout.addWidget(btns)

    def _on_cancel(self):
        """Emit cancel_requested once and update message to 'Cancelling ...'."""
        if not self._cancel_sent:
            self._cancel_sent = True
            self.cancel_requested.emit()
        self.set_message("Cancelling ...")

    def set_message(self, text: str):
        """Update the status label text."""
        self.label.setText(text)

    def finish(self):
        """Allow dialog to really close now (worker done)."""
        self._force_close = True
        self.close()

    def closeEvent(self, event):
        """On Cancel, keep dialog open and notify once; else close normally."""
        if self._force_close or not self._cancellable:
            return super().closeEvent(event)
        if not self._cancel_sent:
            self._cancel_sent = True
            self.cancel_requested.emit()
        self.set_message("Cancelling ...")
        event.ignore()


class RangeSelectDialog(QDialog):
    """Dialog to enter a start/end range as HH:MM:SS values."""

    def __init__(self, parent, default_start_sec: float, default_end_sec: float):
        """Create the dialog and initialize spin boxes.

        Note: The original behavior initializes spin boxes to zero rather
        than the suggested defaults.
        """
        super().__init__(parent)
        self.setWindowTitle("Reference Range")
        self.setModal(True)
        self.setMinimumWidth(360)

        # Always initialize to zero (ignore provided defaults)
        sh = sm = ss = 0
        eh = em = es = 0

        layout = QVBoxLayout(self)

        # Start time row
        start_row = QHBoxLayout()
        start_row.addWidget(QLabel("Start (HH:MM:SS):"))
        self.start_h = QSpinBox(); self.start_h.setRange(0, 999); self.start_h.setValue(sh)
        self.start_m = QSpinBox(); self.start_m.setRange(0, 59);  self.start_m.setValue(sm)
        self.start_s = QSpinBox(); self.start_s.setRange(0, 59);  self.start_s.setValue(ss)
        for w in (self.start_h, self.start_m, self.start_s):
            w.setFixedWidth(60)
        start_row.addWidget(self.start_h)
        start_row.addWidget(QLabel(":"))
        start_row.addWidget(self.start_m)
        start_row.addWidget(QLabel(":"))
        start_row.addWidget(self.start_s)
        start_row.addStretch()
        layout.addLayout(start_row)

        # End time row
        end_row = QHBoxLayout()
        end_row.addWidget(QLabel("End (HH:MM:SS):"))
        self.end_h = QSpinBox(); self.end_h.setRange(0, 999); self.end_h.setValue(eh)
        self.end_m = QSpinBox(); self.end_m.setRange(0, 59);  self.end_m.setValue(em)
        self.end_s = QSpinBox(); self.end_s.setRange(0, 59);  self.end_s.setValue(es)
        for w in (self.end_h, self.end_m, self.end_s):
            w.setFixedWidth(60)
        end_row.addWidget(self.end_h)
        end_row.addWidget(QLabel(":"))
        end_row.addWidget(self.end_m)
        end_row.addWidget(QLabel(":"))
        end_row.addWidget(self.end_s)
        end_row.addStretch()
        layout.addLayout(end_row)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._start_seconds = None
        self._end_seconds = None

    def _on_accept(self):
        """Validate and store start/end seconds then accept dialog."""
        start_sec = self.start_h.value()*3600 + self.start_m.value()*60 + self.start_s.value()
        end_sec = self.end_h.value()*3600 + self.end_m.value()*60 + self.end_s.value()
        if end_sec <= start_sec:
            QMessageBox.warning(self, "Invalid Range", "End must be greater than Start.")
            return
        self._start_seconds = start_sec
        self._end_seconds = end_sec
        self.accept()

    @staticmethod
    def get_range(parent, default_start_sec: float, default_end_sec: float) -> Tuple[Optional[int], Optional[int], bool]:
        """Convenience static method to run the dialog and return values."""
        dlg = RangeSelectDialog(parent, default_start_sec, default_end_sec)
        ok = dlg.exec_() == QDialog.Accepted
        return dlg._start_seconds, dlg._end_seconds, ok


class EditSubtitleDialog(QDialog):
    """Dialog to edit a subtitle row's start/end and text."""

    def __init__(self, parent, start_text: str, end_text: str, subtitle_text: str):
        """Create the dialog prefilled with the provided row contents."""
        super().__init__(parent)
        self.setWindowTitle("Edit Subtitle Line")
        self.setModal(True)
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)

        # Start time
        row_start = QHBoxLayout()
        row_start.addWidget(QLabel("Start (HH:MM:SS,mmm):"))
        self.le_start = QLineEdit()
        self.le_start.setText(start_text)
        row_start.addWidget(self.le_start)
        layout.addLayout(row_start)

        # End time
        row_end = QHBoxLayout()
        row_end.addWidget(QLabel("End (HH:MM:SS,mmm):"))
        self.le_end = QLineEdit()
        self.le_end.setText(end_text)
        row_end.addWidget(self.le_end)
        layout.addLayout(row_end)

        # Text
        layout.addWidget(QLabel("Text:"))
        self.te_text = QTextEdit()
        self.te_text.setPlainText(subtitle_text)
        layout.addWidget(self.te_text)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.result_start = None
        self.result_end = None
        self.result_text = None

    def _normalize_time(self, t: str) -> Optional[str]:
        """Normalize to HH:MM:SS,mmm; accept HH:MM:SS.mmm and pad ms to 3 digits."""
        t = (t or "").strip()
        if not t:
            return None
        # Accept variants like HH:MM:SS.mmm or HH:MM:SS,mmm
        if "." in t and "," not in t:
            parts = t.rsplit(".", 1)
            if len(parts[-1]) in (1, 2, 3) and parts[-1].isdigit():
                t = parts[0] + "," + parts[1].ljust(3, '0')
        # Ensure milliseconds
        if "," not in t and t.count(":") == 2:
            t += ",000"
        # Pad milliseconds to 3
        if "," in t:
            a, b = t.split(",", 1)
            if not b.isdigit():
                return None
            b = b[:3].ljust(3, '0')
            t = a + ',' + b
        return t

    def _parse_to_seconds(self, t: str) -> Optional[float]:
        """Parse HH:MM:SS,mmm string into seconds; return None on failure."""
        try:
            h, m, rest = t.split(":")
            s, ms = rest.split(",")
            return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0
        except Exception:
            return None

    def _on_accept(self):
        """Validate times and set result values before accepting dialog."""
        start_raw = self._normalize_time(self.le_start.text())
        end_raw = self._normalize_time(self.le_end.text())
        if not start_raw or not end_raw:
            QMessageBox.warning(self, "Invalid", "Invalid start or end time format.")
            return
        s = self._parse_to_seconds(start_raw)
        e = self._parse_to_seconds(end_raw)
        if s is None or e is None:
            QMessageBox.warning(self, "Invalid", "Could not parse start/end times.")
            return
        if e <= s:
            QMessageBox.warning(self, "Invalid", "End must be greater than Start.")
            return
        self.result_start = start_raw
        self.result_end = end_raw
        self.result_text = self.te_text.toPlainText()
        self.accept()

    @staticmethod
    def edit(parent, start_text: str, end_text: str, subtitle_text: str):
        """Convenience method to edit a row and return (start, end, text, ok)."""
        dlg = EditSubtitleDialog(parent, start_text, end_text, subtitle_text)
        ok = dlg.exec_() == QDialog.Accepted
        return dlg.result_start, dlg.result_end, dlg.result_text, ok
