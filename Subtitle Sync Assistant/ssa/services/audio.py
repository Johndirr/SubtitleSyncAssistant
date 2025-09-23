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

"""Audio helpers: pydub <-> numpy conversion and simple resampling.

Pure functions so they can be reused by workers and widgets without Qt.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from pydub import AudioSegment


def segment_to_float_array(seg: AudioSegment) -> Tuple[np.ndarray, int]:
    """Convert an AudioSegment to mono float32 numpy array and return with SR.

    - Mixes down multi-channel by averaging channels.
    - Normalizes to [-1, 1] based on sample width.
    """
    arr = np.array(seg.get_array_of_samples()).astype(np.float32)
    if seg.channels > 1:
        # Re-shape to (n_frames, channels) then average axis=1 -> mono
        arr = arr.reshape((-1, seg.channels)).mean(axis=1)
    # Normalize using peak integer value for the given sample width
    arr /= float(2 ** (8 * seg.sample_width - 1))
    return arr, seg.frame_rate


def stride_decimate(samples: np.ndarray, original_sr: int, stride: int) -> Tuple[np.ndarray, int]:
    """Decimate by taking every Nth sample, returning new array and effective SR.

    Useful for downsampling display-only arrays at load time.
    """
    if stride <= 1:
        return samples, original_sr
    if samples.ndim == 1:
        display = samples[::stride]
    else:
        display = samples[:, ::stride]
    return display, max(1, original_sr // stride)


def resample_linear(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Lightweight linear resampler used by offset workers to match SRs.

    Uses np.interp over a normalized domain [0, 1) to map src -> dst length.
    """
    if src_sr == dst_sr or samples.size == 0:
        return samples
    ratio = dst_sr / src_sr
    new_len = max(1, int(round(samples.shape[0] * ratio)))
    x_old = np.linspace(0.0, 1.0, samples.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)
