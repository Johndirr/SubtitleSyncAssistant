# Project Structure and Architecture

This document explains the folder structure, the key modules, and how the application components work together.

## Top-Level Layout

- subtitle_sync_assistant.py
  - Entry-point shim to launch the application. Preferentially calls `ssa.app.main.run()`. If the package import fails (e.g., when run directly), it falls back to creating a minimal QApplication and shows MainWindow.
- LICENSE
  - GPLv3 license text.
- README.md
  - User-focused overview (features, installation, run instructions).
- README_STRUCTURE.md (this file)
  - Developer-focused structure and architecture notes.

## Package: ssa/

The application is organized as a Python package named `ssa` with subpackages for application shell, UI, background workers, services, and utilities.

```
ssa/
|-- app/
|   |-- main.py
|   |-- main_window.py
|   `-- __init__.py
|-- ui/
|   |-- dialogs.py
|   |-- plot_widget.py
|   |-- preview.py
|   `-- __init__.py
|-- workers/
|   |-- analyze.py
|   |-- offset.py
|   `-- __init__.py
|-- services/
|   |-- audio.py
|   `-- __init__.py
|-- utils/
|   `-- timecodes.py
`-- __init__.py
```

### ssa/app/

- main.py
  - Defines `run()`: creates the QApplication, builds and shows `MainWindow`, and enters the Qt event loop.
- main_window.py
  - The main GUI controller (Qt Widget). Orchestrates:
    - File selection for reference media, new media, and subtitle files.
    - Launches analysis in a background thread via `AnalyzeWorker`.
    - Displays two waveform viewers (`MatplotlibPlotWidget`) with subtitle overlays.
    - Provides a pair of tables (reference vs sync) with row selection, editing, shifting, and export to SRT.
    - Integrates offset search via workers in `ssa.workers.offset`.
    - Integrates a Preview Images tool window (see `ssa.ui.preview.PreviewImagesWindow`) accessible via table context menus.
  - Also handles lightweight, local audio playback of selected table row intervals (using `QAudioOutput`).

### ssa/ui/

- plot_widget.py
  - `MatplotlibPlotWidget` is a reusable widget that renders a scrollable mono waveform window using Matplotlib.
  - Features:
    - Renders a fixed-duration horizontal window (e.g., 20 s).
    - Mouse interactions: right-click sets the playhead; left-drag pans.
    - Overlays subtitle intervals, with special highlight for selected rows.
    - Lightweight audio preview from the playhead using Qt audio.
  - Exposes `playingChanged` so only one plot instance plays at a time.
- dialogs.py
  - Common dialogs used by MainWindow:
    - `BusyDialog`: modal, indeterminate progress with optional cancel.
    - `RangeSelectDialog`: input for selecting a start/end range in seconds.
    - `EditSubtitleDialog`: editor for one subtitle row (start, end, text) with validation.
- preview.py
  - `PreviewImagesWindow`: a small, floating tool window that shows two images side-by-side (Reference | New) taken from the loaded media files at specific timestamps.
  - Behavior:
    - Activated from both tables via context menu entry "Show preview images".
    - Images correspond to the start time of the selected row in each table.
    - Auto-updates when selection changes, after editing a row, or after shifting times.
    - Extracts a single frame using `ffmpeg` in a background `QThread` to keep the UI responsive.
    - Frames are cached per (file, time) in 0.5 s buckets to limit repeated extraction.
    - Scales images to fit the window; resizes dynamically on window resize.
    - Provides graceful fallbacks: "(loading...)", "(no image)", or "(preview unavailable)".
    - Cleans up background workers on window close to avoid shutdown warnings.

### ssa/workers/

- analyze.py
  - `AnalyzeWorker` runs in a `QThread` to keep the UI responsive. Responsibilities:
    - Loads reference and new media with pydub.
    - Produces display-friendly arrays (optionally decimated for performance), and retains full-quality audio segments for playback.
    - Parses SRT using `pysrt` and computes per-row (start, end) intervals in seconds.
    - Emits a result dict: display arrays + sample rates, full audio segments, parsed rows, and intervals.
- offset.py
  - Workers that use the BBC `audio-offset-finder` library:
    - `OffsetWorker`: correlates each selected snippet from the new wave against the reference wave to find a global alignment offset per row.
    - `SlidingOffsetWorker`: similar, but uses a per-row shifted reference window derived from a base time span (useful for multi-row searches across a range).
  - Results are emitted per-row and consumed by the main window to populate the "Found offset" column.

### ssa/services/

- audio.py
  - Pure helper functions used by workers and UI:
    - `segment_to_float_array(seg)`: convert pydub `AudioSegment` to mono float array in [-1, 1] along with its sample rate.
    - `stride_decimate(samples, sr, stride)`: reduce array size for efficient display while adjusting effective sample rate.
    - `resample_linear(samples, src_sr, dst_sr)`: simple linear resampler used by offset workers to match sample rates.

### ssa/utils/

- timecodes.py
  - Utilities for formatting and parsing SRT timecodes:
    - `format_seconds(seconds)`: HH:MM:SS,mmm formatting with proper rollovers.
    - `parse_srt(ts)`, `normalize_srt(ts)`, `parse_any(text)`: robust helpers for diverse input formats.

## How the Pieces Work Together

1. The user starts the app (via `subtitle_sync_assistant.py` or directly with `ssa.app.main.run`).
2. In `MainWindow`, the user selects:
   - Reference media (A), New media (B), and a subtitle file (SRT).
3. On "Analyze", `AnalyzeWorker` (in a thread) loads A and B via pydub, builds display arrays, parses the SRT, and reports:
   - Display wave arrays and sample rates for both A and B.
   - Full `AudioSegment` objects for local playback.
   - A list of subtitle rows and their intervals.
4. `MainWindow` receives the result and:
   - Feeds plots with display arrays and SRs (`plot_waveform`).
   - Attaches full audio segments for playback (`set_audio_segment`).
   - Populates both tables (reference and sync) with the subtitle lines.
5. Selection in a table highlights corresponding intervals on the respective plot.
6. Preview Images window:
   - Right-click a table and choose "Show preview images" to open.
   - Shows frames from A (Reference) and B (New) at the start times of the selected rows (one per table).
   - Updates automatically when selection changes, and also after editing a line or shifting times.
   - Frames are fetched with ffmpeg in background threads and cached per 0.5 s bucket.
7. Playback:
   - The plots handle their own play/pause and draw the playhead while streaming audio through `QAudioOutput` with a `QBuffer` (16-bit PCM).
   - Only one plot plays at a time; they notify each other via `playingChanged`.
8. Offset finding (optional):
   - From the sync table, selected rows trigger either `OffsetWorker` or `SlidingOffsetWorker`.
   - Workers resample snippets if needed, call `find_offset_between_buffers`, and compute a per-row delta so start/end times can be adjusted.
   - Results are written to the "Found offset" column with status coloring.
9. Editing & Export:
   - A single row can be edited via `EditSubtitleDialog`.
   - Times can be shifted (selected rows or all rows) and visually marked.
   - Export writes a validated SRT with consecutive indices using `pysrt`.

## Data Types and Conventions

- Display waveform arrays are numpy `float32`, mono or single-channel view (multi-channel averaged to mono).
- Time is expressed in seconds for processing and in `HH:MM:SS,mmm` for UI and export.
- For Qt audio playback, raw PCM bytes (16-bit, signed little-endian) are pushed to `QAudioOutput` using `QBuffer`.
- Preview frames are `QPixmap` objects cached by `(file_path, bucketed_time)`.

## Dependencies and Runtime Assumptions

- Python 3.10+ recommended.
- ffmpeg must be available on PATH for pydub to decode media and for preview frame extraction.
- `audio-offset-finder` is required for offset search features; the app remains usable without it.

## Extending the Application

- Add new services in `ssa/services/` (pure helpers, decoders, transforms).
- Add new workers in `ssa/workers/` to keep the UI responsive for heavy tasks.
- Extend UI with new dialogs in `ssa/ui/` and wire actions in `MainWindow`.
- Reuse `MatplotlibPlotWidget` for additional waveform views; it is self-contained and signals playback state.
- `PreviewImagesWindow` can be reused to show comparisons at arbitrary timestamps; it exposes `show_for_selection()` and `update_images()`.
- Keep business logic in services/workers and leave `MainWindow` to orchestrate UI.

## Coding Notes

- Modules include docstrings and inline comments to aid maintainability.
- Headers include SPDX and GPL notice to keep licensing clear.
- Avoid long blocking operations on the UI thread - prefer `QThread` workers.
- The preview window cancels and joins its threads on close; `MainWindow.closeEvent` also shuts down background work to avoid interpreter shutdown warnings.
