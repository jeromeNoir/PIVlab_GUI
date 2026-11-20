# PIVlab processing — desktop GUI (version 1)

This python GUI was created by Jerome Noir (jnoir@ethz.ch).
Created: 18 November 2026. Last modified: 20 November 2026.

A Qt desktop application (PySide6 or PyQt5, through matplotlib's
`qt_compat`) for
looking at and processing one PIVlab run at a time. How it is built, and
how to modify it or write scripts for it (also with Claude or another AI),
is in [`Claude_for_Developer.md`](Claude_for_Developer.md).

| file | role |
| --- | --- |
| `PIV_GUI.py` | the window: menus and menu toolbar, parameter panels, figure tabs, figure list, separate windows, sessions |
| `piv_gui_lib.py` | the library (no Qt): default parameters, folders, acquisition log, PIVlab `.mat` reader, the loaded run (`PIVData`), background image, figures and their data |
| `gui_variables.py` | **View > Show all variables** window (adapted from the DTS GUI) |
| `Claude_for_Developer.md` | the guide for AI assistants (and developers): modifying the GUI, writing scripts for Advanced > Python script |
| `Example_Scripts/` | example scripts for Advanced > Python script |
| `gui_export.py` | **Save data** writer, copied unchanged from the DTS GUI (`DTS/GUI/gui_export.py`): npz, NetCDF, HDF5, txt, csv, MATLAB |
| `requirements.txt` | the Python packages needed (`pip install -r requirements.txt`) |

## Publishing on GitHub

The folder is published with **`update_github.command`** (in the folder,
not published itself, nor `TODO.md`: see `.gitignore`).

The first time:

1. On GitHub, create a new **public** repository (e.g. `piv_gui`) --
   **empty**: no README, no .gitignore, no licence (they would clash with
   the files of the folder).
2. Double-click `update_github.command` in the Finder (the first time
   macOS may ask to confirm: right click > Open). It asks the address of
   the repository (e.g. `https://github.com/jeromeNoir/piv_gui.git`),
   makes the folder a git repository linked to it, then publishes every
   file.
3. When git asks for a password, give a **personal access token** (GitHub
   > Settings > Developer settings > Personal access tokens, with the
   *repo* scope), not your GitHub password; macOS keeps it in the
   keychain for the next times.

Then, after every change: double-click `update_github.command` again (or
run `./update_github.command "what changed"` in a Terminal). It writes
today's date as *Last modified* at the top of this file, records every new,
changed or deleted file in a commit (asking for a short description) and
pushes it to GitHub.

## Starting it

```
cd GUI
~/anaconda3/bin/python PIV_GUI.py
~/anaconda3/bin/python PIV_GUI.py my.pivsession.json    # reopen a session
```

The GUI needs numpy, scipy, matplotlib and a Qt binding: PySide6 (the one
the DTS GUI uses) or PyQt5. Both are tested. Other modules are needed only
for some files:

- `h5py`: MATLAB v7.3 `.mat` files and Save data in HDF5;
- `netCDF4`: Save data in NetCDF;
- `ffmpeg` (on the PATH, e.g. from Homebrew or conda), or the Python package
  `imageio-ffmpeg` (`pip install imageio-ffmpeg`, it brings its own ffmpeg):
  MP4 movies (Maps and Quiver > Make Movie). Without either, the movies are
  animated GIFs.

The GUI does not depend on the `Notebook` folder.

## Window

The window is laid out like the DTS processing GUI.

- **Menus**:
  - **File**: Load data, New session, Load session, Save session, Apply
    session steps to another dataset, Batch processing, Save variables,
    Quit.
  - **Pre-processing**: Calibration, Define ROI, Mask.
  - **Processing**: Maps and Quiver, Time series, Profiles.
  - **Advanced**: Python script.
  - **Spectral analysis**: FFT in time, FFT in space, 2D FFT, f-Filter,
    k-Filter.
  - **View**: Show / hide the log (Ctrl+L), Clear the log, Show all
    variables.

  The menus are in the menu bar and are also repeated as a toolbar of
  drop-down buttons at the top of the window and of every separate figure
  window. This matters on macOS, where the menu bar sits at the top of the
  screen. Choosing an entry opens that step's parameters on the left and
  brings its figures forward.
- **Left**: the parameters of the current step.
- **Bottom**: the **log** (as in the DTS GUI), below the three panels;
  drag the handle above it to resize it.
  - Each line starts with the time. It records the start (Python, Qt
    binding, numpy, matplotlib, missing modules), each Load (run,
    calibration / ROI / mask files, velocity) and its result (fields,
    grid, velocity and valid vectors, calibration, time base and its
    source), warnings (no log file, no calibration, velocity not in the
    file), the calibration steps (segment, Apply, Save, Load calibration, Reset,
    time base applied or back to the log,
    time base saved), calibration / ROI / mask files saved or loaded,
    data saved,
    sessions, and errors.
  - Anything printed (stdout, stderr) goes to the log as well as to the
    terminal. An uncaught error (e.g. in a button) is written in the log
    with its traceback, instead of being lost; the log is shown again if
    it was hidden.
  - **View > Show / hide the log** (Ctrl+L) and **View > Clear the log**.
- **Data in memory**: only the uncalibrated data of the velocity loaded
  are kept (px, px/frame), once, for every step; the calibration factors
  are applied when the data are used (see *Adding a step*).
- **View > Show all variables**: a window listing the variables kept in
  memory, as in the DTS GUI, **grouped by step** like File > Save
  variables (see *Save variables* for the names): each with a
  description, its format, its size and its memory; dictionaries open to
  their entries, and the entries have their own short description.
  **Search** keeps the variables and entries whose name, step or
  description hold all the keywords; the **full description** of the row
  selected is shown below the list; **Expand all**, **Collapse**,
  **Refresh**; right click > **Show value** opens
  the value (arrays with min, max, mean, NaN count and a shortened
  content).
- **Centre**: the ticked figures, as tabs, each with the matplotlib toolbar
  (zoom, pan, cursor readout, save image). Closing a tab unticks the figure.
- **Resizing**: the central area has no minimum size (as in the DTS GUI).
  Drag the handle on its left or on its right to narrow or widen it. A
  figure keeps its own size (about 75 px per inch of the matplotlib
  figure): when the area is smaller than the figure, scroll bars appear,
  and when it is larger the figure grows with it. The toolbar then shows
  an arrow for the buttons that do not fit.
- **Right**: the list of figures, grouped by step, each with a tick box:
  - **tick**: the figure is shown as a tab; untick to hide it. A step
    heading ticks or unticks all the figures of its step.
  - **click** on a name: the figure comes forward (and is ticked).
  - **double click** on a name: the figure is shown in the central panel,
    brought back from its separate window if it has one.
  - **right click** on a name, a figure or a table:
    - **Focus**: tick the figure and bring it forward.
    - **Open in separate window**, or **Show in the central panel** when it
      is already in one. The separate window repeats the menus at its top;
      closing it brings the figure back to the centre.
    - **Save data...**: the data of the figure, in the format you choose
      (npz, NetCDF, HDF5, txt, csv, MATLAB), as in the DTS GUI. For the
      Maps and Quiver figure this is the grid x, z, the field shown (u, v,
      |u|), the divergence, vorticity and kinetic energy, the vector field,
      the ROI, the mask, the frame and its time, the time base and the
      units. For the run
      information it is the table.
    - **Update**: redraw the figure with the current parameters of its
      panel. For the run information, the run is read again.

Version 1 implements **File > Load data**, **Pre-processing > Calibration** and
sessions. The other menu entries open an empty panel.

## Load data

1. **Root folder**: the folder holding the run folders. It is remembered
   between launches.
2. **run folder**: a drop-down list of the run folders in the root, one run
   at a time. A folder without the PIV file (`PIVlab_results_uncalibrated.mat`)
   is marked "(no ...)" but stays selectable: Load asks for the `.mat` (see
   *Files of a run* below).
3. **No parameter file any more.** The run is loaded with the default
   parameters (`PIVlab_results_uncalibrated.mat`, `acquisition_log.txt`,
   ROTATE 0); the calibration, the ROIs / lines and the masks come from the
   files saved by their steps, each loaded only when its box is ticked:
   - **Calibration** (tick box, file selector, Browse...; off by default):
     the calibration files `*.pivcalib.json` of the root folder (the
     default folder, where Save proposes them), of `<run>/PostProcessing`
     and of the run folder. Ticked: the file is loaded,
     the calibration applied (XSCALE / YSCALE, rotation, time base: the
     run in m, m/s, s) and the **Calibration step run from it** (image,
     segment, length in mm). Off: no file is loaded and the run is
     uncalibrated (px, px/frame, frame numbers; ROIs, which are in m, are
     then not listed).
   - **ROI** (same, `*.pivroi.json`): ticked, the **Define ROI step is run
     from the file**: the ROIs and lines with their names and ticks (active
     or not), the figure, then Apply (the ROI quantities of the active
     ROIs). Needs the calibration.
   - **Mask** (same, `*.pivmask.json`): ticked, the **Mask step is run from
     the file**: the masks, the selection areas, the masks shown and the
     mask applied.
   - **velocity** (drop-down list, below the Mask row): the
     velocity to load.
     - *Validated*: the PIVlab filtered velocity.
     - *Unvalidated*: the original velocity.
     - *Unvalidated, rejected = NaN*: the original velocity with NaN
       where `typevector_filt` ≠ 1 (the vectors PIVlab rejected).

     The default is *Validated*. **Only the velocity selected is read and
     kept in memory**: Load reads from the `.mat` only the variables it
     needs (Validated: `u_filt`, `v_filt`, and `u`, `v` for their NaN mask,
     dropped right after; Unvalidated: `u`, `v`; Unvalidated, rejected =
     NaN: `u`, `v`, `typevector_filt`). Another choice is read at the next
     **Load** (the status bar says so). After a Load, the velocities that
     are not in the `.mat` are greyed out. A session keeps the choice.

   The file dialogs (Browse..., the Load / Save buttons of the steps,
   the sessions) filter on `*.json`: the macOS dialog greys out the files
   of a double-extension pattern such as `*.pivcalib.json`. A name saved
   without the full extension gets it (`name` or `name.json` ->
   `name.pivcalib.json`), so Load data finds it.

   When the GUI opens (and after File > New session), the boxes are off
   and the Calibration, Define ROI and Mask steps are empty. Nothing is
   restored from `run_information.json`, which is only a record.
4. **Load**, the button below the selection, is disabled until a run
   folder is chosen. It first finds the files of the run (the `.mat` and the
   log, asking for them when they are missing: see *Files of a run*), then
   reads the run in the background, so the window stays responsive. It
   then creates the run's **post-processing folder**
   `<run>/PostProcessing` if it does not exist (the folder the notebooks
   use), builds the run information table and **saves it in
   `<run>/run_information.json`**. **Load data only loads**: no figure is
   drawn (the maps are in **Processing > Maps and Quiver**). The entry
   below appears in the figure area, listed under *Load data* on the
   right:
   - **Run information**: a table with these sections:
     - *Run*: run folder, the **calibration, ROI and mask files** the
       steps were run from -- loaded (Load data, Load calibration / ROI
       selection / mask, a session) or saved (Save of the step) --, PIV
       file and its size,
       acquisition log file (or "skipped"), post-processing folder, run
       information file.
     - *Time base*: number of PIV fields, the time between frames dt (pulse
       separation), the camera frame rate, the frames per PIV field, the PIV
       field sampling frequency, the time between fields, the record length,
       the Nyquist frequency and the frequency resolution.
     - *Space*: grid, vector spacing, extent, ROTATE, calibration, ROI.
     - *Calibration*: XSCALE / YSCALE, shown calibrated or not, and the
       calibration record of the run: `CALIB_IMAGE`, `CALIB_SEGMENT_PX`
       (`[[x0, y0], [x1, y1]]` in px), `CALIB_LENGTH_PX`,
       `CALIB_LENGTH_MM` (set by Pre-processing > Calibration > Apply).
     - *Velocity*: the vector fields of the `.mat`, the velocity loaded,
       the valid vectors, the median and maximum |u|.
     - *Acquisition log*: every column of the log's last row.

     **The table is saved again in `run_information.json` every time
     something it shows changes**: a calibration applied, the time base
     (Apply, Manual unticked), the Display choice calibrated /
     uncalibrated, an ROI or mask applied, or any new Load.
     The JSON holds the table as shown (`run_information`) and,
     machine-readable, the files used (`files`, with `calibration_file`,
     `roi_file`, `mask_file`), the time base
     (`time_base`), the calibration with its CALIB_ record
     (`calibration`), the masks (`mask`: the masks and their areas, the
     selection areas, the mask applied) and the ROI fields
     (`roi_fields`).

     Right-click the table to save it (CSV or text), copy it (tab-separated)
     or open it in a separate window.

### Time base (acquisition log; kept in memory)

The last data row of the log gives two values:

- `pulse_sep` [µs] is the separation of the two images of a pair. This is
  **dt**, the time base of the velocity.
- `cam_fps` is the camera frame rate. PIVlab pairs the frames per PIV
  field (2) into one velocity field, so the **PIV field rate is
  `cam_fps / frames per PIV field`**. The field times are
  `index × TSCALE`, with `TSCALE = frames per PIV field / cam_fps`.

The time base is **kept in memory**: the frames per PIV
field (default 2), the pulse separation and the camera fps used without a
log (defaults 1 s and 2 Hz), and whether the time base is manual are kept
in memory by the GUI. They are set in **Pre-processing > Calibration >
Time base**, kept until the GUI is closed (and in a saved session), and
shown in the **Run information** (*Time base*: where it comes from --
*log*, *manual* or *memory (no log)* --, frames per PIV field, pulse
separation, camera frame rate, PIV field rate, TSCALE, ...). If the log is
missing or unreadable, or the time base is manual, the values in memory
are used.

### The parameters of a run (in memory; no parameter file)

| key | meaning |
| --- | --- |
| `PIV_FILENAME` | PIVlab result file in each run (`PIVlab_results_uncalibrated.mat`) |
| `LOG_FILENAME` | acquisition log in each run (`acquisition_log.txt`) |
| `XSCALE`, `YSCALE` | m/px (calibration file or Calibration > Apply); `null` means uncalibrated, and the run is shown in px and px/frame |
| `ROTATE` | 0, 90, 180 or −90 deg (from the calibration file; 0 without) |
| `PTS_ROI`, `LINES` | the ROIs (rectangles) and the lines in m, `{name: [[x1, z1], [x2, z2]]}` (ROI file or Define ROI > Apply) |
| `MASKS`, `MASK_APPLIED`, `MASK_SELECTION` | the masks (`{name: {"colour", "areas": [{"name", "kind", "points_px"}]}}`, px of the run's grid), the mask applied, the selection areas (mask file or the Mask step) |
| `CALIB_INFO` | the calibration record (`CALIB_IMAGE`, `CALIB_SEGMENT_PX`, `CALIB_LENGTH_PX`, `CALIB_LENGTH_MM`) |
| `L`, `nu` | length scale [m], kinematic viscosity [m²/s] (`null`) |

They are carried over when a step loads the run again (calibration, time
base) and kept in a saved session; View > Show all variables lists them
(*parameters*). The `param_GUI*.json` files of older versions are no
longer read or written (a session saved with one still loads it).

The `.mat` reader accepts classic MATLAB files and v7.3/HDF5 files, with the
velocity stored either as 3-D arrays or as cell arrays (one frame per cell).
The axes follow the notebooks' convention: x to the right, z upward, V
positive upward.

### Files of a run: missing `.mat` or log

Before reading, Load looks for `PIVlab_results_uncalibrated.mat` and
`acquisition_log.txt` (`PIV_FILENAME`, `LOG_FILENAME`) in the run folder.

- **No `.mat`**: if the run's `run_information.json` records a `.mat`
  chosen before and it still exists, that one is used. Otherwise a message
  asks you to **select the PIVlab `.mat` file** of the run (a file dialog
  opens in the run folder); cancelling it cancels the Load.
- **No log**: if `run_information.json` records a log chosen before, or
  that the log was skipped (with the values entered then), that is used.
  Otherwise a window asks you to:
  - **Select a log file...**: a file dialog; the log chosen is read as
    usual;
  - **Skip**: a window then asks for the **pulse separation** [s] and the
    **camera FPS** [Hz] (pre-filled with the values in memory). The run is
    read with them (time base "entered by hand (no log)"), and they go to
    the time base in memory (Pre-processing > Calibration > Time base);
  - **Cancel**: no Load.

The choices are written in `run_information.json` (`files`, `time_base`),
so they are asked only once per run. A log that exists but cannot be read
is reported after the Load (**Go to Calibration** opens the Calibration
panel, where the time base can be typed).

## Pre-processing > Calibration

The panel on the left, from top to bottom:

1. **Calibration image**: a selector of the image files (tif, png, jpg, bmp)
   and `background*.mat` files of the run folder and of the root folder,
   with the files named `*calib*` first; **Browse...** picks any other
   file. The selected image is shown in the figure **Calibration image**
   (listed under *Calibration*), in pixels and rotated by `ROTATE`, as in
   `Calibration_PIV.py`. A 90° rotation keeps the pixel distances.
   Opening the Calibration step shows the selected image if it is not
   shown yet. **Choosing an image** (list or Browse...) updates the run
   information at once (*Calibration*: `CALIB_IMAGE`), and -- as for
   every change of the run information -- saves `run_information.json`.

   **Rotation** (below the image selector, in deg): the rotation of the
   PIV fields and of the images, 0, 90, 180 or −90 (another value goes to
   the nearest one; 270 = −90). Changing it **rotates the Calibration
   image figure at once** -- the segment drawn turns with the image, so
   it stays on the same features and keeps its length -- and loads the
   run again rotated
   (the grid and the velocity turned, as the notebooks' `rotate_fields`).
   It is shown and updated in the run information (*Space*: ROTATE;
   *Calibration*: rotation) and kept in the calibration file (Save), so
   Load calibration and Load data apply it again; without a calibration
   file a Load uses 0.
2. **Draw segment** and **Reset** (buttons, side by side).

   **Reset** deletes the segment from the figure and clears its length in
   px and in mm. Nothing else changes: the selection stays on or off, and
   the applied calibration is kept.

   **Draw segment** starts the selection on the image as it is shown. The image
   is drawn only if it is not on screen yet, so your zoom is kept, and the
   toolbar zoom or pan tool is switched off. While the button is on (it reads *Drawing... (click to stop)*):
   - Hold the left button and drag: the segment starts where you press and
     ends where you release.
   - Press on one of its two ends (circles) and drag to move that end.
   - A simple click without dragging keeps the current segment.
   - If you switch the toolbar zoom or pan tool on again, the mouse does
     not draw and the status bar says so.

   Click the button again to stop; Apply also stops it. When it is off,
   clicks on the image do nothing, so you can zoom and pan freely.
3. **Spatial calibration (px -> m)**:
   - **Segment length**: the length of the segment in px, with its two
     ends. It is updated while you drag.
   - **Length in mm**: the real length of the segment, which you type.
   - **Scale**: the resulting XSCALE = YSCALE in m/px (and px/mm), updated
     as you draw or type.
   - **Apply**: computes the scaling and loads the run again with it (m,
     m/s, s). It is kept in memory only -- **nothing is written in a
     parameter file** (there is none any more); **Save** keeps it in a
     calibration file.
   - **Save** (next to Apply): writes everything needed to run the step
     again into a calibration file `<name>.pivcalib.json` (proposed in the
     root folder: `<root>/<run>.pivcalib.json`): the run, the calibration image and the rotation, the segment (end points
     in px), its length in px and in mm, the scale it gives and the scale
     applied (XSCALE, YSCALE in m/px) and the time base (Manual or not,
     frames per PIV field, pulse separation, camera fps, and the time base
     in use).
   - **Load calibration...** (at the top of the panel) reads such a file
     and **runs the step again** with it: the image is selected (if it
     still exists; otherwise the segment is used without it), the segment
     and the length in mm are set, the time base is set in memory (Manual
     and its values, or the log), then **Apply** calibrates the loaded run
     (loaded again once, with the time base). A file with scales but no
     segment applies its XSCALE / YSCALE. Without a loaded run the fields
     are only filled. The file used is shown below the button; the log
     records saves and loads. **File > Load data** can load the same file
     with the run (Calibration tick box): the step is then run from it at
     the Load.
   - Apply records the calibration of the run -- `CALIB_IMAGE`,
     `CALIB_SEGMENT_PX`, `CALIB_LENGTH_PX`, `CALIB_LENGTH_MM` -- in the
     run's parameters in memory (`CALIB_INFO`) and in the run information
     (table and `run_information.json`).
   - **Applied**: the calibration used by the loaded run.

   After Apply, the figure title gives the applied value. When the GUI
   opens (or after File > New session) the step is empty: nothing is taken
   from the run information.
4. **Time base** (kept in memory):
   - **Manual** (tick box): use the values typed below instead of the
     log, exactly as for a run without a log. It is always ticked (and
     cannot be unticked) when the run has no log. Unticking it after a
     manual time base was applied goes back to the log at once (the run is
     loaded again).
   - **Frames per PIV field** (default 2).
   - **Pulse separation in s**: the time between frames A and B, used to
     convert px/frame to px/s.
   - **Camera FPS in Hz**. The line below gives `TSCALE` = frames per PIV
     field / camera fps (2 / camera fps), the time between PIV fields used
     for the timestamps, and the PIV field frequency.
   - **Apply**: active only when Manual is ticked. It keeps the three
     values in memory (manual when the run has a log, so the log is not
     used) and loads the run again with them. Nothing is written in a
     file. The values stay in use for the next Loads until
     Manual is unticked or the GUI is closed.


   Without Manual, the fields show the log's values and are read-only.

Save data on the Calibration image writes the image path, ROTATE, the
segment (px), its length in px and in mm, and XSCALE. The session also keeps
the calibration image, the segment and the length in mm.

## Pre-processing > Define ROI

The ROIs (rectangles, `PTS_ROI`) and the lines for the extraction of data
(segments, `LINES`), drawn on a figure. Everything is
in m, in the calibrated view (the rule of every step), so the run must be
calibrated (XSCALE / YSCALE); otherwise Add is disabled and the panel says
so. The lists are empty when the GUI opens (and after File > New session).
The panel on the left, from top to bottom:

0. **Load ROI selection...** (top of the panel): reads an ROI file
   (`.pivroi.json`, made with Save) and runs the step from it: the ROIs and
   lines (names, coordinates, ticks), the figure and its display, then
   Apply. **File > Load data** can load the same file with the run (ROI
   tick box). The file used is shown below the button.
1. **Figure** -- **Select from**: the figure to select from, shown in the
   central panel as *ROI selection*:
   - **Current Frame**: the frame chosen in Load data (*Frame* of its
     Display, or the time mean / rms), |u| as a colour map with the
     vectors. It follows the Load frame.
   - **Select Image**: an image (image files and `background*.mat` of the
     run and root folders, or **Browse...**), placed on the PIV grid and
     rotated by ROTATE, as the background of Load data.
2. **ROI area**: the list of the ROIs (name, x range, z range).
   - Each ROI has a **tick box**: ticked ROIs are drawn on the figure
     (dashed); the selected one is drawn thicker.
   - **Each ROI and each line has its own colour** (shown as a small square
     in front of its name): red, orange, cyan, white, pink, light green,
     yellow, brown for the ROIs; magenta, blue, black, green, purple,
     olive, grey, light orange for the lines (no colour is shared between
     the two lists; after 8, the colours repeat). The colour follows the
     place in the list, so unticking an ROI does not change the others.
   - **One name per ROI, its place in the list**: `ROI01`, `ROI02`, ...
     -- the same xx as its fields `<velocity>_ROIxx` and quantities
     `<quantity>_ROIxx`, everywhere in the GUI (tables, figures, time
     series, files). ROI names cannot be edited. ROIs of a file with other
     names (`"wake"`, ...) are renamed by their place (the log lists the
     renaming).
   - **Add ROI**: drag a rectangle on the figure with the left button held
     (Esc cancels; a click without dragging is ignored). The zoom is kept,
     and the toolbar zoom / pan tool is switched off. The new ROI is the
     last of the list (`ROI<n+1>`).
   - **Remove ROI**: removes the selected ROI; the ROIs after it move up
     one place and are renamed (ROI03 -> ROI02, ...). Its quantities
     `<quantity>_ROIxx` are deleted at once, and those of the ROIs that
     moved are renamed with them (`vorticity_ROI03` -> `vorticity_ROI02`),
     not calculated again.
   - Adding or removing an ROI updates the loaded run at once (its ROI
     fields, Maps and Quiver, Profiles, Time series).
3. **ROI Line**: the same for the lines (name, start, end): **Add line**
   draws a segment (drag from one end to the other; **hold Shift** while
   dragging to keep it horizontal or vertical -- 0, 90, 180 or 270 deg on
   the figure, whichever is nearer), each line in its own
   colour; named `LINE<n>`, n one more than the highest number used; a
   line can be renamed (double click; a name already used gets its number
   incremented, no `_2` suffix).
4. **Apply** and **Save** (side by side).

   **Apply** uses the ROIs (`PTS_ROI`) and the lines (`LINES`) of the two
   lists -- all of them, ticked or not -- for the loaded run, as
   `{name: [[x1, z1], [x2, z2]]}` in m, **in memory: nothing is written in
   a parameter file** (there is none any more). The loaded run follows
   without reading it again: the ROI list of Maps and Quiver, its figure
   (ROIs, and lines on the full view) and the run information (saved
   again in `run_information.json`). Above the buttons: *Not applied yet*
   while the lists were changed, and the number of ROI quantities in
   memory.

   **Save** first **applies** the lists (as Apply: the loaded run, the ROI
   and line quantities), then writes an ROI file `<name>.pivroi.json`
   (proposed in the root folder: `<root>/<run>.pivroi.json`) with everything needed to
   generate the step again: the ROIs and lines with their names,
   coordinates (m) and ticks (active or not), `PTS_ROI`, `LINES`, whether
   they were applied, the figure (source, image, display). Load ROI
   selection (top of the panel) and File > Load data read it.

   Then, for each **active ROI** (ticked in the list; its field
   `<velocity>_ROIxx`, the velocity extracted in the ROI), Apply creates the velocity fields **`U_ROIxx`
   and `V_ROIxx`** (calibrated, mask applied) and calculates from them
   alone the **velocity norm, divergence, vorticity and kinetic energy**
   for every frame, kept in memory as `velocity_norm_ROIxx`,
   `divergence_ROIxx`, `vorticity_ROIxx`, `kinetic_energy_ROIxx` (xx: the
   number of the ROI,
   its place in `PTS_ROI`). They are computed on the ROI's own grid, so the
   derivatives are one-sided at its edges, in calibrated units (m/s, 1/s,
   m²/s²), with the mask applied. Nothing is calculated for the lines,
   nor for an ROI not ticked (the log lists them); the quantities already
   calculated for an ROI that is unticked later are kept (not calculated
   again when it is ticked again).
   - **Only the active ROIs (ticked here) are offered** in every ROI
     drop-down menu and list of the other steps (Maps and Quiver,
     Profiles, Time series, FFT in time), and only the ticked lines in
     Profiles; they follow the ticks at once.
   - Apply first checks which of them exist: an existing one is **not
     calculated again**. It is calculated again only if its ROI changed
     (rectangle, or number after an ROI above it was removed) or the mask
     applied changed. Quantities of ROIs no longer in the lists are deleted.
   - The log lists the quantities calculated, kept and deleted; the run
     information (*ROI quantities*, `run_information.json`) and *Show all
     variables* (`roi_derived_quantities`) list them.
   - **Every step of the Processing and Spectral analysis menus uses
     these stored quantities for the ROI selected, and never calculates
     them again** (Maps and Quiver, Profiles, Time series, and the movies).
     If one is missing (Apply not pressed yet) or no longer fits (the mask
     changed), the ROI quantities are calculated once and stored, as by
     Apply. Only the full field (*Full*) and the uncalibrated display
     calculate on the fly.
   - **Along the lines too**: Apply also calculates, for each **ticked
     line**, the same six quantities (U, V, velocity norm, divergence,
     vorticity, kinetic energy) **along the line, for every frame** --
     in the full field (*Full*: computed on the full grid, mask applied)
     and in **each active ROI the line crosses** (the ROI's own
     `<quantity>_ROIxx`, at the points of the line inside the ROI) --,
     sampled about one point per vector spacing (bilinear). They are kept
     in memory as `<quantity>_<LINE>_<Full or ROIxx>` (e.g.
     `vorticity_LINE1_ROI01`: the values (points × frames), the distance
     along the line and the points), listed in *Show all variables*
     (`line_derived_quantities`). An existing one is not calculated again
     (only if its line, its ROI or the mask changed); those of lines or
     ROIs removed are deleted. **Processing > Profiles uses them and does
     not calculate them** (the figure, its data, the movie); a line with
     none yet (Apply not pressed) gets them calculated once and stored.

After a step loads the run again (calibration, time base) the lists and
their ticks are kept (and the ROI and line quantities calculated
again). The session keeps the lists, the figure choice and the image.

## Pre-processing > Mask

Masks are built from areas drawn on a figure. **The mask applied** -- the
velocity in it is set to **NaN** for every step (Maps and Quiver, the ROI
fields and all the later steps) -- is chosen in **Processing > Maps and
Quiver > Mask**. A mask is applied when the data are read -- the data stored in
memory are not changed -- so it can be changed or removed at any time. The
step is empty when the GUI opens (and after File > New session). The
panel on the left, from top to bottom:

0. **Load mask...** (top of the panel): reads a mask file
   (`.pivmask.json`, made with Save) and runs the step from it: the masks
   (they replace those in memory), the selection areas, the masks shown,
   the mask applied and the figure. **File > Load data** can load the same
   file with the run (Mask tick box).
1. **Masks**: the list of the masks already built (name with its colour,
   number of areas, number of masked vectors).
   - Each mask has a **tick box**: the ticked masks are shown on the figure
     as shaded areas, **each with its own shading colour** (a legend gives
     the names).
   - Double click a name to rename it; **Remove mask** removes the selected
     one (if it was applied, no mask is applied any more).
   - **Save** (it replaces *Update parameter file*): writes a mask file
     `<name>.pivmask.json` (proposed in the root folder:
     `<root>/<run>.pivmask.json`) with everything needed to
     generate the step again: every mask (its areas with points in px of
     the run's grid, its colour), the selection areas, the masks shown, the
     mask applied, and the figure (source, image, tool, display).
   - Below: the mask applied (Processing > Maps and Quiver > Mask).
2. **Figure** -- **Select from**:
   - **Current Frame**: the frame chosen in Load data (|u| with the
     vectors; it follows the Load frame), calibrated when possible;
   - **Select Image**: opens a file selection dialog to choose an image
     on the disk (tif, png, jpg, bmp or a `.mat`), placed on the PIV grid
     as the background of Load data; **Change...** chooses another one
     (cancelling the first dialog goes back to Current Frame).
   The figure is *Mask* in the central panel.
3. **Select masking area**:
   - **Select tool**: *Polygon*, *Rectangle* or *Circle*.
     - Polygon: every left click is a new vertex; **Enter** or the
       **right button** closes the polygon (at least 3 vertices).
       Backspace removes the last vertex.
     - Rectangle: drag from one corner to the other (left button held).
       Circle: drag from the centre to the edge. Once drawn, the rectangle
       or the circle **stays editable**: drag inside it to move it, drag a
       corner (rectangle, square handles) or the edge (circle) to resize
       it. A **right click** or **Enter** validates it and adds it to the
       list.
     - Esc cancels. The zoom is kept (the toolbar zoom / pan tool is
       switched off), and the figure's right-click menu is off while
       drawing.
   - The **list of the areas** (name, tool, size), each with a **tick
     box**: the ticked areas make the selection (unticked ones are drawn
     dotted). Double click a name to rename it.
   - **Add**: draws a new area with the selected tool (named `Area1`,
     `Area2`, ...); **Remove**: removes the selected area.
   - **New mask from selection**: a new mask made of the ticked areas (a
     vector is masked when it is inside any of them), added to the list of
     masks (`Mask1`, `Mask2`, ...) with its own colour.
   - **Add selection to mask**: adds the ticked areas to **every mask
     ticked** in the list of masks (an area already in a mask is not added
     twice); a mask that is applied is applied again with its new areas.

The masks (their areas, with points in px of the run's grid so they do not
depend on the calibration, and their colour), the selection areas and the
mask applied are kept in memory with the run's parameters (`MASKS`,
`MASK_SELECTION`, `MASK_APPLIED`; nothing is written in a parameter file)
and carried over when a step loads the run again. They are recorded in
`run_information.json` (`mask`) and the run information table, but
nothing is restored from it: a new Load starts without masks unless a
mask file is ticked in Load data (another run can load the same mask
file).

## Processing > Maps and Quiver

The panel on the left is the **Display** box (it was in Load data); the
figure is *Maps and Quiver* in the central panel. It is drawn when the
step is opened, then follows every change (and every Load).

- **Velocity**: *calibrated* (m, m/s, s) or *uncalibrated* (px,
  px/frame, frame numbers). *calibrated* is listed only when the
  run has XSCALE and YSCALE (a calibration loaded or applied). The run is kept once in px
  and px/frame, and the calibration is applied when it is drawn, so
  switching costs no memory and reads nothing. The ROI list, the frame
  time, the Run information and the figure follow the choice.
- **Frame**: a single frame (slider and number), the time mean or
  the rms. Below the slider: the frame number and, when the run is
  calibrated, its time = frame × time between PIV fields, where the
  time between PIV fields is TSCALE = frames per PIV field / camera
  fps = 2 / camera fps (e.g. `frame 50 / 299   t = 10.000 s`).
- **ROI**: a drop-down menu: **Full**, then the ROIs ticked in Define ROI
  (`ROI01  (Validated_ROI01)`, ...), each with its colour square of Define
  ROI. Unticking in Define ROI the ROI shown goes back to *Full*. An ROI is
  not a mask: for
  each ROI (`PTS_ROI`) the velocity in its area is
  **extracted as a new field**, named like the full field with
  `_ROIxx`, where xx is the number of the ROI (`Validated_ROI01`,
  `Validated_ROI02`, ...; the ROI ROIxx). The field is
  the rows and columns of the grid inside the ROI -- a window of the
  stored data, so it costs no memory -- and is shown on its own grid
  (nothing masked), calibrated or not. On the full view every ROI is
  outlined with its name (ROIxx), and the lines (`LINES`) are
  drawn. The ROI fields are listed in the run information (and in
  `run_information.json`) and are what the later steps will use.
  `PTS_ROI` may be null (only *Full*), one rectangle
  `[[x1, z1], [x2, z2]]`, a list of rectangles or named rectangles
  `{"ROI01": [[..], [..]], ...}` (named by their place: other names are
  replaced); ROIs are in m, so there is no ROI
  field when the run cannot be calibrated.
  **Only the ROIs ticked in Pre-processing > Define ROI are listed and
  drawn** (an unticked ROI is neither in the list nor on the figure; if
  the ROI chosen is unticked, the display goes back to *Full*), and the
  same holds for the lines. Each ROI and each line is drawn **in its
  colour of Define ROI** (the colour square in front of the name in the
  list).
- **Mask**: *None*, or one of the masks built in Pre-processing >
  Mask (with its colour and number of vectors). The mask chosen is
  **applied**: the velocity in it is NaN for every step (this figure,
  the ROI fields, the later steps). The choice is saved in the run
  information and applied again at the next Load of the run.
- **Quiver** (tick box): draw the vectors of the velocity loaded
  (chosen in the Run box of Load data). Below it:
  - **Quiver**: the arrow colour, either |u| (colour map below) or a
    single colour (black, white, red, ...), then every n-th vector.
    The colour is set by default whenever the background changes:
    **black** without a background or on the velocity norm, **white**
    on an image. You can then choose any other colour; a session
    keeps the colour it was saved with.
  - **Vector scale** (x 1 by default): the length of the vectors on
    the figure. The 99th-percentile speed is drawn this many vector
    spacings long; a key at the top left of the figure gives that
    speed.
- **Background** (tick box), below it the choice:
  - *velocity norm*: |u| as a colour map.
  - *U* and *V*: the velocity components along x and z, in m/s
    (px/frame uncalibrated), diverging colour map (blue-white-red)
    centred on 0.
  - *divergence* du/dx + dv/dz and *vorticity* dv/dx - du/dz: diverging
    colour maps (blue-white-red), centred on 0, in 1/s (1/frame
    uncalibrated); derivatives by centred differences on the grid.
  - *kinetic energy* (u² + v²)/2 per unit mass, in m²/s² (px²/frame²).

  All the quantities are **calculated on the ROI selected** (the ROI field,
  on its own grid) **with the mask selected applied** (NaN in the mask; a
  derivative is NaN next to a NaN too), calibrated by default. For the
  time mean / rms, the quantity of every frame is averaged (rms: root mean
  square over time).
  - *image*: a camera image, in grey with its contrast stretched to
    the 1-99.5 percentiles. The **image** selector lists the image
    files (tif, png, jpg, bmp) and the `background*.mat` files of the
    run folder and of the root folder; **Browse...** picks any other
    file. The image is placed on the PIV grid the way
    `select_ROI_quiver.py` places it, from the pixel positions PIVlab
    writes in `x`/`y`, then rotated by `ROTATE`. A full camera frame
    therefore sits exactly under the vectors.
- **Smoothing**, **Shading** (each with a tick box) and **Colour map**:
  the map display, as in every step that draws a map -- see *Map
  display* below (for an image background the smoothing sigma is in image
  pixels).

The colour scale of |u| runs from 0 to the 99th percentile over the
view; the divergence / vorticity scale is symmetric (± the 99th
percentile of |value|). Save data writes the grid, the field shown (u, v,
|u|), the divergence, the vorticity and the kinetic energy, the vector
field, the ROI, the mask applied, the frame and its time, and the time
base.

### Make Movie

Below the Display box: a movie of the *Maps and Quiver* figure, **with the
settings of this step** (velocity calibrated or not, ROI field, mask,
quiver, vector scale, background, smoothing, colour map), for every frame
between a start and an end frame.

- **Start frame** and **End frame**: a slider and a number each (the start
  cannot be after the end); **FPS**: a slider and a number (1-60, 10 by
  default). Below: the number of frames, the length of the movie and the
  times of the record they cover.
- **Create movie**: draws the frames and makes the movie in the background
  (a progress bar; the button becomes **Cancel**). The colour scale (|u|,
  and the background map) and the arrow length are **fixed** over the
  movie: the 99th percentile over the frame range, so the frames compare.
  The movie is 1280 px wide, with the shape of the figure on screen.
  - With ffmpeg: an **MP4 (H.264, yuv420p, +faststart)**, which plays on
    macOS (QuickTime, Keynote) and on Windows (Films & TV, PowerPoint) as
    well as in browsers and VLC.
  - Without ffmpeg: an animated **GIF**.
- When the movie is finished, the figure **Movie** (Maps and Quiver, in the
  list of figures) opens and plays it: Play / Pause, back to the first
  frame, a frame slider, the frame shown (number in the movie, PIV frame
  and time), the playing speed (the movie's fps by default; the file keeps
  its own) and **Loop**. It plays the images of the movie kept in memory
  (about 150 kB per frame), so it needs no video codec. It can be opened in
  a separate window (right click). A new movie replaces it; Reset, Cancel,
  a failed movie or a new run removes it.
- **Save**: saves the movie created (until then it is a temporary file); the
  default folder is the run's `PostProcessing`, the default name
  `<run>_<background>_<ROI field>_f<start>-<end>_<fps>fps.mp4`.
- **Reset**: start, end and FPS back to the whole record at 10 fps; a movie
  not saved is dropped (as when a new run is loaded or the GUI is closed).

### Profiles (Processing > Profiles)

The layout of Maps and Quiver, with **Lines** in place of ROI, and no
quiver or background. The figure **Profiles** plots a quantity along the
lines of Pre-processing > Define ROI: x is the distance along the line from
its first end (m, or px uncalibrated), y the quantity.

- **Velocity** (calibrated / uncalibrated) and **Mask**: the run's
  choices, the same as in Maps and Quiver -- changing one here changes it
  there, and the other way round.
- **Frame**: single frame (slider + number, the frame and its time below),
  time mean or rms. This frame is the step's own (not the Maps one).
- **Lines**: *All lines* (every line on the same figure) or one line. Only
  the lines ticked in Define ROI are listed and drawn, each in its colour
  there (the colour square in front of the name). Lines are in m: none
  when the run cannot be calibrated.
- **ROI**: the same drop-down menu as in Maps and Quiver (Full + the ROIs
  ticked in Define ROI). With an ROI, the quantity is the one
  calculated on that ROI alone (Define ROI > Apply: `<quantity>_ROIxx`),
  extracted at the points of the selected line(s) **inside the ROI only**
  (x is still the distance from the line's first end). If a selected line
  has no point in the ROI, a message says *The selected line yy does not
  intersect the selected ROI xx* (once per choice of line and ROI; the
  line is then listed in the legend as "not in the ROI").
- **Extracted quantity**: velocity norm, U, V, divergence, vorticity,
  kinetic energy (the variables of the Maps background). **They are not
  calculated here**: the step uses the quantities along the lines
  calculated by Define ROI > Apply (`<quantity>_<LINE>_<Full or ROIxx>`,
  every frame; calculated once and stored if Apply was not pressed). On
  the full field, the quantity is computed with the mask applied, then
  interpolated (bilinear) along the line, about one point per vector
  spacing. A point next to a
  masked or rejected vector, or outside the grid, is NaN (a gap in the
  curve). The time mean / rms is taken over the frames of the stored
  values at each point of the line.
- The legend gives the line names; for U, V, divergence and vorticity a
  grey line marks 0. **Save data** (right click) writes, per line, its
  ends, the points (x, z), the distance along the line and the values.
- **Make Movie**: as in Maps and Quiver (start / end frame, FPS, Create
  movie / Cancel, Reset, Save; the figure **Profile movie** plays it). The
  y limits are fixed over the movie (0.5 - 99.5 percentiles of the values
  over the range, symmetric about 0 for the signed quantities). Default
  name: `<run>_profile_<quantity>_<line or all_lines>_f<start>-<end>_<fps>fps.mp4`.

### Time series (Processing > Time series)

For each active ROI, a time series built from its ROI quantities of
Define ROI > Apply (not calculated again): for every frame, a statistic
over the points of the ROI (NaN -- mask, rejected vectors -- left out).

- **ROI**: a replica of the ROI table of Define ROI (name, x, z, colour)
  with its **own tick boxes**: tick the ROIs to make them active, untick
  to make them inactive (ticked at first as in Define ROI; Define ROI is
  not changed). An ROI not applied yet is grey.
- **Quantity** (drop-down menu):

  | name | definition |
  |---|---|
  | `<U>` | mean over the ROI of `U_ROIxx` |
  | `U_std` | standard deviation over the ROI of `U_ROIxx` |
  | `<\|U\|>` | mean over the ROI of \|`U_ROIxx`\| |
  | `\|U\|_std` | standard deviation over the ROI of \|`U_ROIxx`\| |
  | `<V>`, `V_std`, `<\|V\|>`, `\|V\|_std` | the same with `V_ROIxx` |
  | `<Vorticity>`, `Vorticity_std`, `<\|Vorticity\|>`, `\|Vorticity\|_std` | the same with `vorticity_ROIxx` |
  | `<Kinetic energy>` | mean over the ROI of `kinetic_energy_ROIxx` |
  | `Kinetic Energy_std` | standard deviation over the ROI of `kinetic_energy_ROIxx` |

  (standard deviation: population, over the points of the ROI that are
  not NaN). The definition of the one chosen is shown below it.
- Figure **Time series**: x the time (s), y the quantity (m/s, 1/s,
  m²/s²), one curve per active ROI in its colour, named `ROIxx`;
  a grey line at 0 for `<U>`, `<V>`, `<Vorticity>`. **Save data** (right
  click) writes, per ROI, t, the values and the number of points used per
  frame.
- The mask applied is that of Maps and Quiver; sessions keep the quantity
  and the ticks.

## Map display (every step that draws a map)

Maps and Quiver (its Display box), Define ROI, Mask (a **Map display**
box) and FFT in time (**FFT map display**, for the FFT map) all have the
same three lines:

- **Smoothing** (tick box, unticked by default) and its **sigma** (in
  vector spacings, 1 by default): a Gaussian filter of the map values;
  NaN (mask, rejected vectors) are left out of the filter and stay NaN.
- **Shading** (tick box, unticked by default): unticked, one flat cell
  per vector; ticked, the shading chosen: **gouraud** (the colours
  interpolated between the vectors; it stops at the outer vectors) or an
  image interpolation **bilinear, bicubic, spline16, lanczos, hanning,
  gaussian** (on the regular PIV grid, the cells centred on the vectors).
  The shading only changes how the map is drawn; the smoothing changes
  the values. Both can be used together. A masked area stays empty.
- **Colour map**: **auto** (viridis; blue-white-red centred on 0 for the
  signed quantities U, V, divergence, vorticity), viridis, plasma,
  inferno, magma, cividis, turbo, jet, RdBu_r, coolwarm, seismic, PuOr_r,
  gray. In Maps and Quiver it also colours the arrows coloured by |u|.
  A signed quantity keeps its limits centred on 0 whatever the map.

Movies use the smoothing, shading and colour map of Maps and Quiver;
sessions keep them (tick boxes included).

## Spectral analysis > FFT in time

The FFT in time of a quantity of the Time series list, on an ROI or on the
full field.

- **ROI**: a drop-down menu: **Full**, then the ROIs ticked in Define ROI
  (ROI01, ...). An ROI uses its quantities of Define ROI > Apply (not
  calculated again); for *Full* the quantity is calculated once on the full
  field (kept until the next Load or mask change).
- **FFT quantity**: the same drop-down menu as Time series (`<U>`,
  `U_std`, `<|U|>`, `|U|_std`, the same for V, Vorticity, and
  `<Kinetic energy>`, `Kinetic Energy_std`); its definition is shown below.
- **Spectral function** (drop-down menu, below the FFT quantity):
  - **FFT**: the one-sided FFT, scaled as amplitudes; 4 plots: amplitude,
    phase, real part, imaginary part.
  - **Power spectrum**: the power spectral density by Welch's method: the
    record cut into segments (Hann window, 50 % overlap), the periodogram
    of each segment averaged -- smoother, coarser in frequency.
    **Segment length** (shown for this one): 0 = auto (a power of 2 near a
    quarter of the record: 64 frames for 300); longer segments give finer
    frequencies but more noise.
  - **Periodogram**: the power spectral density of the whole record in one
    piece (no window, no averaging): the finest frequencies (as the FFT),
    noisy.

  Two tick boxes act on each series before the spectral function:
  - **Remove mean** (ticked by default): its mean removed -- the value at
    f = 0 becomes 0.
  - **Detrend**: its linear trend (the best straight line in time)
    removed, and so its mean too: ticking it ticks and greys *Remove
    mean*. It takes away the leakage of a slow drift into the low
    frequencies.

  For the power spectrum (Welch) they act on each segment; for the FFT
  and the periodogram on the whole record. The figure title says which
  was used ("mean removed", "detrended"); Save data and sessions keep
  them.

  The two power spectra are in (unit)²/Hz (e.g. (m/s)²/Hz); their integral over f is the variance of the series (checked:
  1.00 and 1.005). They show one plot (the density; *log scale* is ticked
  when one is chosen).
- **Order of the calculation** (two tick boxes, independent: tick one or
  both; with both, each plot has one curve per order -- blue: average then
  spectral analysis, orange: spectral analysis then average -- with a
  legend below the plots):
  - *Average over ROI first then spectral analysis*: the quantity is
    first averaged over the ROI (the time series: mean, or standard
    deviation for the `_std` ones, of the quantity or of its absolute
    value), then its spectral function.
  - *Spectral analysis first then average over the ROI*: the spectral
    function of the quantity (or its absolute value) at each point of the
    ROI, then averaged over the ROI. FFT: real and imaginary parts = mean
    (standard deviation for `_std`) of those of the points; amplitude =
    mean (std) of the amplitudes of the points; phase = phase of the mean
    spectrum. Power spectrum, periodogram: mean (std) of the densities. For a mean quantity
    the real and imaginary parts are the same as with the other order
    (the FFT is linear); the amplitude differs: it is the mean of the
    amplitudes, so waves that cancel in the average still show.
- **Apply**: calculates and shows the figure **FFT in time**: for the FFT
  4 plots -- amplitude, phase (rad), real part, imaginary part --, for
  the power spectrum and the periodogram 1 plot, the density. One-sided spectrum,
  f = 0 to the Nyquist frequency, df = 1 / record length; the amplitude
  is scaled so that a sinusoid of amplitude A gives a peak of height A
  (the mean, f = 0, as it is). NaN in time (rejected vectors) are filled
  by linear interpolation; a point (or series) more than half NaN (e.g.
  masked) is left out. The line below the button gives the number of
  frequencies, points and the largest peak.
- **Display**: x axis **frequency** (Hz) or **period** (s, log scale; f = 0
  left out); below it **fmin** and **fmax** (Hz; 0 = auto): the frequency
  range shown on all the plots -- in period display the periods 1/fmax to
  1/fmin --, the y axes following the values in that range (the phase
  stays -pi .. pi). The line below gives the frequencies of the spectrum
  and the periods of the range (fmax <= fmin: fmax not used). They only
  change the display (not the calculation, the ranges or the peaks).
  **log scale** (amplitude / density).
- **Frequency / period ranges**: **Add range**, then drag on one of the
  plots (left button held; Esc cancels) -- in frequency or in period,
  as displayed; **Remove range** removes the one selected. Each range
  (W1, W2, ...; frequency and period) has a tick box; the boxes are
  exclusive: **one range ticked at a time**, the range used (FFT maps,
  Save data). A new range is ticked. None ticked: all frequencies. On the
  figure: green = the range ticked, grey = the others.
- **Plot FFT maps** (shown when *Spectral analysis first then average* is
  ticked): the figure **FFT map**, at each point of the ROI the FFT
  amplitude -- or the power spectral density -- integrated over the range
  ticked (sum x df; for a density: the variance in that range). **Only
  when a range is ticked**: with no range ticked (or no range) the button
  is greyed out, and unticking or removing the ticked range closes the
  map. Its shading and colour map: **FFT map display**. It
  follows the ranges and each new Apply.
- **Find peaks** (below Plot FFT maps): **Number of peaks n** (1 to 50,
  3 by default) and **Find peaks**: the n dominant peaks of the amplitude
  (FFT) or of the density (power spectrum, periodogram), for each order
  calculated -- the n highest local maxima (`scipy.signal.find_peaks`; a
  maximum at the end of the range counts), f = 0 left out, inside the
  frequency range ticked (all frequencies if none). They are listed in a
  table (#, order, frequency, period, amplitude / density), marked on the
  amplitude plot (triangles numbered by rank, in the colour of the order)
  and written in the log; a new Apply finds them again.
- **Save data** (right click): the spectrum (f, amplitude, phase, real,
  imaginary), the peaks found, the ranges and the frequencies used; for the map, the grid
  and the integrated amplitude. Sessions keep the ROI, quantity, order,
  display, spectral function, segment length, number of peaks and ranges.

## Spectral analysis > FFT in space

The spectrum of a quantity in space, at every frame. The panel on the
left, from top to bottom:

1. **Region and direction**
   - **Region**: *Full* or an active ROI (its stored quantities of Define
     ROI > Apply are used).
   - **Direction**: *along x* -- the spectrum of each row of the region,
     then averaged over the rows (wavenumber `k_x`); *along z* -- each
     column (`k_z`); *along a line* -- a line ticked in Define ROI, its
     values in the region stored by Define ROI > Apply (wavenumber `k`
     along the line, spacing about one vector spacing). **Line**: the
     line, for the last choice.
2. **FFT quantity**: U, V, velocity norm, divergence, vorticity, kinetic
   energy.
3. **Spectral function**: *FFT* (one-sided, scaled as amplitudes: a
   sinusoid of amplitude A gives a peak of height A; amplitude, phase,
   real and imaginary parts), *Power spectrum* (Welch: segments of
   **Segment length** points, Hann window, 50 % overlap) or *Periodogram*
   -- power spectral density per cycle per metre, (unit)²·m, whose
   integral over 1/λ is the variance. **Window** (FFT, periodogram): none
   or Hann (FFT amplitudes corrected by the window's mean). **Remove
   mean**, **Detrend** (the linear trend along the series -- the mean
   too). NaN along a series are filled by linear interpolation; a series
   more than half NaN (e.g. masked) is left out.
4. **Time**: *mean over all frames* (the spectra of every frame averaged)
   or *single frame* (the frame number; its time below).
5. **Apply** -> figure **FFT in space**: amplitude, phase, real part,
   imaginary part -- or the density alone for the power spectra -- the
   spectra of the series averaged (amplitude: mean of |F|; real and
   imaginary parts: mean; phase: of the mean complex spectrum). The line
   below gives the wavenumbers (dk, kmax), the number of series and the
   largest peak.
6. **Display**: x axis **wavenumber k** = 2π/λ (rad/m), **1/λ** (1/m) or
   **wavelength λ** (m, log scale); **kmin**, **kmax** in rad/m (0 = auto;
   they also limit the peaks and the k–t diagram); log scale (on for the
   power spectra).
7. **Find peaks**: the n dominant peaks (local maxima, k = 0 left out,
   within kmin .. kmax): k, wavelength, value; marked on the figure.
8. **k–t diagram** (its own section): **Wavenumber** -- the ones
   available for the region: `k_x` and `k_z` (at least 4 points along x /
   z) and `k along LINEx` for each ticked line crossing the region;
   colour map; log colour scale. **Plot k–t diagram** -> figure **k–t
   diagram**: the spatial spectrum (the quantity and spectral options
   above) at every frame -- the wavenumber in x (the Display x axis, kmin
   / kmax), the time in y, the amplitude or density in colour.

Save data (right click) writes f, k, the wavelength, the spectrum (and the
spectrum at every frame, `kt_...`, with the times) and the peaks. A
session keeps every setting and performs Apply, Find peaks and Plot k–t
diagram again; Save variables lists `fft_space_options`,
`fft_space_spectrum`, `fft_space_peaks`, `fft_space_kt_diagram`.

## Advanced > Python script

Your own Python scripts, run on the loaded run inside the GUI: to compute
what the GUI does not offer, make your own figures, or even add your own
steps to the GUI.

### Using it

1. **Script**: the drop-down lists the `.py` files of the root folder, of
   its `scripts` folder and of the GUI's **`Example_Scripts`** folder, and
   those opened. **Open...** chooses any `.py` file; **New...** writes a
   template script (the names available and an example) and selects it.
   The script is shown below (read only: edit it in your editor; it is
   read again at each Execute).
2. **Execute** runs the script on the loaded run.
3. **Output**: the printed text, the error with its traceback (the GUI goes
   on), the time taken; the list of the results by script; **Remove the
   results of this script**.

### What a script can use

| name | what |
|---|---|
| `d` | the loaded run (`PIVData`: `d.run`, `d.run_dir`, `d.nt`, `d.t`, `d.X_px`, `d.U_px`, ..., `d.params`) |
| `np`, `lib` | numpy, `piv_gui_lib` (all the calculations and figures of the GUI) |
| `view(roi=None, calibrated=True)` | the run as shown (`.x`, `.z` in m, `.t` in s, `.uv()` all frames in m/s, `.uv(i)` one frame); `roi`: `"ROI01"` (or its field name `"Validated_ROI01"`) |
| `quantity(name, roi=None)` | a quantity for every frame (ny × nx × nt): `"U"`, `"V"`, `"velocity norm"`, `"divergence"`, `"vorticity"`, `"kinetic energy"` (an ROI: its stored quantity of Define ROI > Apply) |
| `line(name, quantity, roi=None)` | the values along a line of Define ROI: `values` (points × frames), `s`, `x`, `z`, `unit` |
| `roi_field(name)` | `"ROI01"` -> `"Validated_ROI01"` |
| `variables()` | every variable of Show all variables, as a dict |
| `results` | a dict: **put here what the script produces** |
| `figure(name)` | a new matplotlib figure, shown in the GUI under *Python script* |
| `log(text)` | a line in the log (and the output); `print()` goes to the output |
| `run_dir`, `post_dir` | the run folder, its PostProcessing folder |
| `gui`, `G`, `QtWidgets`, `QtCore`, `QtGui` | the main window, the GUI module (its classes), Qt -- to build widgets |
| `add_advanced_item(id, label, panel)` | **a new item of the Advanced menu** with its own panel on the left (and its figures under `label`) |

### What happens to the results

The results are kept in memory (until another run is loaded), listed in
Show / Save variables (*Advanced > Python script*: `script_results`, one
entry per script named by its file, and `script_runs`) and **saved with the
session**: Save session writes them in `<session>.scripts.npz` next to the
session file. Load session, Apply session and Batch processing execute the
scripts again (a successful Execute is a step of the session); Load
session takes the saved results of a script whose file is missing (same
run only). An Advanced item made by a script is made again, with its
settings, when the session is loaded. The **run information** has a
section **Advanced: Python scripts** listing each script executed --
executed or FAILED, the time, the duration, the file, the results, the
error -- updated and saved after each Execute.

### The examples (`Example_Scripts/`, next to the GUI)

| script | what it does |
|---|---|
| `map_and_quiver_figure.py` | the map of a quantity with the velocity vectors on top, drawn **with plain matplotlib from the calibrated data**: every step can be read and changed. Parameters at the top: `ROI` (`"Full"`, `"ROI01"`, ...), `FRAME` (index, `"mean"`, `"rms"`), `BACKGROUND`, `QUIVER`, `SKIP`, `VECTOR_SCALE`, `QUIVER_COLOUR`, `CMAP`, `FIGURE_NAME`. The figure appears under *Python script*; the arrays (x, z, u, v, speed, background) in `results` |
| `map_and_quiver_lib.py` | the same figure made **by the GUI's own function** (`lib.display_figure`, identical to Processing > Maps and Quiver), with all its options as parameters at the top (`ROI`, `FRAME`, `BACKGROUND`, `IMAGE`, `QUIVER`, `SKIP`, `VECTOR_SCALE`, `SMOOTHING`, `SHADING`, `CMAP`); its data in `results` |
| `reproduce_maps_and_quiver.py` | **adds an item to the Advanced menu, *Reproduce maps and quiver***, that reproduces Processing > Maps and Quiver exactly -- the same Display options (velocity, frame, ROI, mask, quiver, background, smoothing, shading, colour map), the same figure and the same Make Movie section -- built with the GUI's own classes. It shows how a script can add a whole step to the GUI |

### Developing scripts with Claude (or any other AI)

The file **`Claude_for_Developer.md`**, next to the GUI, holds everything
an AI assistant needs: how the GUI is organised, its data model and rules,
the names given to a script, the library functions, the examples and
recipes. Give it to Claude (or any other AI assistant) with your request --
e.g. *"Using Claude_for_Developer.md, write a script for the PIV GUI that
plots the vorticity averaged over ROI01 against time and saves it"* --
and run the script it writes with Advanced > Python script. The same file
lets an AI assistant modify the GUI itself (adding a step, a variable, a
session action) following its rules.

**Note**: a script runs inside the GUI with the rights of your Python: it
can read, write and delete files. Run only scripts you trust. The GUI
waits while a script runs.

## Apply session steps to another dataset (File menu)

A window with a **Session file** (Browse...) and a **Data set** (the run
folder, Browse...). **Run session** loads the session file, changes its
paths to the new data set (its root folder = the parent folder), loads the
run with **the parameters of the session** (its calibration, ROIs, lines
and masks, saved in the session), the same velocity,
then sets every widget back and performs again every step of the session
(Calibration > Apply, Define ROI > Apply, FFT, ...), as Load session does.
The **run information** (table and `run_information.json`) is that of the
new data set; it is updated and saved again at the end, with the
calibration / ROI / mask files of the session (the same at the end of
each run of a batch). The line at the bottom says when it is done (or why it
failed).

## Batch processing (File menu)

1. **Root folder** (Browse...): its run folders are listed.
2. **Session processing to apply**: the session file, then **Load and run
   session**: the session is loaded and run on its own run (the one in
   the session file), so that every variable it makes is in memory.
3. **Runs**: the run folders of the root, each with a tick box. **Search**:
   key words (space separated, all of them) that must be in the folder
   name; **All** / **None** tick or untick the runs shown. The count of
   runs, shown and ticked is given below.
4. **Variables to save**: **Select variables...** opens the window of
   File > Save variables on the variables available after the processing;
   tick them and press **Use this selection**. **Format**: npz, NetCDF,
   HDF5, txt, MATLAB (no CSV).
5. **Batch process**: for each run ticked, in turn, the session steps are
   applied exactly as *Apply session steps to another dataset* does; when
   a run is done, the variables selected are saved in
   `<run>/PostProcessing/Variables_<session name>.<ext>` and the session
   is saved automatically **in the run folder** as `<run>/<run>.session.json`
   (it can be loaded with Load session to look at that run). **Stop** stops after the run being processed.

   **The batch runs in the background**: the runs are loaded and
   processed by a hidden worker window, so the main window -- its run,
   panels, figures and log -- is not changed (only a line at the start and
   at the end of its log). The batch window stays open and shows a
   **progress bar** (the runs done and the steps of the run being
   processed), a line *Run k / n: <run> -- loading the run / step i / n /
   saving the variables*, the **batch log** and a table with each run's
   result and files. Closing the window does not stop the batch.

In a batch no question is asked: a run without its PIVlab .mat file is
skipped (reported as failed), a run without an acquisition log uses the
time base of the session; an entry of the selection that a run does not
have (e.g. an ROI it lacks) is left out and reported. Everything is also
written in the batch log.

## Save variables (File > Save variables)

As Advanced > Save variables of the DTS GUI: a window listing all the
variables, grouped by step in the menu order, with their description,
format, size and memory (the steps start collapsed; a step not run yet
shows "nothing yet"). The names say what the variable holds (and its
unit when it matters):

| Step | Variables |
|---|---|
| File > Load data | run_name, run_folder, pivlab_result_file, acquisition_log_file, setup_files_loaded, parameters, velocity_type_loaded, velocity_types_in_file, x_grid_px, z_grid_px, u_velocity_px_per_frame, v_velocity_px_per_frame, rejected_vectors, number_of_fields, field_sources, valid_vector_fraction, mat_file_variables, acquisition_log_last_row |
| Pre-processing > Calibration | calibration_scales, calibration_segment, time_base, velocity_calibrated_m_per_s (computed when saved) |
| Pre-processing > Define ROI | roi_and_line_coordinates, roi_velocity_windows, roi_derived_quantities, line_derived_quantities, velocity_calibrated_m_per_s_ROIxx (computed when saved) |
| Pre-processing > Mask | mask_applied, mask_applied_name, mask_applied_areas, masks_defined |
| Processing > Maps and Quiver | maps_display_options, maps_figure_data, maps_movie_info |
| Processing > Time series | time_series_options, time_series_data (once the figure is made) |
| Processing > Profiles | profiles_options, profiles_data (once the figure is made), profiles_movie_info |
| Spectral analysis > FFT in time | fft_time_options, fft_time_frequency_ranges, fft_time_spectra (per order: f, amplitude or psd, phase, real, imaginary, peaks), fft_time_peaks, fft_time_spectrum_per_point (spectrum at each point, spectral analysis first), fft_time_integrated_map (amplitude integrated over the ticked range) |
| Spectral analysis > FFT in space | fft_space_options, fft_space_spectrum (f, k, wavelength, amplitude or psd, phase, real, imaginary; kt_ spectrum at every frame), fft_space_peaks, fft_space_kt_diagram |
| Advanced > Python script | script_results (per script: its `results`), script_runs (the scripts executed) |

- **Descriptions**: each variable has a detailed description (what it
  holds, shape, units, how it was obtained, the conversion to physical
  units); the column shows it without the step, the box below the list
  shows the full text of the row selected (and the tool tip). The entries
  inside a variable (f, amp, phase, t_s, x_m, `<quantity>_ROIxx`, W1, ...)
  have their own short description (ENTRY_HELP of piv_gui_lib.py).
- **Search** (top): keywords, case ignored; an entry is shown when its
  name, its parents' names, its step or its description hold every
  keyword (`fft amplitude`, `ROI01`, `vorticity`); the parents of a match
  are shown and expanded, and a match shows all its entries. **All** /
  **None** tick / untick only the entries shown.
- Each step has a tick box that ticks all its variables; each
  variable has its own tick box, and a dictionary variable (parameters,
  roi_derived_quantities, masks_defined, ...) opens to its entries, each
  with a tick box (only the entries ticked are saved).
- **All**, **None**, **Expand all**, **Collapse**, **Refresh**; below the
  list: the number of variables and entries ticked.
- **Save...** asks the file name, the folder and the format: **npz,
  NetCDF, HDF5, txt, MATLAB** (gui_export, the same writer as the
  DTS GUI and as Save data of the figures; **CSV is not offered** for the
  variables -- a `.csv` name is saved as npz). Text writes one file per
  table plus an index file. Proposed: `<run>_variables` in the folder
  of the last save, or `<run>/PostProcessing`.
- The data in memory are uncalibrated (px, px/frame).
  `velocity_calibrated_m_per_s` (x_m, z_m, u_m_per_s, v_m_per_s for every
  frame, t_s; NaN in the mask applied) and the same for each ROI
  (`velocity_calibrated_m_per_s_ROIxx`) are computed only when saved, so
  they use no memory before.
- `roi_derived_quantities` holds the quantities of Define ROI > Apply
  (`<quantity>_ROIxx`).
- The window stays open (not modal) and its list follows the run (Load,
  Apply, mask, ...). The files written are listed in the log.
- The names changed on 2026-11-20 (they were RUN, U_PX, FFT_IN_TIME, ...):
  a batch variable selection made before must be made again.

## Sessions

**File > New session** starts a fresh session: all the variables in
memory (the run, the ROI quantities, the FFT results, ...), the figures
(and their separate windows), the steps performed, the settings of every
step and the log are removed, and a new window opens in their place (same
size and place, the root folder kept). Movies not saved are deleted.
Files on disk are not changed. It asks first: **Save session first...**
(then the new session; cancelling the save cancels everything), **New
session**, or **Cancel**. Not while a run is being loaded.

**Quitting** (File > Quit or closing the window) asks first, as New
session does: **Save session first...** / **Quit without saving** /
**Cancel** (only when a run or figures are in memory; a running batch
stops).

**File > Save session** writes a `*.pivsession.json` file with everything
needed to come back to the same point:

- the files: root, run, the calibration / ROI / mask files and their
  tick boxes, the calibration / ROI / mask files the steps were run from
  or saved to (`setup_files`), velocity loaded, the time base in memory;
- the parameters of the run in memory (`params`: calibration, ROIs,
  lines, masks, mask applied) -- the run is loaded again with them;
- the state of **every panel and every widget**: all tick boxes, drop-down
  menus, numbers, sliders and text fields of Load data's display (Maps and
  Quiver), Define ROI, Mask, Profiles, Time series, FFT in time, the two
  Make Movie boxes and the map displays; the calibration image, segment
  and length; the ROI and line lists with their ticks (also edits not
  applied yet); the masks shown; the FFT frequency / period ranges and
  their ticks; the step shown;
- **the steps performed**, in the order they were last performed:
  Calibration > Apply (the scale in memory, not saved), Define ROI > Apply,
  FFT in time > Apply, Find peaks, Plot FFT maps (shown in the log when
  saved);
- the figures shown.

**File > Load session** reads the file, loads the run, sets every widget
back, then **performs again every step marked as performed**, in the same
order -- a step that loads the run again (Calibration > Apply) waits for
the run before the next one --, shows the figures of each step again and
ends on the step shown when it was saved. The log lists each step
performed again; the status bar sums it up. Edits of the ROI lists not
applied when the session was saved come back after the steps, still not
applied.

Not performed again: a movie (Create movie: long; the movies saved are
files), and the masks (they come back with the session's parameters). Pressing
**Load** in Load data starts the list of steps performed again. A
session saved before this version still loads (its settings, without
the steps).

## Adding a step (V2+)

1. Give the step a panel class (like `LoadPanel`) and register it in
   `MainWindow.__init__` in place of its `PlaceholderPanel`.
2. Put the computations and figure builders in `piv_gui_lib.py`.
   **Data and calibration (rule for every step):**
   - Only the **uncalibrated** data are kept in memory, once: the loaded
     run (`PIVData`) holds `X_px`, `Z_px` (px) and `U_px`, `V_px`
     (px/frame) of the velocity loaded. No step stores calibrated copies.
   - By default a step **applies the calibration factors before any
     calculation**: it works on `view = d.view()` (a `RunView`,
     calibrated by default): `view.x`, `view.z` [m], `view.u`, `view.v`,
     `view.uv(frame)` [m/s], `view.t` [s], `view.lunit`, `view.vunit`,
     `view.tunit`. The arrays are computed from the px data when they are
     asked for (x XSCALE / YSCALE, velocities / the pulse separation) and
     dropped after use, so a view stores nothing.
   - A step uses the uncalibrated data only when it has an option for it
     (as **Velocity** in Maps and Quiver): `d.view(calibrated=False)`
     (px, px/frame, frame numbers). When the run cannot be calibrated
     (XSCALE / YSCALE null), `d.view()` is uncalibrated; check
     `d.can_calibrate` / `view.calibrated` to say so.
   - A step does not depend on the Maps and Quiver choice (`d.display`,
     `d.set_calibrated`), which only sets what Load data shows. The
     uncalibrated arrays a view returns are read-only, so a calculation
     cannot change the stored data by mistake.
3. Add its figures with `self.figures.add("<step id>", "<name>", fig,
   data_fn=..., update_fn=...)` (tables: `self.figures.add_table(...)`).
   `data_fn` returns what Save data writes and `update_fn` redraws the
   figure. The figures then appear in the list under the step, with tick
   boxes, the right-click menu and separate windows included.
