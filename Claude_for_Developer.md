# Claude_for_Developer.md

Everything an AI assistant (Claude or any other) -- or a person -- needs to
**modify this GUI** or to **help a user write Python scripts** for its
*Advanced > Python script* item. Read it whole before changing anything.

- Part A: what the GUI is, how it is organised, its rules.
- Part B: how to change the GUI (add a step, a variable, a session action).
- Part C: how to write scripts for *Advanced > Python script* (the API, the
  examples, recipes).
- Part D: how to test, and the checklist before handing over.

---------------------------------------------------------------------------

## A. The GUI

### A.1 What it is

A desktop GUI (Qt through matplotlib's `qt_compat`: **PySide6 or PyQt5**,
whichever is installed) to post-process **PIVlab** results (`.mat` files of
velocity fields) of laboratory experiments: calibration, regions of
interest (ROIs) and lines, masks, maps and quiver plots, profiles, time
series, spectral analysis in time and space, sessions, batch processing,
user scripts.

Start it with `python PIV_GUI.py` (optionally `python PIV_GUI.py
session.pivsession.json`).

### A.2 The files

| file | what |
|---|---|
| `PIV_GUI.py` | the Qt application: the main window, one panel class per step, the figure manager, sessions, batch, scripts |
| `piv_gui_lib.py` | the library, **no Qt**: reading the data, the run (`PIVData`, `RunView`), all calculations, all figures (matplotlib `Figure`s) and their data for saving |
| `gui_variables.py` | the windows View > Show all variables and File > Save variables (grouped by step, search, details) |
| `gui_export.py` | the writer of npz / NetCDF / HDF5 / txt / csv / MATLAB -- **shared with another GUI (DTS): do not modify it** |
| `README.md` | the user documentation -- keep it up to date with every change |
| `TODO.md` | the user's to-do list (reminders for the next work sessions): read it at the start of a session; **not published** on GitHub |
| `update_github.command` | publishes the folder on GitHub (commit + push), see `README.md`; **not published** itself |
| `Claude_for_Developer.md` | this file |
| `Example_Scripts/` | example user scripts (Part C) |

Separation: **calculations and figures go in `piv_gui_lib.py`** (testable
without Qt), **widgets and wiring in `PIV_GUI.py`**.

### A.3 The window

- Left: the panel of the step chosen in the menus (a `QStackedWidget`,
  `main.panels`; `main.panel_of[step_id]` is the panel).
- Centre: the figures as tabs; right: the list of figures by step, each
  with a tick box (shown or not). `main.figures` is the `FigureManager`:
  `add(step, name, fig, data_fn=None, update_fn=None)`, `find(step,
  name)`, `focus(page)`, `remove(page)`, `add_table(...)`,
  `add_movie(...)`. `data_fn` returns the data written by *Save data*
  (right click); `update_fn` redraws.
- Bottom: the log (`main.write_log(text, error=False)`), and the status
  bar (`main.status(text, ms=8000, log=False)`).
- Menus (`MENUS` in `PIV_GUI.py`): File (Load data, sessions, batch, Save
  variables, Quit), Pre-processing (Calibration, Define ROI, Mask),
  Processing (Maps and Quiver, Time series, Profiles), Spectral analysis
  (FFT in time, FFT in space, ...), Advanced (Python script, + items added
  by scripts), View.
- `main.set_step(step_id)` opens a step: it calls the panel's `open()`.

### A.4 The data model (important rules)

1. **One run in memory**: `main.data` is a `lib.PIVData`, or None.
   - Grid `d.X_px`, `d.Z_px` (ny, nx) in **px**; velocity `d.U_px`,
     `d.V_px` (ny, nx, nt) float32 in **px/frame** -- **only the
     uncalibrated data are kept in memory**; the calibration (m, m/s, s) is
     applied when the data are read through a view.
   - `d.params`: the parameters **in memory** (there is **no parameter
     file** any more): `XSCALE`, `YSCALE` (m/px), `ROTATE` (0, 90, 180,
     -90), `PTS_ROI`, `LINES` (`{name: [[x1, z1], [x2, z2]]}` in m),
     `MASKS`, `MASK_APPLIED`, `MASK_SELECTION`, `CALIB_INFO`, ...
     They come from the calibration / ROI / mask files of Load data and
     from the steps, and are carried over when a step reloads the run.
   - `d.can_calibrate` (XSCALE set), `d.calibrated` (shown calibrated),
     `d.nt`, `d.t` (s), `d.frame_text(i)`, `d.run`, `d.run_dir`,
     `d.post_dir` (`<run>/PostProcessing`), `d.masks`, `d.mask_name`,
     `d.mask` (bool ny x nx, True = masked; NaN velocity there).
2. **Views**: `lib.RunView(d, calibrated, window=None)` -- `.x`, `.z`
   (m or px), `.t`, `.uv()` all frames / `.uv(i)` one frame (m/s or
   px/frame, NaN in the mask), `.lunit`, `.vunit`, `.tunit`,
   `.calibrated`. `lib._roi_view(d, calibrated, roi)` gives the view of an
   ROI field.
3. **ROIs**: named by their place in the list, `ROI01`, `ROI02`, ...
   Each ROI makes a field `<velocity>_ROIxx` (e.g. `Validated_ROI01`), a
   window of the grid (no copy): `d.roi_fields()` = {field name: dict(roi,
   number, window, rect, shape)}. **Active ROIs** = ticked in Define ROI
   (`main.roi_panel.shown("rect")`): only they are offered in the
   drop-downs (`RoiCombo`) and calculated.
4. **Stored quantities -- never recalculate them in a later step**:
   - Define ROI > Apply computes, for each active ROI, `U`, `V`,
     `velocity norm`, `divergence`, `vorticity`, `kinetic energy` for every
     frame: `d.roi_quantities['<quantity>_ROIxx']` (dict with `data`
     (ny, nx, nt) float32, calibrated units). Use
     `lib.roi_quantity(d, field, name)` / `lib._quantity_cube(d, roi, q)`.
   - and the same quantities along each ticked line, in the full field and
     in each active ROI it crosses: `d.line_quantities['<q>_<LINE>_<Full or
     ROIxx>']` (data (npts, nt), s, x, z). Use `lib._stored_line(...)` /
     `lib.line_profile(..., line=name)`.
5. **Rotation**: applied when the run is read (`lib.rotate_fields`): the
   grid **and the velocity components** are rotated; U is along the
   horizontal axis of the rotated figure.
6. **Setup files** (the steps' own files, JSON, double extensions):
   `.pivcalib.json` (Calibration > Save), `.pivroi.json` (Define ROI >
   Save), `.pivmask.json` (Mask > Save); Load data loads each when its box
   is ticked and runs the step from it. File dialogs must filter on
   `*.json` (`json_filter(label, ext)`): the macOS native dialog greys out
   files of a double-extension pattern; `with_double_ext(path, ext)` adds
   the extension on save.
7. **Run information**: `main.show_info(d)` rebuilds the table and
   **saves `<run>/run_information.json`** -- call it whenever something it
   shows changes. It is a record only: nothing is restored from it.

### A.5 Sessions, the journal, batch

- `main.session_state()` collects every panel's `get_state()`, the
  generic widget states (`widget_states(obj, exclude)` -- every widget
  attribute by name), `params` (the run's parameters), the setup files,
  the scripts, and **`executed`: the journal** of the steps performed
  (`main.mark_done(action)`).
- Load session loads the run with the session's `params`, restores the
  widgets, then **performs the journal again** (`MainWindow._perform`):
  `calibration`, `roi_apply`, `fft_apply`, `fft_peaks`, `fft_map`,
  `fftspace_apply`, `fftspace_peaks`, `fftspace_kt`,
  `script:<path>` (a Python script executed again). A step that reloads
  the run returns True and the replay resumes in `_loaded`.
- Apply session to another dataset / Batch processing: the same on
  another run. The batch runs in a hidden worker `MainWindow(worker=True)`
  (no questions, the user's window untouched).
- Every new action that produces something must be: marked with
  `mark_done`, performed in `_perform`, described in `STEP_ACTIONS`.

### A.6 Variables (Show / Save variables)

`MainWindow._variables()` returns `(values, descriptions, note,
VARIABLE_STEPS)`. Names are explicit snake_case (`u_velocity_px_per_frame`,
`roi_derived_quantities`, `fft_time_spectra`, ...), grouped by step in
`VARIABLE_STEPS`, each with a detailed description (`lib.VARIABLE_HELP`,
or the text given to `add(...)` in `_step_variables`). Entries inside a
variable get a short description from `lib.ENTRY_HELP`. Values computed
only when saved: `_variables_to_compute` (`gui_variables.Lazy`).

### A.7 Coding rules

- **Both bindings**: use `E("Qt.Orientation.Horizontal")` for every Qt
  enum (scoped in Qt6, not in Qt5), `_qexec(dialog)()` instead of
  `exec_`/`exec`, `Signal` from `qt_compat`.
- Never block the GUI with a modal dialog in batch mode (`main._batch`).
- Simple, explicit English in the GUI texts, tool tips and docs; units
  everywhere (m, m/s, s, Hz, rad/m).
- A step's calculations: in `piv_gui_lib.py`, returning dicts of numpy
  arrays + a `*_figure(d, ..., fig=None)` + a `*_data(...)` for Save data.

---------------------------------------------------------------------------

## B. Changing the GUI

### B.1 Adding a step (a menu item with its panel)

1. `MENUS`: add `(step_id, "Label")` to its menu.
2. A panel class `XxxPanel(QtWidgets.QWidget)` with `STEP = step_id`,
   `open()`, `data_loaded(d)`, `sync()` (when ROIs / masks change),
   `get_state()` / `set_state(s)` (+ `apply_pending()` for widgets filled
   after a Load).
3. `MainWindow.__init__`: create it, map it in the `panel_of` loop;
   `set_step`: call its `open()`; `_loaded`: `data_loaded(d)`; every place
   that calls `fft_time_panel.sync()` calls it too.
4. Sessions: `session_state` / `_start_session` (`set_state`),
   `_widget_panels` (generic widget states), journal actions (A.5).
5. Variables: `VARIABLE_STEPS` + `_step_variables` (A.6).
6. The calculation and the figure in `piv_gui_lib.py`.
7. Docs: `README.md` (a section) and this file if the architecture
   changes; mark the item done in `TODO.md` (`[done dd.mm.yyyy]`).

### B.2 Without changing the GUI: an Advanced item from a script

A script can add a whole step at run time with
`add_advanced_item(id, label, panel)` (Part C,
`Example_Scripts/reproduce_maps_and_quiver.py`): good for prototypes; the
item is made again when a session that executed the script is loaded, and
its widgets are saved with the session.

---------------------------------------------------------------------------

## C. Writing scripts for Advanced > Python script

### C.1 How a script is run

The user chooses a `.py` file (root folder, its `scripts` folder, the
GUI's `Example_Scripts` folder, or Open...) and presses **Execute**. The
file is read and executed with `exec` in a namespace holding the names
below, **in the GUI's process** (so it can use the GUI itself). `print`
goes to the panel's output; an exception is shown with its traceback (the
GUI goes on). A successful Execute is recorded in the session journal
(`script:<path>`), so the script is executed again by Load session, Apply
session and Batch processing.

### C.2 The names given to a script

| name | what |
|---|---|
| `d` | the loaded run (`lib.PIVData`, A.4) |
| `np`, `lib` | numpy, `piv_gui_lib` |
| `view(roi=None, calibrated=True)` | a `RunView` of the full field (`None` / `"Full"`) or of an ROI (`"ROI01"` or its field name `"Validated_ROI01"`): `.x`, `.z` (m), `.t` (s), `.uv()` (u, v (ny, nx, nt) m/s, NaN in the mask), `.uv(i)`, `.lunit`, `.vunit`, `.tunit` |
| `roi_field(name)` | `"ROI01"` -> its field name `"Validated_ROI01"` (`"Full"` stays) |
| `quantity(name, roi=None)` | (roi: `"ROI01"`, a field name or None / `"Full"`) (ny, nx, nt) of `"U"`, `"V"`, `"velocity norm"`, `"divergence"`, `"vorticity"`, `"kinetic energy"` -- an ROI: its stored quantity (Define ROI > Apply), the full field: computed once and cached |
| `line(name, quantity, roi=None)` | the stored values along a line (in the full field, or the part inside an ROI): dict `values` (npts, nt), `s`, `x`, `z`, `unit` |
| `variables()` | every variable of Show all variables (dict) |
| `results` | **a dict to fill with what the script produces**: kept (one entry per script, named by its file), listed in Show / Save variables (`script_results`), saved with the session (`<session>.scripts.npz`) |
| `figure(name)` | a new matplotlib `Figure`, shown in the GUI under *Python script* once the script ends (a figure of the same name is replaced) |
| `log(text)` | a line in the log |
| `run_dir`, `post_dir` | the run folder, `<run>/PostProcessing` |
| `gui` | the main window (`MainWindow`): `gui.figures`, `gui.roi_panel`, `gui.load_panel`, `gui.set_step(id)`, `gui.write_log`, ... |
| `G` | the GUI module (`PIV_GUI`): `G.RoiCombo`, `G.MovieBox`, `G.MapStyleBox`, `G.E`, `G.BACKGROUNDS`, `G._smooth_spin`, ... |
| `QtWidgets`, `QtCore`, `QtGui` | Qt (the binding in use) |
| `add_advanced_item(id, label, panel)` | adds an item to the Advanced menu with `panel` (a `QWidget`) on the left; its figures are listed under `label` (use `step = id` in `gui.figures.add`). The panel may define `open()` (the item is opened) and `data_loaded(d)` (a run is loaded). Executing again replaces it |

### C.3 The useful library functions

| function | what |
|---|---|
| `lib.display_figure(d, fig=fig, frame=, roi=, quiver=, quiver_colour=, skip=, vec_scale=, background=, image=, smoothing=, shading=, cmap=, rois_shown=, lines_shown=)` | the figure of Maps and Quiver |
| `lib.display_data(d, frame, None, roi)` | its data |
| `lib.frame_quantity(d, vw, frame, name)` | (map, label) of a quantity at a frame / `"mean"` / `"rms"` |
| `lib.quantity(vw, u, v, name)`, `lib.quantity_unit(vw, name)` | a quantity from (u, v) on a view's grid, its unit |
| `lib.draw_map(ax, X, Z, V, shading=, cmap=...)`, `lib.map_cmap(cmap, signed)` | a map as the GUI draws it |
| `lib.roi_time_series(d, field, name)`, `lib.TIME_SERIES` | the time series of Processing > Time series |
| `lib.line_profile(d, rect, quantity, frame, calibrated, None, roi, line=name)` | a profile along a line |
| `lib.fft_in_time(d, roi, name, order, function, ...)` | FFT in time (dict f, amp, phase, ...) |
| `lib.fft_in_space(d, roi, q, direction, line, function, ...)` | FFT in space (dict f (cycles/m), amp, amp_t (nt, nk), ...) |
| `lib.find_spectral_peaks(res, n)` | the n dominant peaks of a spectrum |
| `lib.parse_shapes(d.params.get("LINES"), "LINE")` | the lines `[(name, ((x1, z1), (x2, z2)))]` (same for `PTS_ROI`, `"ROI"`) |
| `lib.new_figure(figsize)` | a Figure for the GUI |

### C.4 The examples (`Example_Scripts/`)

| script | shows |
|---|---|
| `map_and_quiver_figure.py` | a map of a quantity with the vectors, **with plain matplotlib** from the calibrated data -- parameters at the top: `ROI`, `FRAME`, `BACKGROUND`, `QUIVER`, `SKIP`, `VECTOR_SCALE`, `CMAP`, ...; the arrays in `results` |
| `map_and_quiver_lib.py` | the same figure **with the GUI's own function** `lib.display_figure` (identical to Processing > Maps and Quiver), parameters at the top |
| `reproduce_maps_and_quiver.py` | **adds an item to the Advanced menu**, *Reproduce maps and quiver*, that reproduces the step Processing > Maps and Quiver exactly: its Display options, its figure and its Make Movie section, built with the GUI's classes (`G.RoiCombo`, `G.MovieBox`) |

### C.5 Recipes

```python
# parameters at the top, in capitals: the user changes them and Executes
ROI = "ROI01"; FRAME = "mean"

field = roi_field(ROI)                     # "Validated_ROI01"
vw = view(field)                           # m, m/s, s
w = quantity("vorticity", field)           # (ny, nx, nt), stored, 1/s

# a time series and a figure
series = np.nanmean(w, axis=(0, 1))
fig = figure("Mean vorticity in " + ROI)
ax = fig.add_subplot(111)
ax.plot(vw.t, series)
ax.set_xlabel("time (s)"); ax.set_ylabel("vorticity (1/s)")
results["vorticity_mean_" + ROI] = series

# along a line
L = line("LINE1", "U")                     # values (npts, nt), s, x, z
results["U_LINE1_time_mean"] = np.nanmean(L["values"], axis=1)

# a file in the run's PostProcessing folder
np.savetxt(post_dir + "/vorticity_mean.txt", series)
```

Good practice for scripts:

- parameters at the top, in capitals, each with a comment;
- check what is needed and stop with a clear error: `if not
  d.can_calibrate: raise RuntimeError("calibrate the run first")`;
- use the stored quantities (`quantity`, `line`), do not recompute them;
- put every result in `results` (arrays, numbers, strings, dicts of
  them), with explicit names and units in the name when useful;
- one `figure(name)` per figure; do not call `plt.show()` (no pyplot);
- do not change `d.U_px` / `d.V_px`;
- avoid long loops in pure Python over frames: the GUI waits while the
  script runs (numpy over the whole (ny, nx, nt) array is fast).

### C.6 Helping a user with a script (for an AI assistant)

1. Ask (or infer) what quantity, region (ROI / line / full), frames and
   output (figure, numbers, file) the user wants.
2. Start from the closest example of `Example_Scripts/`; keep the
   parameters at the top.
3. Use only the names of C.2 and the functions of C.3; check names
   against `piv_gui_lib.py` when unsure (`grep "^def " piv_gui_lib.py`).
4. Explain how to run it: put the file in the root folder (or its
   `scripts` folder), Advanced > Python script, choose it, Execute; the
   output, the results and the figure appear in the GUI.

---------------------------------------------------------------------------

## D. Testing and handing over

- Test **offscreen under both bindings**:
  `QT_QPA_PLATFORM=offscreen QT_API=PySide6 python test.py` and
  `QT_API=PyQt5`. A test builds `PIV_GUI.MainWindow()`, loads a run
  (`load_panel.root.setText(root); load_panel.refresh(select_run=run);
  load_panel.load_btn.click()`, then process events until
  `main._thread is None`), drives the panels, checks values and
  `w.grab().save("x.png")` to look at the window. Stub
  `QMessageBox` / `QFileDialog` static methods; set
  `w._quit_asked = True` before `w.close()` (else the quit question
  blocks).
- `python -m pyflakes PIV_GUI.py piv_gui_lib.py gui_variables.py`.
- Check the numbers (e.g. a sinusoid of amplitude A gives an FFT peak of
  height A; a stored profile equals the one computed on the fly).
- Update `README.md` with every change; keep this file right; mark the
  items done in `TODO.md` (the user's file: merge, never overwrite his
  edits).
- Do not change `gui_export.py` (shared).
