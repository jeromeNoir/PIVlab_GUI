"""
piv_gui_lib -- the library behind the PIVlab desktop GUI (PIV_GUI.py).

Everything here is independent of Qt, so it can be used, tested and scripted
on its own; PIV_GUI.py only builds the window around it.

Layout
------
  parameter file   param_GUI*.json in the root folder (separate from the
                   notebooks' param_postProcessing.json): DEFAULT_PARAMS,
                   list_param_files, read_param_file, write_param_file,
                   default_params, is_calibrated
  folders          list_runs (the run sub-folders of the root folder)
  acquisition log  read_acquisition_log -> AcqInfo (dt between the two
                   images of a pair, camera fps, PIV field rate)
  PIVlab .mat      load_piv_mat (MATLAB v5 and v7.3/HDF5, 3-D arrays or
                   cell arrays, validated or original velocity),
                   rotate_fields
  loaded run       PIVData = the loaded run: grid, U, V, time axis, units,
                   calibration -- load_run builds it in one call
  run information  PIVData.info_sections -- the table shown after loading
                   (time base, grid, velocity, the .mat variables via
                   mat_variables, the log's last row)
  background       find_background_images, read_image, place_background
                   (a camera frame placed under the vectors, ROTATE applied)
  calibration      find_calibration_images, rotate_image_px,
                   segment_length, calibration_figure
  figures          display_figure (frame / mean / rms; ROI; quiver of any
                   vector field; background velocity norm or image),
                   display_data and sections_data (what Save data writes)

Conventions (same as the notebook toolchain)
--------------------------------------------
* Time base. The log gives `pulse_sep` [us] -- the separation between the
  two images of a pair, i.e. the velocity time base dt -- and `cam_fps`, the
  camera frame rate. PIVlab pairs frames_per_field (= 2) camera frames into
  one velocity field, so the PIV field rate is cam_fps / frames_per_field.
  The time base is NOT in the parameter file: it is kept in memory
  (TIME_BASE_DEFAULT; set in Pre-processing > Calibration > Time base) and
  shown in the run information.
* Geometry. The fields are flipped as PIVlab exports them (x left to right,
  z upward, V positive upward), then rotated by ROTATE (0, 90, 180, -90).
* Calibration. XSCALE/YSCALE [m/px] null -> the run stays in px and
  px/frame (uncalibrated); set -> m and m/s (velocity = px * XSCALE / dt).
  A missing/unreadable log, or a manual time base, uses the pulse
  separation and the camera fps in memory. The time between PIV fields is
  TSCALE = frames_per_field / camera fps (computed, not stored).
"""

import copy
import json
import os
import re

import numpy as np

# --------------------------------------------------------------------------- #
#  Parameter file
# --------------------------------------------------------------------------- #
PARAM_FILENAME = "param_GUI.json"

# None is written as JSON null. XSCALE/YSCALE/PTS_ROI/L/nu are filled in by
# the Calibration steps (V2+); until then the run is shown uncalibrated.
DEFAULT_PARAMS = {
    "PIV_FILENAME": "PIVlab_results_uncalibrated.mat",
    "LOG_FILENAME": "acquisition_log.txt",
    "XSCALE": None,
    "YSCALE": None,
    "ROTATE": 0,
    "PTS_ROI": None,
    "LINES": None,
    "MASKS": None,
    "MASK_APPLIED": None,
    "MASK_SELECTION": None,
    "CALIB_INFO": None,
    "L": None,
    "nu": None,
}

# What each parameter means (tooltips in the GUI).
PARAM_HELP = {
    "PIV_FILENAME": "PIVlab result file in each run folder",
    "LOG_FILENAME": "acquisition log in each run folder (pulse_sep, cam_fps)",
    "XSCALE": "m/px along x; null = uncalibrated (px)",
    "YSCALE": "m/px along z; null = uncalibrated (px)",
    "ROTATE": "rotation of the fields: 0, 90, 180 or -90 deg",
    "PTS_ROI": "ROI corners [[x1, z1], [x2, z2]] in m; null = none yet",
    "LINES": "lines for the extraction of data, in m: null (none) or "
             "{name: [[x1, z1], [x2, z2]]} (Pre-processing > Define ROI > "
             "ROI Line)",
    "MASKS": "the masks built in Pre-processing > Mask: null (none) or "
             "{name: {\"colour\": ..., \"areas\": [{\"name\", \"kind\" "
             "(poly, rect, circle), \"points_px\"}]}} -- points in px of "
             "the run's grid (independent of the calibration); written "
             "whenever the masks change",
    "MASK_APPLIED": "the mask applied (its name), or null",
    "MASK_SELECTION": "the selection areas of Pre-processing > Mask "
                      "(name, kind, points_px, use)",
    "CALIB_INFO": "the calibration record (CALIB_IMAGE, CALIB_SEGMENT_PX, "
                  "CALIB_LENGTH_PX, CALIB_LENGTH_MM)",
    "L": "length scale [m]",
    "nu": "kinematic viscosity [m2/s]",
}


# no longer in the parameter file (older files: dropped when read)
REMOVED_KEYS = ("FRAMES_PER_FIELD", "PULSE_SEPARATION", "CAMERA_FPS",
                "MANUAL_TIME", "UNCAL_SCALE", "REGIONS", "VALIDATE_VELOCITY",
                "TSCALE", "TPIV_FIELDS", "UNCAL_DT", "UNCAL_FPS",
                "CALIB_IMAGE", "CALIB_SEGMENT_PX", "CALIB_LENGTH_PX",
                "CALIB_LENGTH_MM")

# per run (not in the parameter file): the post-processing folder and the
# run information file
POSTPROC_DIR = "PostProcessing"          # <run>/PostProcessing (notebooks)
RUN_INFO_FILE = "run_information.json"   # <run>/run_information.json
# the calibration record of a run (in the run information, not in the
# parameter file)
CALIB_KEYS = ("CALIB_IMAGE", "CALIB_SEGMENT_PX", "CALIB_LENGTH_PX",
              "CALIB_LENGTH_MM")

# The time base, kept in memory (not in the parameter file): used when the
# run has no readable log, or always when manual is True; frames_per_field
# is always used (PIV field rate = camera fps / frames_per_field).
TIME_BASE_DEFAULT = {"frames_per_field": 2, "pulse_separation": 1.0,
                     "camera_fps": 2.0, "manual": False}


def full_time_base(tb=None):
    """A complete time base: TIME_BASE_DEFAULT updated with `tb`."""
    out = dict(TIME_BASE_DEFAULT)
    out.update({k: v for k, v in (tb or {}).items() if k in out})
    out["frames_per_field"] = int(out["frames_per_field"])
    return out


def param_path(root):
    """The default parameter file of a root folder."""
    return os.path.join(root, PARAM_FILENAME)


def list_param_files(root):
    """GUI parameter files in the root folder: param_GUI*.json, sorted."""
    if not root or not os.path.isdir(root):
        return []
    return sorted(f for f in os.listdir(root)
                  if f.startswith("param_GUI") and f.endswith(".json")
                  and not f.startswith("."))


def default_params():
    return copy.deepcopy(DEFAULT_PARAMS)


def write_param_file(path, params):
    """Write a parameter file (atomic replace); return its path."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(params, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def read_param_file(path):
    """The file's values over DEFAULT_PARAMS (missing keys take the
    default)."""
    with open(path) as fh:
        raw = json.load(fh)
    params = default_params()
    params.update(raw)
    # keys no longer in the parameter file: the time base is kept in
    # memory, the others were removed (dropped here; the file loses them
    # when it is saved)
    for key in REMOVED_KEYS:
        params.pop(key, None)
    return params


def read_params(root, create=True):
    """(params, created) from root/param_GUI.json; the default file is
    written first when it is missing and create=True."""
    path = param_path(root)
    if not os.path.isfile(path):
        if not create:
            return default_params(), False
        write_param_file(path, DEFAULT_PARAMS)
        return read_param_file(path), True
    return read_param_file(path), False


def is_calibrated(params):
    return params.get("XSCALE") not in (None, 0) and \
        params.get("YSCALE") not in (None, 0)


# --------------------------------------------------------------------------- #
#  Folders
# --------------------------------------------------------------------------- #
def _visible_dirs(folder):
    if not folder or not os.path.isdir(folder):
        return []
    out = []
    for d in sorted(os.listdir(folder)):
        p = os.path.join(folder, d)
        low = d.lower()
        if os.path.isdir(p) and not d.startswith(".") \
                and not low.startswith("obsolete") \
                and low not in ("postprocessing", "__pycache__"):
            out.append(d)
    return out


def list_runs(root, piv_filename=None):
    """[(run name, has PIV file)] for the sub-folders of the root folder
    (has PIV file is None when piv_filename is not given)."""
    return [(d, None if piv_filename is None else
             os.path.isfile(os.path.join(root, d, piv_filename)))
            for d in _visible_dirs(root)]


# --------------------------------------------------------------------------- #
#  Acquisition log
# --------------------------------------------------------------------------- #
class AcqInfo(object):
    """Time base of a run.

    dt        separation of the two images of a pair [s] (pulse_sep)
    cam_fps   camera frame rate [Hz]
    fps       PIV field rate [Hz] = cam_fps / frames_per_field
    tscale    time between PIV fields [s] = frames_per_field / cam_fps
    ok        True when dt and fps come from the log
    log_ok    True when the log was read (even if not used: manual)
    source    "log", "manual" or "parameter file (no log)"
    row       the log's last data row, all columns (strings)
    """

    def __init__(self, dt, cam_fps, fps, ok, row=None, message="",
                 log_ok=None, source=None):
        self.dt, self.cam_fps, self.fps, self.ok = dt, cam_fps, fps, ok
        self.row = row or {}
        self.message = message
        self.log_ok = ok if log_ok is None else log_ok
        self.source = source or ("log" if ok else "memory (no log)")
        self.frames_per_field = 2

    @property
    def tscale(self):
        """TSCALE: time between two PIV fields [s]."""
        return 1.0 / float(self.fps)


def read_acquisition_log(log_path, frames_per_field=2, pulse_sep=1.0,
                         camera_fps=2.0, manual=False):
    """AcqInfo from a tab-separated PIVlab acquisition log.

    The recording row is the LAST data row; pulse_sep is in microseconds.
    Without a readable log, or with manual=True, dt = pulse_sep [s] and the
    camera rate = camera_fps [Hz] (the log is still read for display).
    """
    fpf = float(frames_per_field)
    acq = _read_log(log_path, fpf, pulse_sep, camera_fps)
    if manual:
        acq = AcqInfo(float(pulse_sep), float(camera_fps),
                      float(camera_fps) / fpf, False, row=acq.row,
                      message=acq.message + "; manual time base used",
                      log_ok=acq.log_ok, source="manual")
    acq.frames_per_field = int(fpf)
    return acq


def _read_log(log_path, frames_per_field, pulse_sep, camera_fps):
    fail = lambda msg: AcqInfo(float(pulse_sep), float(camera_fps),
                               float(camera_fps) / frames_per_field, False,
                               message=msg)
    if not os.path.isfile(log_path):
        return fail("no log file %s" % os.path.basename(log_path))
    try:
        with open(log_path, errors="replace") as fh:
            lines = [ln.rstrip("\r\n") for ln in fh if ln.strip()]
        header = [h.strip() for h in lines[0].split("\t")]
        i_pulse = header.index("pulse_sep")
        i_fps = header.index("cam_fps")
    except (IndexError, ValueError, OSError) as exc:
        return fail("log unreadable (%s)" % exc)
    for ln in reversed(lines[1:]):
        cols = ln.split("\t")
        try:
            dt = float(cols[i_pulse]) * 1e-6
            cam_fps = float(cols[i_fps])
        except (IndexError, ValueError):
            continue
        row = dict(zip(header, cols))
        return AcqInfo(dt, cam_fps, cam_fps / float(frames_per_field), True,
                       row=row, message="log read")
    return fail("no valid data row in the log")


# --------------------------------------------------------------------------- #
#  PIVlab .mat
# --------------------------------------------------------------------------- #
FILTERED_NAMES = (("u_filt", "v_filt"), ("u_filtered", "v_filtered"))
UNFILTERED_NAMES = (("u", "v"), ("u_original", "v_original"))
TYPEVECTOR_NAMES = ("typevector_filt", "typevector_filtered")
_WANTED = (["x", "y"] + [n for p in FILTERED_NAMES + UNFILTERED_NAMES
                         for n in p] + list(TYPEVECTOR_NAMES))


def _read_mat(path, wanted=_WANTED):
    """{name: array or list of 2-D arrays} for the PIV variables `wanted`
    (only those are read).

    MATLAB v5 files via scipy; v7.3 (HDF5) files via h5py, where cell arrays
    are references to per-frame datasets and every array is transposed
    (column-major on disk).
    """
    try:
        import h5py
        is_h5 = h5py.is_hdf5(path)
    except ImportError:
        is_h5 = False
    if not is_h5:
        from scipy.io import loadmat
        mat = loadmat(path, variable_names=list(wanted))
        return {k: v for k, v in mat.items() if not k.startswith("__")}
    out = {}
    with h5py.File(path, "r") as f:
        for name in wanted:
            if name not in f:
                continue
            ds = f[name]
            if ds.dtype == h5py.ref_dtype:      # cell array
                out[name] = [np.asarray(f[r]).T for r in ds[()].ravel()]
            else:
                out[name] = np.asarray(ds).T
    return out


def _as_frames(a):
    """(ny, nx, nt) float stack from a 3-D array or a cell array of frames."""
    if isinstance(a, list):
        return np.stack([np.asarray(c, float) for c in a], axis=-1)
    a = np.asarray(a)
    if a.dtype == object:
        return np.stack([np.asarray(c, float) for c in a.ravel()], axis=-1)
    a = a.astype(float)
    return a[:, :, None] if a.ndim == 2 else a


def _as_grid(a):
    if isinstance(a, list):
        return np.asarray(a[0], float)
    a = np.asarray(a)
    if a.dtype == object:
        return np.asarray(a.ravel()[0], float)
    return (a[:, :, 0] if a.ndim == 3 else a).astype(float)


def _first(mat, pairs):
    for a, b in pairs:
        if a in mat and b in mat:
            return a, b
    return None


def mat_variables(path):
    """[(name, size, class)] of every variable in a PIVlab .mat, read from
    the file's directory without loading the data. Cell arrays are shown
    as e.g. '1x300 cell'."""
    out = []
    try:
        import h5py
        is_h5 = h5py.is_hdf5(path)
    except ImportError:
        is_h5 = False
    if not is_h5:
        from scipy.io import whosmat
        for name, shape, cls in whosmat(path):
            out.append((name, "x".join(str(s) for s in shape), cls))
        return out
    import h5py
    with h5py.File(path, "r") as f:
        for name in f:
            if name.startswith("#"):
                continue                        # #refs#, #subsystem#
            obj = f[name]
            if not isinstance(obj, h5py.Dataset):
                out.append((name, "", "struct/group"))
                continue
            shape = tuple(reversed(obj.shape))  # MATLAB order
            cls = obj.attrs.get("MATLAB_class", b"")
            cls = cls.decode() if isinstance(cls, bytes) else str(cls)
            if obj.dtype == h5py.ref_dtype:
                cls = "cell"
            out.append((name, "x".join(str(s) for s in shape),
                        cls or str(obj.dtype)))
    return out


VALIDATED = "Validated"
UNVALIDATED = "Unvalidated"
REJECTED_NAN = "Unvalidated, rejected = NaN"
# names used before (sessions)
OLD_FIELD_NAMES = {"validated (filtered)": VALIDATED,
                   "unvalidated (original)": UNVALIDATED,
                   "unvalidated, rejected = NaN": REJECTED_NAN}


def available_velocities(names):
    """The velocities a .mat holds, from its variable names: [VALIDATED,
    UNVALIDATED, REJECTED_NAN] (those present)."""
    names = set(names)
    has = lambda pairs: any(a in names and b in names for a, b in pairs)
    out = []
    if has(FILTERED_NAMES):
        out.append(VALIDATED)
    if has(UNFILTERED_NAMES):
        out.append(UNVALIDATED)
        if any(n in names for n in TYPEVECTOR_NAMES):
            out.append(REJECTED_NAN)
    return out


def load_piv_mat(path, velocity=VALIDATED, names=None):
    """ONE velocity of a PIVlab .mat, in px and px/frame: only the
    variables it needs are read.

    velocity  VALIDATED    PIVlab's filtered velocity (u_filt /
                           u_filtered), NaN where the original vector is
                           NaN (the original velocity is read for that
                           mask, then dropped);
              UNVALIDATED  the original velocity (u / u_original);
              REJECTED_NAN the original velocity with NaN where
                           typevector_filt != 1 (rejected / interpolated or
                           masked).
              When the file does not hold it, the first available one is
              read instead.
    names     the variable names of the file (mat_variables), if known.

    Returns X, Y (ny, nx), U, V (ny, nx, nt), the velocity read, reject
    (bool (ny, nx, nt) for REJECTED_NAN, else None), its note, the
    velocities available in the file, and y_top_px, the raw image row of
    the top vector (to place a background image). The flips match the
    notebooks' load_piv: x increasing to the right, z (Y) increasing upward
    from 0, V positive upward.
    """
    if names is None:
        try:
            names = [n for n, _s, _c in mat_variables(path)]
        except Exception:
            names = list(_WANTED)
    avail = available_velocities(names)
    if not avail:
        raise ValueError("no velocity field in %s" % os.path.basename(path))
    name = velocity if velocity in avail else avail[0]
    wanted = ["x", "y"]
    if name in (UNVALIDATED, REJECTED_NAN, VALIDATED):
        wanted += [n for p in UNFILTERED_NAMES for n in p]
    if name == VALIDATED:
        wanted += [n for p in FILTERED_NAMES for n in p]
    if name == REJECTED_NAN:
        wanted += list(TYPEVECTOR_NAMES)
    mat = _read_mat(path, [n for n in wanted if n in set(names)] or wanted)
    if "x" not in mat or "y" not in mat:
        raise ValueError("no x/y grid in %s" % os.path.basename(path))
    filt, orig = _first(mat, FILTERED_NAMES), _first(mat, UNFILTERED_NAMES)
    X = _as_grid(mat["x"])[:, ::-1]
    Y = _as_grid(mat["y"])[::-1, :]

    def fr(pair):
        u = _as_frames(mat.pop(pair[0]))[::-1, ::-1, :]
        v = -_as_frames(mat.pop(pair[1]))[::-1, ::-1, :]       # V up
        return (np.ascontiguousarray(u, np.float32),
                np.ascontiguousarray(v, np.float32))

    reject = None
    if name == VALIDATED:
        U, V = fr(filt)
        if orig is not None:                      # NaN mask of the original
            Uo, Vo = fr(orig)
            if Uo.shape == U.shape:
                bad = np.isnan(Uo) | np.isnan(Vo)
                U[bad] = np.nan
                V[bad] = np.nan
            del Uo, Vo
        note = "PIVlab filtered velocity (%s, %s)" % filt
    else:
        U, V = fr(orig)
        note = "PIVlab original velocity (%s, %s)" % orig
        if name == REJECTED_NAN:
            tv = next(n for n in TYPEVECTOR_NAMES if n in mat)
            T = _as_frames(mat.pop(tv))[::-1, ::-1, :]
            if T.shape == U.shape:
                reject = T != 1
                U[reject] = np.nan
                V[reject] = np.nan
                note = ("PIVlab original velocity (%s, %s), NaN where %s != "
                        "1 (%.2f %% of the vectors)"
                        % (orig + (tv, 100 * reject.mean())))
    y_top_px = float(np.nanmax(Y))        # raw image row of the top vector
    Y = y_top_px - Y
    return X, Y, U, V, name, reject, note, avail, y_top_px


def rotate_fields(X, Y, U, V, rotate):
    """Rotate grid and velocity by 0, 180, 90 or -90 deg (the notebooks'
    rotate_fields): coordinates stay positive, 0 at the new left/bottom."""
    rotate = int(rotate)
    if rotate == 0:
        return X, Y, U, V
    xm, zm = np.nanmax(X), np.nanmax(Y)
    if rotate == 180:
        return xm - X, zm - Y, -U, -V
    T = lambda A: np.ascontiguousarray(np.swapaxes(A, 0, 1))
    if rotate == -90:
        return T(zm - Y), T(X), T(-V), T(U)
    if rotate == 90:
        return T(Y), T(xm - X), T(V), T(-U)
    raise ValueError("ROTATE must be 0, 90, 180 or -90, got %r" % rotate)


# --------------------------------------------------------------------------- #
#  A loaded run
# --------------------------------------------------------------------------- #
def parse_run_name(name):
    """(frot, flib, dphi) from 'frot0.50Hz_flib0.400Hz_dphi2deg_SS1'; NaN
    for missing tokens."""
    g = lambda pat: (float(re.search(pat, name).group(1))
                     if re.search(pat, name) else np.nan)
    return (g(r"frot([\d.]+)Hz"), g(r"flib([\d.]+)Hz"), g(r"dphi([\d.]+)deg"))


def parse_shapes(value, default="ROI"):
    """[(name, ((x1, z1), (x2, z2)))] from a PTS_ROI / LINES value: null
    (none), one pair of points [[x1, z1], [x2, z2]] (named `default`), a
    list of them ('<default> 1', ...) or a dict {name: [[x1, z1], [x2,
    z2]]}. Rectangles (two opposite corners) or segments (two ends)."""
    if not value:
        return []
    if isinstance(value, dict):
        items = list(value.items())
    elif np.asarray(value, dtype=object).shape == (2, 2) and \
            np.isscalar(np.asarray(value, dtype=object)[0, 0]):
        items = [(default, value)]
    else:
        items = [("%s %d" % (default, i + 1), r) for i, r in enumerate(value)]
    out = []
    for name, r in items:
        try:
            (x1, z1), (x2, z2) = np.asarray(r, float)
            out.append((str(name), ((float(x1), float(z1)),
                                    (float(x2), float(z2)))))
        except (TypeError, ValueError):
            continue
    return out


def shapes_to_param(shapes, ndigits=6):
    """The PTS_ROI / LINES value of [(name, ((x1, z1), (x2, z2)))]: a dict
    {name: [[x1, z1], [x2, z2]]} (None when empty)."""
    out = {}
    for name, ((x1, z1), (x2, z2)) in shapes:
        out[str(name)] = [[round(float(x1), ndigits), round(float(z1), ndigits)],
                          [round(float(x2), ndigits), round(float(z2), ndigits)]]
    return out or None


MASK_TOOLS = [("Polygon", "poly"), ("Rectangle", "rect"), ("Circle", "circle")]
# one shading colour per mask (in the order the masks are made)
MASK_COLOURS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
                "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]


def area_mask(Xp, Zp, kind, pts):
    """True on the grid points (Xp, Zp) inside an area: 'rect' (two
    opposite corners), 'circle' (centre, a point of the circle) or 'poly'
    (its vertices)."""
    p = np.asarray(pts, float)
    if kind == "rect":
        (x1, z1), (x2, z2) = p[:2]
        return ((Xp >= min(x1, x2)) & (Xp <= max(x1, x2))
                & (Zp >= min(z1, z2)) & (Zp <= max(z1, z2)))
    if kind == "circle":
        (xc, zc), (xe, ze) = p[:2]
        return np.hypot(Xp - xc, Zp - zc) <= np.hypot(xe - xc, ze - zc)
    if kind == "poly":
        from matplotlib.path import Path
        if len(p) < 3:
            return np.zeros(Xp.shape, bool)
        return Path(p).contains_points(
            np.c_[Xp.ravel(), Zp.ravel()]).reshape(Xp.shape)
    raise ValueError("unknown area kind %r" % kind)


def masks_to_param(d, ndigits=2):
    """The MASKS value of a run's masks: {name: {colour, areas}} (None when
    there is none)."""
    out = {}
    for name, m in d.masks.items():
        out[name] = {"colour": m["colour"], "areas": [
            {"name": a.get("name"), "kind": a["kind"],
             "points_px": [[round(float(x), ndigits), round(float(z), ndigits)]
                           for x, z in a["points_px"]]}
            for a in m["areas"]]}
    return out or None


def build_mask(d, areas):
    """The mask (bool (ny, nx), True = masked) of a run from areas
    [{'kind', 'points_px'}] in px of the run's grid (X_px, Z_px); None when
    there is none."""
    m = np.zeros(d.X_px.shape, bool)
    for a in areas:
        m |= area_mask(d.X_px, d.Z_px, a["kind"], a["points_px"])
    return m if m.any() else None


class RunView(object):
    """A calibrated or uncalibrated view of a loaded run (PIVData): the
    positions, velocities and times computed from the uncalibrated data in
    memory (px, px/frame) when they are asked for -- nothing is stored, so
    any number of views costs no memory.

    calibrated   True: m, m/s, s (XSCALE / YSCALE and the pulse
                 separation applied); only when the run can be calibrated,
                 otherwise the view is uncalibrated
    x, z         (ny, nx) positions;  u, v (ny, nx, nt) velocity;
    t            (nt,) times [s], or frame numbers when uncalibrated
    uv(frame)    (u, v) of one frame (or all frames: frame=None)
    """

    def __init__(self, run, calibrated=True, window=None):
        self.run = run
        self.calibrated = bool(calibrated) and run.can_calibrate
        # (i0, i1, j0, j1): the rows / columns of an ROI field (a window of
        # the stored data: no copy); None = the full field
        self.window = window

    @property
    def _sl(self):
        if self.window is None:
            return (slice(None), slice(None))
        i0, i1, j0, j1 = self.window
        return (slice(i0, i1), slice(j0, j1))

    lunit = property(lambda self: "m" if self.calibrated else "px")
    vunit = property(lambda self: "m/s" if self.calibrated else "px/frame")
    tunit = property(lambda self: "s" if self.calibrated else "frame")

    @property
    def pix_scale(self):
        """(XSCALE, YSCALE) [m/px] applied, (1, 1) when uncalibrated."""
        r = self.run
        return (r.xs, r.ys) if self.calibrated else (1.0, 1.0)

    @property
    def vel_scale(self):
        """(ku, kv): px/frame -> the view's unit (XSCALE / pulse
        separation)."""
        r = self.run
        if not self.calibrated:
            return 1.0, 1.0
        return r.xs / r.acq.dt, r.ys / r.acq.dt

    @property
    def x(self):
        return self.run.X_px[self._sl] * self.pix_scale[0]

    @property
    def z(self):
        return self.run.Z_px[self._sl] * self.pix_scale[1]

    @property
    def mask(self):
        """The mask on the view's grid (True = masked), or None."""
        m = self.run.mask
        return None if m is None else m[self._sl]

    def uv(self, frame=None):
        """(u, v): all frames, or one frame index (a copy when calibrated
        or masked -- NaN where the mask is --, a read-only view of the
        stored data otherwise)."""
        r = self.run
        sl = self._sl + ((slice(None),) if frame is None else (int(frame),))
        u, v = r.U_px[sl], r.V_px[sl]
        m = self.mask
        if m is not None and m.any():
            mm = m if frame is not None else m[:, :, None]
            u = np.where(mm, np.float32(np.nan), u)
            v = np.where(mm, np.float32(np.nan), v)
        ku, kv = self.vel_scale
        if ku != 1.0 or kv != 1.0:
            return u * np.float32(ku), v * np.float32(kv)
        u, v = u.view(), v.view()
        u.flags.writeable = v.flags.writeable = False   # protect the data
        return u, v

    @property
    def u(self):
        return self.uv()[0]

    @property
    def v(self):
        return self.uv()[1]

    @property
    def t(self):
        r = self.run
        return (np.arange(r.nt) * r.dt_field if self.calibrated
                else np.arange(r.nt, dtype=float))


class PIVData(object):
    """One loaded run: ONE velocity (the one chosen in Load data), stored
    once in px and px/frame; the calibration (m, m/s, s) is applied on the
    fly when it is shown calibrated, so switching between calibrated and
    uncalibrated (Display) costs no memory.

    x, z         (ny, nx) positions [m when shown calibrated, else px]
    u, v         (ny, nx, nt) the velocity [m/s, or px/frame]
    default      the velocity loaded (VALIDATED, UNVALIDATED or
                 REJECTED_NAN); field_names = [default]
    available    the velocities the .mat holds
    t            (nt,) field times [s when calibrated, else frame numbers]
    can_calibrate  XSCALE / YSCALE are set in the parameters
    calibrated   shown calibrated now (set_calibrated)
    acq          AcqInfo; params: the values of the parameter file used
    """

    def __init__(self, run_dir, params, X, Y, U, V, name, reject, note, acq,
                 available=None, piv_path=None, log_path=None,
                 variables=None, y_top_px=None, grid_max=None,
                 calibrate=True, velocity=None):
        self.run_dir = run_dir
        # background-image geometry: raw top row [px] and the unrotated
        # grid maxima [px] (the offsets rotate_fields used)
        self.y_top_px = y_top_px
        self.grid_max = grid_max
        self.piv_path, self.log_path = piv_path, log_path
        self.variables = variables or []
        self.run = os.path.basename(run_dir.rstrip("/"))
        self.dataset_dir = os.path.dirname(run_dir.rstrip("/"))
        self.params = params
        self.acq = acq
        # the data, in px and px/frame (stored once)
        self.X_px = np.asarray(X, float)
        self.Z_px = np.asarray(Y, float)
        self.U_px = np.asarray(U, np.float32)
        self.V_px = np.asarray(V, np.float32)
        self.reject = reject
        self.default = name
        self.field_names = [name]
        self.available = list(available or [name])
        self.notes = {name: note}
        self.velocity_asked = velocity
        self.ny, self.nx, self.nt = self.U_px.shape
        self.valid = float(np.isfinite(self.U_px).mean())
        # calibration: XSCALE / YSCALE (px -> m) and the pulse separation
        # (px/frame -> m/s, field index -> s with TSCALE)
        self.can_calibrate = is_calibrated(params)
        if self.can_calibrate:
            self.xs, self.ys = float(params["XSCALE"]), float(params["YSCALE"])
        else:
            self.xs = self.ys = 1.0
        self.calibration_asked = bool(calibrate)
        self.calibrated = False
        self.set_calibrated(calibrate)
        self.dt_field = 1.0 / float(acq.fps)     # TSCALE [s]
        self.frot, self.flib, self.dphi = parse_run_name(self.run)
        self.log_skipped = False
        self.mask = None                         # (ny, nx) bool or None
        self.mask_areas = []
        self.mask_record = {"areas": []}         # the selection areas
        self.masks = {}          # name -> {areas, colour, mask (ny, nx)}
        self.mask_name = None    # the mask applied (Maps and Quiver > Mask)
        self.calib_info = {k: None for k in CALIB_KEYS}
        self.post_dir = None                     # set by the GUI

    # ---- calibrated / uncalibrated views --------------------------------- #
    # Rule for every step: the data are stored ONCE, uncalibrated (px,
    # px/frame); a step works on a view, calibrated by default (the
    # calibration factors applied when the arrays are asked for, nothing
    # stored), uncalibrated only when the step offers that option (as
    # Processing > Maps and Quiver > Velocity).
    def view(self, calibrated=True, field=None):
        """A RunView of the run for a step: calibrated (m, m/s, s) by
        default -- when XSCALE / YSCALE are set -- or uncalibrated (px,
        px/frame, frames). Independent of the Maps and Quiver display. field: an
        ROI field (roi_fields) -- the velocity extracted in the ROI -- or
        None / 'Full' for the full field."""
        win = None
        if field and field != "Full":
            info = self.roi_fields().get(field)
            win = info["window"] if info else None
        return RunView(self, calibrated, win)

    # ---- ROI fields and mask --------------------------------------------- #
    def roi_fields(self):
        """{field name: info} -- one field per ROI of PTS_ROI: the velocity
        extracted in the ROI (the rows / columns of the grid inside it: a
        window of the stored data, no copy), named '<velocity>_ROIxx' (xx:
        the number of the ROI, 01, 02, ...). info: roi (its name), number,
        window (i0, i1, j0, j1), rect ((x1, z1), (x2, z2)) in m, shape.
        ROIs are in m: none when the run cannot be calibrated."""
        if not self.can_calibrate:
            return {}
        key = json.dumps(self.params.get("PTS_ROI"), sort_keys=True,
                         default=str) + self.default
        if getattr(self, "_roi_key", None) == key:
            return self._roi_fields
        X, Z = self.X_px * self.xs, self.Z_px * self.ys
        out = {}
        for k, (name, ((x1, z1), (x2, z2))) in enumerate(
                parse_shapes(self.params.get("PTS_ROI"), "ROI"), 1):
            inside = ((X >= min(x1, x2)) & (X <= max(x1, x2))
                      & (Z >= min(z1, z2)) & (Z <= max(z1, z2)))
            if not inside.any():
                continue
            rows = np.where(inside.any(axis=1))[0]
            cols = np.where(inside.any(axis=0))[0]
            win = (int(rows[0]), int(rows[-1]) + 1, int(cols[0]),
                   int(cols[-1]) + 1)
            out["%s_ROI%02d" % (self.default, k)] = dict(
                roi=roi_name(k), number=k, window=win,
                rect=((x1, z1), (x2, z2)),
                shape=(win[1] - win[0], win[3] - win[2]))
        self._roi_key, self._roi_fields = key, out
        return out

    def set_mask(self, mask, areas=None):
        """The mask applied (bool (ny, nx), True = masked: the velocity
        there is NaN for every step), or None; areas: what built it."""
        self.mask = None if mask is None or not np.any(mask) else \
            np.asarray(mask, bool)
        self.mask_areas = list(areas or []) if self.mask is not None else []
        good = np.isfinite(self.U_px)
        if self.mask is not None:
            good = good & ~self.mask[:, :, None]
        self.valid = float(good.mean())

    # ---- several masks: built in Pre-processing > Mask, one applied ------- #
    def store_mask(self, name, areas, colour=None):
        """Add or replace the mask `name` built from `areas` [{name, kind,
        points_px}] (the boolean array is rebuilt); applied again when it
        is the one applied."""
        old = self.masks.get(name, {})
        self.masks[name] = {
            "areas": [dict(a) for a in areas],
            "colour": colour or old.get("colour") or MASK_COLOURS[
                len(self.masks) % len(MASK_COLOURS)],
            "mask": build_mask(self, areas) if areas else None}
        if self.mask_name == name:
            self.apply_mask(name)

    def rename_mask(self, old, new):
        self.masks = {(new if k == old else k): v
                      for k, v in self.masks.items()}
        if self.mask_name == old:
            self.mask_name = new

    def remove_mask(self, name):
        self.masks.pop(name, None)
        if self.mask_name == name:
            self.apply_mask(None)

    def apply_mask(self, name):
        """Apply the mask `name` (velocity NaN in it for every step), or
        none (None)."""
        rec = self.masks.get(name) if name else None
        self.mask_name = name if rec is not None else None
        self.params["MASK_APPLIED"] = self.mask_name   # kept at a reload
        self.set_mask(rec["mask"] if rec else None,
                      rec["areas"] if rec else None)

    @property
    def display(self):
        """The view shown by Processing > Maps and Quiver (calibrated or not)."""
        return RunView(self, self.calibrated)

    def set_calibrated(self, on):
        """Processing > Maps and Quiver > Velocity: show the run calibrated (m, m/s,
        s) -- possible only when XSCALE / YSCALE are set -- or
        uncalibrated (px, px/frame, frames). Only the Maps and Quiver display."""
        self.calibrated = bool(on) and self.can_calibrate
        return self.calibrated

    # the Maps and Quiver display view, as attributes (used by the Maps and Quiver figure)
    lunit = property(lambda self: self.display.lunit)
    vunit = property(lambda self: self.display.vunit)
    tunit = property(lambda self: self.display.tunit)
    pix_scale = property(lambda self: self.display.pix_scale)
    vel_scale = property(lambda self: self.display.vel_scale)
    x = property(lambda self: self.display.x)
    z = property(lambda self: self.display.z)
    u = property(lambda self: self.display.u)
    v = property(lambda self: self.display.v)
    t = property(lambda self: self.display.t)

    @property
    def fields(self):
        return {self.default: (self.U_px, self.V_px)}

    def calibration_text(self):
        P = self.params
        if not self.can_calibrate:
            return ("not available (XSCALE / YSCALE null) -> px, px/frame, "
                    "frames")
        cal = "XSCALE = %s, YSCALE = %s m/px, pulse separation %.6g s" % (
            P["XSCALE"], P["YSCALE"], self.acq.dt)
        if self.calibrated:
            return "shown calibrated: %s -> m, m/s, s" % cal
        return "available (%s), shown uncalibrated -> px, px/frame, frames" \
            % cal

    def frame_text(self, i):
        """'frame i / last' and, when calibrated, its time."""
        i = int(i)
        txt = "frame %d / %d" % (i, self.nt - 1)
        if self.calibrated:
            txt += "   t = %.3f s" % self.t[i]
        return txt

    def uv(self, name=None, frame=None):
        """(u, v) of the velocity loaded -- all frames, or one frame index
        -- in the unit of the Maps and Quiver display (`name` is kept for
        compatibility: only one velocity is in memory)."""
        return self.display.uv(frame)

    @property
    def note(self):
        return self.notes.get(self.default, self.default)

    def rois(self):
        """[(name, ((x1, z1), (x2, z2)))] -- the ROIs of the parameter file.

        PTS_ROI is null (none), one rectangle [[x1, z1], [x2, z2]] (named
        'ROI'), a list of rectangles ('ROI 1', 'ROI 2', ...) or a dict
        {name: rectangle}. In the run's length unit (m when calibrated)."""
        if not self.calibrated:                  # ROIs are in metres
            return []
        return parse_shapes(self.params.get("PTS_ROI"), "ROI")

    def lines(self):
        """[(name, ((x1, z1), (x2, z2)))] -- the LINES of the parameter file
        (in m: only when shown calibrated)."""
        if not self.calibrated:
            return []
        return parse_shapes(self.params.get("LINES"), "LINE")

    def info_sections(self):
        """[(section title, [(quantity, value)])] -- the run information
        table shown after loading."""
        a = self.acq
        P = self.params
        fpf = int(a.frames_per_field)
        dx = float(np.nanmedian(np.abs(np.diff(self.x, axis=1))))
        dz = float(np.nanmedian(np.abs(np.diff(self.z, axis=0))))
        run = [
            ("run folder", self.run_dir),
            ("PIV file", "%s  (%.1f MB)" % (
                self.piv_path if os.path.dirname(self.piv_path or "") !=
                self.run_dir else os.path.basename(self.piv_path or ""),
                os.path.getsize(self.piv_path) / 1e6
                if self.piv_path and os.path.isfile(self.piv_path) else 0)),
            ("acquisition log file",
             "skipped (values entered by hand)" if self.log_skipped else
             (self.log_path if os.path.dirname(self.log_path or "") !=
              self.run_dir else os.path.basename(self.log_path or ""))),
            ("post-processing folder", self.post_dir or "--"),
            ("run information file", run_info_path(self.run_dir)),
        ]
        if np.isfinite(self.frot) or np.isfinite(self.flib):
            run.append(("from folder name", "f_rot = %g Hz, f_lib = %g Hz, "
                        "dphi = %g deg" % (self.frot, self.flib, self.dphi)))
        time = [
            ("acquisition log", "%s -- %s" % (
                os.path.basename(self.log_path or "") or "--", a.message)
             + ("" if a.ok else "; time base in memory used")),
            ("time base from", a.source
             + ("  (Pre-processing > Calibration > Time base)"
                if a.source == "manual" else "")),
            ("number of PIV fields", "%d" % self.nt),
            ("time between frames dt (pulse separation)", "%.6g s" % a.dt),
            ("camera frame rate", ("%.6g Hz" % a.cam_fps)
             + ("" if a.ok else " (in memory)")),
            ("frames per PIV field", "%d" % fpf),
            ("PIV field sampling frequency",
             ("%.6g Hz  (= camera fps / %d)" % (a.fps, fpf)) if a.ok
             else "%.6g Hz (camera fps in memory / %d)" % (a.fps, fpf)),
            ("time between PIV fields TSCALE",
             "%.6g s  (= %d / camera fps)" % (a.tscale, fpf)),
            ("record length", "%.6g s" % (self.nt / a.fps)),
            ("Nyquist frequency", "%.6g Hz" % (0.5 * a.fps)),
            ("frequency resolution", "%.6g Hz" % (a.fps / self.nt)),
        ]
        space = [
            ("grid", "%d x %d vectors (z x x)" % (self.ny, self.nx)),
            ("vector spacing", "dx = %.4g %s, dz = %.4g %s"
             % (dx, self.lunit, dz, self.lunit)),
            ("extent", "x: %.4g .. %.4g %s,  z: %.4g .. %.4g %s"
             % (np.nanmin(self.x), np.nanmax(self.x), self.lunit,
                np.nanmin(self.z), np.nanmax(self.z), self.lunit)),
            ("ROTATE", "%d deg" % int(P["ROTATE"])),
            ("calibration", self.calibration_text()),
            ("ROI", "; ".join("%s %s" % (n, [list(c) for c in r])
                              for n, r in parse_shapes(P.get("PTS_ROI"),
                                                       "ROI")) or "not set"),
            ("lines", "; ".join("%s %s" % (n, [list(c) for c in r])
                                for n, r in parse_shapes(P.get("LINES"),
                                                         "LINE"))
             or "not set"),
        ]
        sp = np.hypot(*self.uv())
        vel = [("velocities in the .mat", ", ".join(self.available)),
               ("velocity loaded (in memory)", self.default
                + ("" if self.velocity_asked in (None, self.default)
                   else "  (%s is not in the file)" % self.velocity_asked)),
               ("  source", self.notes.get(self.default, "")),
               ("memory", "%.1f MB" % ((self.U_px.nbytes + self.V_px.nbytes
                                        + (self.reject.nbytes
                                           if self.reject is not None
                                           else 0)) / 1e6))]
        rf = self.roi_fields()
        rq = getattr(self, "roi_quantities", {})
        vel += [
            ("ROI fields", "; ".join(
                "%s (%s, %d x %d vectors)" % (n, i["roi"], i["shape"][0],
                                              i["shape"][1])
                for n, i in rf.items()) or "none"),
            ("ROI quantities", ", ".join(sorted(rq)) + "  (%.1f MB)" % (
                sum(e["data"].nbytes for e in rq.values()) / 1e6)
             if rq else "none (Define ROI > Apply)"),
            ("mask applied", ("%s: %d vectors masked (%.1f %%), %d "
                              "area(s): %s -> velocity NaN" % (
                                  self.mask_name, int(self.mask.sum()),
                                  100 * self.mask.mean(),
                                  len(self.mask_areas),
                                  ", ".join(str(a.get("name")) for a in
                                            self.mask_areas)))
             if self.mask is not None else "none"),
            ("masks", "; ".join(
                "%s (%d area(s), %d vectors)" % (
                    n, len(m["areas"]), int(np.sum(m["mask"]))
                    if m["mask"] is not None else 0)
                for n, m in self.masks.items()) or "none"),
            ("valid vectors", "%.2f %%" % (100 * self.valid)),
            ("|u| median / max", "%.4g / %.4g %s"
             % (np.nanmedian(sp), np.nanmax(sp), self.vunit)),
        ]
        ci = self.calib_info
        seg = ci.get("CALIB_SEGMENT_PX")
        cal = [
            ("XSCALE / YSCALE", "%s / %s m/px" % (P.get("XSCALE"),
                                                  P.get("YSCALE"))
             if self.can_calibrate else "not set"),
            ("shown", "calibrated (m, m/s, s)" if self.calibrated
             else "uncalibrated (px, px/frame, frames)"),
            ("rotation", "%d deg" % int(P.get("ROTATE") or 0)),
            ("CALIB_IMAGE", ci.get("CALIB_IMAGE") or "--"),
            ("CALIB_SEGMENT_PX", "%s" % seg if seg else "--"),
            ("CALIB_LENGTH_PX", "%s px" % ci["CALIB_LENGTH_PX"]
             if ci.get("CALIB_LENGTH_PX") else "--"),
            ("CALIB_LENGTH_MM", "%s mm" % ci["CALIB_LENGTH_MM"]
             if ci.get("CALIB_LENGTH_MM") else "--"),
        ]
        logrow = [(k, v) for k, v in a.row.items()]
        out = [("Run", run), ("Time base", time), ("Space", space),
               ("Calibration", cal), ("Velocity", vel)]
        if logrow:
            out.append(("Acquisition log (last row)", logrow))
        return out


def run_info_path(run_dir):
    return os.path.join(run_dir, RUN_INFO_FILE)


def read_run_info(run_dir):
    """The content of <run>/run_information.json ({} when there is none or
    it cannot be read)."""
    try:
        with open(run_info_path(run_dir)) as fh:
            out = json.load(fh)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def postproc_dir(run_dir, create=True):
    """(path, created) of <run>/PostProcessing, created when missing."""
    path = os.path.join(run_dir, POSTPROC_DIR)
    created = False
    if create and not os.path.isdir(path):
        os.makedirs(path)
        created = True
    return path, created


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def write_run_info(d, sections, param_file=None, setup_files=None):
    """Write <run>/run_information.json: the run information table (as
    shown) and, machine-readable, the files used, the time base and the
    calibration (with the CALIB_ record). Returns its path."""
    import datetime
    a = d.acq
    out = {
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "run": d.run,
        "run_information": {t: {k: str(v) for k, v in rows}
                            for t, rows in sections},
        "files": {"run_folder": d.run_dir,
                  "calibration_file": (setup_files or {}).get("calibration"),
                  "roi_file": (setup_files or {}).get("roi"),
                  "mask_file": (setup_files or {}).get("mask"),
                  "piv_file": d.piv_path,
                  "log_file": None if d.log_skipped else d.log_path,
                  "log_skipped": bool(d.log_skipped),
                  "post_processing_folder": d.post_dir},
        "time_base": {"source": a.source,
                      "manual": a.source == "manual",
                      "frames_per_field": int(a.frames_per_field),
                      "pulse_separation_s": float(a.dt),
                      "camera_fps_Hz": float(a.cam_fps),
                      "PIV_field_rate_Hz": float(a.fps),
                      "TSCALE_s": float(a.tscale)},
        "calibration": dict(
            {"XSCALE": d.params.get("XSCALE"),
             "YSCALE": d.params.get("YSCALE"),
             "shown_calibrated": bool(d.calibrated)},
            **{k: d.calib_info.get(k) for k in CALIB_KEYS}),
        # Pre-processing > Mask: the areas (points in px of the run's
        # grid; use = ticked) and whether the mask is applied
        "mask": {"selection_areas": list(d.mask_record.get("areas")
                                          or []),
                 "masks": [{"name": n, "colour": m["colour"],
                            "areas": m["areas"],
                            "vectors": int(np.sum(m["mask"]))
                            if m["mask"] is not None else 0}
                           for n, m in d.masks.items()],
                 "applied": d.mask_name,
                 "masked_vectors": int(d.mask.sum())
                 if d.mask is not None else 0},
        "roi_fields": {n: {"roi": i["roi"], "window_rows_cols":
                           list(i["window"]), "shape": list(i["shape"])}
                       for n, i in d.roi_fields().items()},
        "roi_quantities": {n: {"roi": e["roi"], "quantity": e["quantity"],
                               "unit": e["unit"], "mask": e["mask"][0],
                               "shape": list(e["data"].shape)}
                           for n, e in sorted(getattr(
                               d, "roi_quantities", {}).items())},
    }
    path = run_info_path(d.run_dir)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(_jsonable(out), fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def load_run(run_dir, params=None, calibrate=True, velocity=None,
             time_base=None, piv_path=None, log_path=None):
    """PIVData for one run folder (params: the values of a parameter file;
    when not given, the root's param_GUI.json, created if missing).
    time_base: the time base in memory (see TIME_BASE_DEFAULT).
    piv_path: the PIVlab .mat (default <run>/PIV_FILENAME).
    log_path: the acquisition log (default <run>/LOG_FILENAME); False: no
    log (skipped by the user: the time base in memory is used)."""
    run_dir = run_dir.rstrip("/")
    if params is None:
        params, _ = read_params(os.path.dirname(run_dir))
    piv = piv_path or os.path.join(run_dir, params["PIV_FILENAME"])
    if not os.path.isfile(piv):
        raise FileNotFoundError("no %s in %s" % (os.path.basename(piv),
                                                 os.path.dirname(piv)))
    params = dict(params)
    tb = full_time_base(time_base)
    skipped = log_path is False
    if skipped:
        acq = AcqInfo(float(tb["pulse_separation"]), float(tb["camera_fps"]),
                      float(tb["camera_fps"]) / tb["frames_per_field"],
                      False, message="no log (skipped: values entered by "
                      "hand)", source="entered by hand (no log)")
        acq.frames_per_field = tb["frames_per_field"]
        log_path = None
    else:
        log_path = log_path or os.path.join(run_dir, params["LOG_FILENAME"])
        acq = read_acquisition_log(
            log_path, tb["frames_per_field"], tb["pulse_separation"],
            tb["camera_fps"], manual=tb["manual"])
    try:
        variables = mat_variables(piv)
    except Exception:
        variables = []
    names = [n for n, _s, _c in variables] or None
    if velocity is None:
        velocity = VALIDATED
    X, Y, U, V, name, reject, note, avail, y_top = load_piv_mat(
        piv, velocity, names)
    grid_max = (float(np.nanmax(X)), float(np.nanmax(Y)))
    rot = int(params["ROTATE"])
    X, Y, U, V = rotate_fields(X, Y, U, V, rot)
    if reject is not None and rot in (90, -90):   # index order follows
        reject = np.ascontiguousarray(np.swapaxes(reject, 0, 1))
    d = PIVData(run_dir, params, X, Y, U, V, name, reject, note, acq,
                available=avail, piv_path=piv, log_path=log_path,
                variables=variables, y_top_px=y_top, grid_max=grid_max,
                calibrate=calibrate, velocity=velocity)
    d.log_skipped = skipped
    # the parameters in memory (no parameter file any more): the
    # calibration record, the masks, the mask applied and the selection
    # areas -- set by the calibration / ROI / mask files of Load data and
    # by the steps, carried over when the run is loaded again. Nothing is
    # taken from run_information.json (a fresh Load starts empty).
    ci = params.get("CALIB_INFO") or {}
    d.calib_info = {k: ci.get(k) for k in CALIB_KEYS}
    d.mask_record = {"areas": list(params.get("MASK_SELECTION") or [])}
    pm = params.get("MASKS") or {}
    for name, m in (pm.items() if isinstance(pm, dict) else []):
        if not isinstance(m, dict):
            continue
        try:
            d.store_mask(str(name), m.get("areas") or [], m.get("colour"))
        except Exception:
            continue
    applied = params.get("MASK_APPLIED")
    if applied in d.masks:
        d.apply_mask(applied)
    return d


# --------------------------------------------------------------------------- #
#  Variables in memory (View > Show all variables)
# --------------------------------------------------------------------------- #
VARIABLE_HELP = {
    # ---- File > Load data
    "run_name": "Name of the run folder loaded in File > Load data (the "
                "experiment, e.g. frot0.50Hz_flib0.400Hz_...). Text.",
    "run_folder": "Full path of the run folder: the folder that holds the "
                  "PIVlab result (.mat), the acquisition log and the "
                  "PostProcessing folder where results are saved. Text.",
    "pivlab_result_file": "Full path of the PIVlab result file (.mat) read "
                          "at Load: the source of the grid and of the "
                          "velocity fields. Text.",
    "acquisition_log_file": "Full path of the acquisition log of the run "
                            "(camera frame rate, pulse separation, ...), "
                            "used to build the time base; empty when there "
                            "is none. Text.",
    "setup_files_loaded": "The files ticked in File > Load data and "
                          "loaded with the run: the calibration file "
                          "(.pivcalib.json), the ROI file (.pivroi.json) "
                          "and the mask file (.pivmask.json); None for a "
                          "box not ticked.",
    "parameters": "All the parameters of the loaded run, kept in memory (no "
                  "parameter file): XSCALE / YSCALE in m/px, ROTATE, "
                  "PTS_ROI and LINES (m), MASKS, MASK_APPLIED, "
                  "MASK_SELECTION (px), CALIB_INFO, file names. They come "
                  "from the files of Load data and the steps (Apply).",
    "velocity_type_loaded": "The velocity loaded and used by every step "
                            "(Load data > velocity): Validated, "
                            "Unvalidated or Smoothed -- the only one kept "
                            "in memory. Text.",
    "velocity_types_in_file": "The velocity types the PIVlab .mat holds "
                              "(the choices of Load data > velocity). "
                              "List of text.",
    "x_grid_px": "Horizontal position of each velocity vector, in pixels "
                 "of the camera image (ny rows x nx columns; constant along "
                 "the columns). x in m = x_grid_px x XSCALE.",
    "z_grid_px": "Vertical position of each velocity vector, in pixels of "
                 "the camera image (ny x nx; constant along the rows). z in "
                 "m = z_grid_px x YSCALE.",
    "u_velocity_px_per_frame": "Horizontal (x) velocity of every vector at "
                               "every PIV field, uncalibrated, in pixels "
                               "per frame (ny x nx x nt, float32; NaN = no "
                               "valid vector or masked). u in m/s = "
                               "u_velocity_px_per_frame x XSCALE / pulse "
                               "separation.",
    "v_velocity_px_per_frame": "Vertical (z) velocity of every vector at "
                               "every PIV field, uncalibrated, in pixels "
                               "per frame (ny x nx x nt, float32; NaN = no "
                               "valid vector or masked). v in m/s = "
                               "v_velocity_px_per_frame x YSCALE / pulse "
                               "separation.",
    "rejected_vectors": "True where PIVlab rejected the vector during its "
                        "validation (typevector_filt != 1), ny x nx x nt "
                        "(bool). Only for the Unvalidated velocity; None "
                        "otherwise (rejected vectors are then NaN).",
    "number_of_fields": "Number of PIV velocity fields (time steps) of the "
                        "run: the length nt of the time axis. Integer.",
    "field_sources": "For each velocity field kept in memory, the variable "
                     "of the .mat it was read from (e.g. u_filtered / "
                     "v_filtered).",
    "valid_vector_fraction": "Fraction (0 to 1) of the vectors of the "
                             "velocity loaded that are valid (finite), the "
                             "mask applied included. Number.",
    "mat_file_variables": "List of the variables stored in the PIVlab .mat "
                          "file: name, size and class of each.",
    "acquisition_log_last_row": "Last data row of the acquisition log: one "
                                "entry per column (camera fps, pulse "
                                "separation, ...); empty without a log.",
    # ---- Pre-processing > Calibration
    "calibration_scales": "The calibration in use: whether it is available "
                          "(XSCALE / YSCALE set), whether the data are "
                          "shown calibrated, the scales in m per pixel, "
                          "the factors px/frame -> m/s for u and v, and "
                          "the units shown.",
    "calibration_segment": "Pre-processing > Calibration: the calibration "
                           "image, the segment drawn on it (end points in "
                           "px), its length in px and in mm typed, the "
                           "scale it gives (m/px) and the scale applied.",
    "time_base": "The time base in use (Calibration > Time base, kept in "
                 "memory): frames per PIV field, pulse separation (s), "
                 "camera frame rate (Hz), PIV field rate (Hz), time scale "
                 "TSCALE (s per field), whether it was typed by hand, and "
                 "where the values come from (log, manual, memory).",
    # ---- Pre-processing > Define ROI
    "roi_velocity_windows": "One entry per ROI defined (field "
                            "'<velocity>_ROIxx'): the ROI name, its number "
                            "xx, the rows / columns of the grid inside it "
                            "(window i0:i1, j0:j1 of the velocity arrays, "
                            "no copy), its rectangle in m and its shape "
                            "(rows x columns).",
    "roi_derived_quantities": "Define ROI > Apply, active ROIs only: "
                              "'<quantity>_ROIxx' -- U, V, velocity norm, "
                              "divergence, vorticity and kinetic energy "
                              "computed from the velocity of ROI xx alone, "
                              "for every frame (ny x nx x nt of the ROI, "
                              "float32, calibrated units m/s, 1/s, m2/s2, "
                              "mask applied).",
    "line_derived_quantities": "Define ROI > Apply, ticked lines only: "
                               "'<quantity>_<LINE>_<region>' -- U, V, "
                               "velocity norm, divergence, vorticity and "
                               "kinetic energy along each line, in the full "
                               "field (Full) and in each active ROI it "
                               "crosses (the ROI's own quantities, at the "
                               "points of the line inside the ROI): values "
                               "(npts x nt, every frame), the distance s "
                               "along the line and the points x, z (m). "
                               "Processing > Profiles uses them.",
    # ---- Pre-processing > Mask
    "mask_applied": "The mask applied to every step: ny x nx boolean array, "
                    "True = masked (the velocity there is NaN at every "
                    "field). None = no mask applied.",
    "mask_applied_name": "Name of the mask applied (chosen in Maps and "
                         "Quiver > Mask), or None. Text.",
    "mask_applied_areas": "The areas (kind and points in px) the mask "
                          "applied is built from.",
    "masks_defined": "Every mask built in Pre-processing > Mask: for each, "
                     "its areas (kind, points), its shading colour on the "
                     "figures and its boolean array (ny x nx, True = "
                     "masked).",
}

# descriptions of the entries inside the variables (by entry name)
ENTRY_HELP = {
    "f": "frequencies of the spectrum (Hz)",
    "amp": "amplitude of the spectrum at each frequency (unit of the "
           "quantity)",
    "psd": "power spectral density at each frequency (unit squared per Hz)",
    "psd_unit": "unit of the power spectral density",
    "phase": "phase of the FFT at each frequency (rad, -pi .. pi)",
    "real": "real part of the FFT at each frequency",
    "imag": "imaginary part of the FFT at each frequency",
    "unit": "unit of the values",
    "units": "units of the coordinates and values",
    "name": "the quantity analysed",
    "desc": "definition of the quantity",
    "roi": "the ROI (or Full) the values come from",
    "n_points": "number of grid points used",
    "dt": "time step between two PIV fields (s)",
    "function": "spectral function: fft, welch (power spectrum) or "
                "periodogram",
    "remove_mean": "True if the mean was removed before the analysis",
    "detrend": "True if the linear trend was removed before the analysis",
    "average": "results of 'average over the ROI first then spectral "
               "analysis'",
    "fft": "results of 'spectral analysis at each point first then average "
           "over the ROI'",
    "peaks": "the dominant peaks found (Find peaks)",
    "rank": "rank of the peak (1 = highest)",
    "period": "period (s) = 1 / frequency",
    "value": "amplitude (or density) at the peak",
    "windows": "frequency ranges (W1, W2, ...)",
    "fmin": "lower frequency of the range (Hz)",
    "fmax": "upper frequency of the range (Hz)",
    "fmin_Hz": "lower frequency of the range (Hz)",
    "fmax_Hz": "upper frequency of the range (Hz)",
    "ticked": "True if the range is the one used",
    "frequencies_used": "frequencies inside the range ticked (Hz)",
    "x": "horizontal positions (m when calibrated)",
    "z": "vertical positions (m when calibrated)",
    "x_m": "horizontal positions of the grid (m)",
    "z_m": "vertical positions of the grid (m)",
    "u": "horizontal velocity",
    "v": "vertical velocity",
    "norm": "velocity norm sqrt(u2 + v2)",
    "u_m_per_s": "horizontal velocity (m/s), ny x nx x nt",
    "v_m_per_s": "vertical velocity (m/s), ny x nx x nt",
    "t_s": "time of each PIV field (s)",
    "mask": "name of the mask applied",
    "mask_applied": "name of the mask applied",
    "run": "name of the run",
    "frame": "frame shown (index, mean or rms)",
    "frame_index": "index of the frame shown",
    "values": "values of the quantity",
    "s": "coordinate along the line (m)",
    "region": "Full (the full field) or the ROI the values come from",
    "length_unit": "unit of s, x and z",
    "ends_m": "end points of the line (m)",
    "lines": "one entry per line",
    "rois": "one entry per ROI",
    "amplitude": "amplitude (or density) at each point and frequency "
                 "(ny x nx x nf)",
    "integrated_amplitude": "amplitude integrated over the range ticked at "
                            "each point (ny x nx)",
    "quantity": "the quantity",
    "window": "rows i0:i1 and columns j0:j1 of the grid inside the ROI",
    "rect": "corners of the ROI ((x1, z1), (x2, z2)) in m",
    "shape": "rows x columns of the ROI grid",
    "number": "number xx of the ROI",
    "areas": "areas of the mask (kind, points in px)",
    "colour": "shading colour on the figures",
    "segment_px": "end points of the segment drawn (px)",
    "length_px": "length of the segment (px)",
    "length_mm": "real length of the segment typed (mm)",
    "applied_m_per_px": "scale applied (m per px)",
    "XSCALE_m_per_px": "horizontal scale (m per px)",
    "YSCALE_m_per_px": "vertical scale (m per px)",
    "pulse_separation_s": "time between the two images of a pair (s)",
    "camera_fps_Hz": "camera frame rate (Hz)",
    "PIV_field_rate_Hz": "number of PIV fields per second (Hz)",
    "TSCALE_s": "time between two PIV fields (s)",
    "frames_per_field": "camera frames per PIV field",
}


def run_variables(d):
    """{name: value} of a loaded run (PIVData), for View > Show all
    variables: the arrays in memory (px, px/frame), not copies."""
    a = d.acq
    return {
        "run_name": d.run, "run_folder": d.run_dir,
        "pivlab_result_file": d.piv_path, "acquisition_log_file": d.log_path,
        "parameters": dict(d.params),
        "velocity_type_loaded": d.default,
        "velocity_types_in_file": list(d.available),
        "x_grid_px": d.X_px, "z_grid_px": d.Z_px,
        "u_velocity_px_per_frame": d.U_px, "v_velocity_px_per_frame": d.V_px,
        "rejected_vectors": d.reject, "number_of_fields": d.nt,
        "calibration_scales": {
            "available": d.can_calibrate, "shown_calibrated": d.calibrated,
            "XSCALE_m_per_px": d.xs if d.can_calibrate else None,
            "YSCALE_m_per_px": d.ys if d.can_calibrate else None,
            "velocity_factor_u": d.xs / a.dt if d.can_calibrate else None,
            "velocity_factor_v": d.ys / a.dt if d.can_calibrate else None,
            "units_shown": {"length": d.lunit, "velocity": d.vunit,
                            "time": d.tunit}},
        "time_base": {"source": a.source, "manual": a.source == "manual",
                      "frames_per_field": a.frames_per_field,
                      "pulse_separation_s": a.dt,
                      "camera_fps_Hz": a.cam_fps,
                      "PIV_field_rate_Hz": a.fps, "TSCALE_s": a.tscale,
                      "log_read": a.log_ok, "message": a.message},
        "acquisition_log_last_row": dict(a.row),
        "mat_file_variables": list(d.variables),
        "field_sources": dict(d.notes), "valid_vector_fraction": d.valid,
        "roi_velocity_windows": {n: dict(i)
                                 for n, i in d.roi_fields().items()},
        "roi_derived_quantities": {n: e["data"] for n, e in sorted(getattr(
            d, "roi_quantities", {}).items())},
        "line_derived_quantities": {
            n: {"values": e["data"], "s": e["s"], "x": e["x"], "z": e["z"],
                "line": e["line"], "region": e["roi"], "unit": e["unit"],
                "length_unit": e["lunit"]}
            for n, e in sorted(getattr(d, "line_quantities", {}).items())},
        "mask_applied": d.mask, "mask_applied_name": d.mask_name,
        "mask_applied_areas": list(d.mask_areas),
        "masks_defined": {n: {"areas": m["areas"], "colour": m["colour"],
                              "mask": m["mask"]}
                          for n, m in d.masks.items()},
    }


# --------------------------------------------------------------------------- #
#  Background image
# --------------------------------------------------------------------------- #
IMAGE_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
BG_MAT_KEYS = ("bg_img_A", "bg_img_B", "bg_img", "background")


def find_background_images(run_dir, root=None):
    """Candidate background images: image files and background*.mat in the
    run folder, then in the root folder -> [path]."""
    out = []
    for folder in (run_dir, root):
        if not folder or not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            low = f.lower()
            p = os.path.join(folder, f)
            if f.startswith(".") or not os.path.isfile(p):
                continue
            if low.endswith(IMAGE_EXT) or (low.endswith(".mat")
                                           and low.startswith("background")):
                if p not in out:
                    out.append(p)
    return out


_IMG_CACHE = {}


def read_image(path):
    """A background image as an array (cached by path and mtime): an image
    file, or a .mat holding bg_img_A / bg_img_B / bg_img / background."""
    key = (path, os.path.getmtime(path))
    if key in _IMG_CACHE:
        return _IMG_CACHE[key]
    if path.lower().endswith(".mat"):
        from scipy.io import loadmat
        mat = loadmat(path)
        k = next((k for k in BG_MAT_KEYS if k in mat), None)
        if k is None:
            raise ValueError("%s holds none of %s" % (os.path.basename(path),
                                                       ", ".join(BG_MAT_KEYS)))
        img = np.asarray(mat[k])
    else:
        import matplotlib.image as mpimg
        img = np.asarray(mpimg.imread(path))
    if img.ndim == 3 and img.shape[2] in (3, 4):
        img = img[..., :3].astype(float).mean(axis=2)   # grey
    _IMG_CACHE.clear()
    _IMG_CACHE[key] = img
    return img


def _rotate_image(img, extent, rotate, x_max, z_max):
    """Rotate an image and its imshow extent like rotate_fields rotates the
    grid (origin='upper'); x_max, z_max are the unrotated grid maxima."""
    left, right, bottom, top = extent
    rotate = int(rotate)
    if rotate == 0:
        return img, extent
    if rotate == 180:
        return np.rot90(img, 2), (x_max - right, x_max - left,
                                  z_max - top, z_max - bottom)
    if rotate == -90:
        return np.rot90(img, 1), (z_max - top, z_max - bottom, left, right)
    if rotate == 90:
        return np.rot90(img, 3), (bottom, top, x_max - right, x_max - left)
    raise ValueError("ROTATE must be 0, 90, 180 or -90, got %r" % rotate)


def place_background(d, img, scale=None):
    """(image, extent) of a full camera frame on the run's axes.

    Pixel p (MATLAB 1-based, what PIVlab writes in x/y) has its centre at
    x = p * scale and z = (y_top - p) * scale, y_top being the raw row of
    the top vector -- the convention of load_piv_mat -- so the picture
    sits exactly under the vectors; then ROTATE is applied. scale: (XSCALE,
    YSCALE) to use (default: the Maps and Quiver display's)."""
    xs, ys = scale or d.pix_scale
    ny, nx = img.shape[:2]
    ytop = float(d.y_top_px)
    extent = (0.5 * xs, (nx + 0.5) * xs,
              (ytop - (ny + 0.5)) * ys, (ytop - 0.5) * ys)
    xm, zm = d.grid_max
    return _rotate_image(img, extent, d.params["ROTATE"], xm * xs, zm * ys)


# --------------------------------------------------------------------------- #
#  Figures (plain matplotlib Figures, embedded by the GUI)
# --------------------------------------------------------------------------- #
def _field(d, frame, name=None, view=None):
    """(u, v, label) of the velocity in a view (default: the Maps and Quiver display,
    full field): a frame index, 'mean' or 'rms'."""
    vw = view or d.display
    if frame in ("mean", "rms"):
        u, v = vw.uv()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if frame == "mean":
                return (np.nanmean(u, axis=2), np.nanmean(v, axis=2),
                        "time mean")
            su = np.sign(np.nanmean(u, axis=2)); su[su == 0] = 1
            sv = np.sign(np.nanmean(v, axis=2)); sv[sv == 0] = 1
            return (np.sqrt(np.nanmean(u ** 2, axis=2)) * su,
                    np.sqrt(np.nanmean(v ** 2, axis=2)) * sv,
                    "rms (sign of the mean)")
    i = int(frame)
    u, v = vw.uv(i)
    return u, v, d.frame_text(i)


def smooth(a, sigma):
    """Gaussian smoothing of a 2-D array (sigma in samples; 0 = none), NaN
    ignored (normalised convolution) and kept NaN (a mask or a rejected
    vector stays visible)."""
    if not sigma or a.ndim != 2:
        return a
    from scipy.ndimage import gaussian_filter
    a = np.asarray(a, float)
    good = np.isfinite(a)
    w = gaussian_filter(good.astype(float), sigma)
    v = gaussian_filter(np.where(good, a, 0.0), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = v / w
    out[(w < 0.05) | ~good] = np.nan
    return out


def _roi_rect(d, roi):
    """((xmin, xmax), (zmin, zmax)) of a named ROI of the run, or None."""
    for name, ((x1, z1), (x2, z2)) in d.rois():
        if name == roi:
            return tuple(sorted((x1, x2))), tuple(sorted((z1, z2)))
    return None


def new_figure(figsize=(8, 6)):
    """A matplotlib Figure with constrained layout, so it re-fits itself
    whenever the GUI resizes it."""
    from matplotlib.figure import Figure
    try:
        return Figure(figsize=figsize, layout="constrained")
    except TypeError:                            # matplotlib < 3.6
        return Figure(figsize=figsize, constrained_layout=True)


QUIVER_COLOURS = ["|u|", "black", "white", "red", "blue", "yellow",
                  "magenta"]


# the quantities a map can show (Processing > Maps and Quiver > Background)
QUANTITIES = ["velocity norm", "U", "V", "divergence", "vorticity",
              "kinetic energy"]
SIGNED = ("U", "V", "divergence", "vorticity")


def quantity_unit(vw, name):
    """The unit of a map quantity in a view."""
    if name in ("velocity norm", "U", "V"):
        return vw.vunit
    if name in SIGNED:
        return "1/s" if vw.calibrated else "1/frame"
    if name == "kinetic energy":
        return "m$^2$/s$^2$" if vw.calibrated else "px$^2$/frame$^2$"
    return vw.vunit


def quantity(vw, u, v, name):
    """A map quantity from the velocity (u, v) of a view -- (ny, nx) or
    (ny, nx, nt) -- on the view's grid (the ROI field, mask NaN included):
    velocity norm |u|, the components U (along x) and V (along z),
    divergence du/dx + dv/dz, vorticity dv/dx - du/dz,
    kinetic energy (u^2 + v^2) / 2 (per unit mass). Derivatives by centred
    differences (np.gradient) on the grid; NaN propagates to the
    neighbours."""
    if name == "velocity norm":
        return np.hypot(u, v)
    if name == "U":
        return np.array(u, dtype=float, copy=True)
    if name == "V":
        return np.array(v, dtype=float, copy=True)
    if name == "kinetic energy":
        return 0.5 * (u * u + v * v)
    X, Z = vw.x, vw.z
    if X.shape[0] < 2 or X.shape[1] < 2:
        return np.full(u.shape, np.nan, np.float32)
    gx = np.gradient(X, axis=1)
    gz = np.gradient(Z, axis=0)
    if u.ndim == 3:
        gx, gz = gx[:, :, None], gz[:, :, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        ddx = lambda a: np.gradient(a, axis=1) / gx
        ddz = lambda a: np.gradient(a, axis=0) / gz
        if name == "divergence":
            return ddx(u) + ddz(v)
        if name == "vorticity":
            return ddx(v) - ddz(u)
    raise ValueError("unknown quantity %r" % name)


def frame_quantity(d, vw, frame, name):
    """(map, label) of a quantity for a frame index, or its time 'mean' /
    'rms' (of the quantity of each frame)."""
    import warnings
    stored = _stored_roi_quantity(d, vw, name) \
        if name in ROI_QUANTITIES else None
    if frame in ("mean", "rms"):
        if stored is not None:
            q = stored
        else:
            u, v = vw.uv()
            q = quantity(vw, u, v, name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if frame == "mean":
                return np.nanmean(q, axis=2), "time mean"
            return np.sqrt(np.nanmean(q * q, axis=2)), "rms"
    i = int(frame)
    if stored is not None:
        return np.asarray(stored[:, :, i], float), d.frame_text(i)
    u, v = vw.uv(i)
    return quantity(vw, u, v, name), d.frame_text(i)


# --------------------------------------------------------------------------- #
#  ROI quantities (Define ROI > Apply): computed from the velocity extracted
#  in each ROI, all frames, kept in memory as '<quantity>_ROIxx'
# --------------------------------------------------------------------------- #
ROI_QUANTITIES = ["U", "V", "velocity norm", "divergence", "vorticity",
                  "kinetic energy"]


def roi_name(number):
    """The name of an ROI: its place in the list (PTS_ROI), 1 -> 'ROI01' --
    the same xx as its fields '<velocity>_ROIxx' and '<quantity>_ROIxx'."""
    return "ROI%02d" % int(number)


def _rect_key(rect, ndigits=6):
    (x1, z1), (x2, z2) = rect
    return tuple(round(float(v), ndigits) for v in (x1, z1, x2, z2))


def renumber_roi_quantities(d, rects):
    """The ROI list changed (an ROI added or removed: the ROIs are numbered
    by their place): the stored quantities '<quantity>_ROIxx' follow their
    rectangle to its new number (renamed, not calculated again); those of
    a rectangle no longer in the list are deleted. rects: the rectangles
    in their new order. Returns (renamed [(old, new)], deleted)."""
    store = getattr(d, "roi_quantities", {})
    where = {_rect_key(r): k for k, r in enumerate(rects, 1)}
    renamed, deleted, new = [], [], {}
    for name, e in store.items():
        k = where.get(_rect_key(e["rect"]))
        if k is None:
            deleted.append(name)
            continue
        nn = roi_quantity_name(e["quantity"], k)
        if nn != name:
            renamed.append((name, nn))
        e = dict(e, number=k, roi=roi_name(k))
        new[nn] = e
    store.clear()
    store.update(new)
    return renamed, deleted


def roi_quantity_name(q, number):
    """'divergence', 2 -> 'divergence_ROI02'."""
    return "%s_ROI%02d" % (q.replace(" ", "_"), int(number))


def _mask_key(d):
    return (d.mask_name, int(d.mask.sum()) if d.mask is not None else 0)


def _roi_q_valid(d, e, info, calibrated):
    """An entry of d.roi_quantities still that of this ROI (same rectangle
    and window), units and mask."""
    return (e["number"] == info["number"] and tuple(e["window"])
            == tuple(info["window"])
            and _rect_key(e["rect"]) == _rect_key(info["rect"])
            and e["calibrated"] == calibrated and e["mask"] == _mask_key(d)
            and e["data"].shape[2] == d.nt)


def compute_roi_quantities(d, calibrated=True, quantities=ROI_QUANTITIES,
                           progress=None, rois=None):
    """Define ROI > Apply: for each ROI field '<velocity>_ROIxx', the
    velocity components U_ROIxx, V_ROIxx and the quantities computed from
    them (velocity norm, divergence, vorticity, kinetic energy) on the ROI
    alone (its own grid:
    derivatives one-sided at its edges; the mask applied), for every frame,
    stored in d.roi_quantities as '<quantity>_ROIxx' (float32, ny x nx x
    nt, calibrated units when the run can be calibrated). A quantity that
    already exists for that ROI (same rectangle, units and mask) is not
    calculated again. Entries of ROIs no longer in the parameter file (or
    renumbered) are removed. rois: the names of the active ROIs (ticked in
    Define ROI) -- only those are calculated (None: all); the quantities
    already calculated for an inactive ROI are kept. Returns (computed
    names, kept names, removed names)."""
    import warnings
    store = d.__dict__.setdefault("roi_quantities", {})
    cal = bool(calibrated) and d.can_calibrate
    fields = d.roi_fields()
    by_number = {i["number"]: i for i in fields.values()}
    removed = []
    for name in list(store):
        e = store[name]
        info = by_number.get(e["number"])
        if info is None or _rect_key(info["rect"]) != _rect_key(e["rect"]):
            del store[name]                      # ROI removed / renumbered
            removed.append(name)
    computed, kept = [], []
    todo = [(info, q) for info in fields.values() for q in quantities
            if rois is None or info["roi"] in rois]
    for k, (info, q) in enumerate(todo):
        name = roi_quantity_name(q, info["number"])
        e = store.get(name)
        if e is not None and _roi_q_valid(d, e, info, cal):
            kept.append(name)
            continue
        vw = RunView(d, cal, info["window"])
        u, v = vw.uv()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            data = np.asarray(quantity(vw, u, v, q), np.float32)
        store[name] = dict(data=data, quantity=q, number=info["number"],
                           roi=info["roi"], window=tuple(info["window"]),
                           rect=info["rect"], calibrated=cal,
                           unit=quantity_unit(vw, q), mask=_mask_key(d))
        computed.append(name)
        if progress is not None:
            progress(k + 1, len(todo))
    return computed, kept, removed


def drop_roi_quantities(d, roi=None, number=None):
    """Remove the quantities '<quantity>_ROIxx' of an ROI (by its name or
    number); returns the names removed."""
    store = getattr(d, "roi_quantities", {})
    gone = [n for n, e in store.items()
            if (roi is not None and e["roi"] == roi)
            or (number is not None and e["number"] == int(number))]
    for n in gone:
        del store[n]
    return gone


def _stored_roi_quantity(d, vw, name):
    """The stored (ny, nx, nt) array of a quantity for the ROI view vw
    ('<quantity>_ROIxx' of Define ROI > Apply). When it does not exist yet,
    or no longer fits (ROI or mask changed), the ROI quantities are
    calculated once (compute_roi_quantities) and stored, so no step
    calculates them again. None for the full field or an uncalibrated
    view (the stored quantities are in calibrated units)."""
    if vw.window is None or not vw.calibrated:
        return None
    for attempt in (0, 1):
        store = getattr(d, "roi_quantities", {})
        for info in d.roi_fields().values():
            if tuple(info["window"]) != tuple(vw.window):
                continue
            e = store.get(roi_quantity_name(name, info["number"]))
            if e is not None and _roi_q_valid(d, e, info, vw.calibrated):
                return e["data"]
        if attempt == 0:
            compute_roi_quantities(d, calibrated=True)
    return None


def roi_quantity(d, field, name):
    """The stored (ny, nx, nt) array '<quantity>_ROIxx' of an ROI field
    (e.g. 'Validated_ROI02', 'vorticity'), calculated once if needed."""
    info = d.roi_fields().get(field)
    if info is None:
        raise KeyError("no ROI field %s" % field)
    return _stored_roi_quantity(d, RunView(d, True, info["window"]), name)


# map display (every step that draws a map): shading and colour map
# shading of a map when its tick box is ticked (unticked: 'nearest', one
# flat cell per vector): gouraud (pcolormesh) or an image interpolation
MAP_SHADINGS = ["gouraud", "bilinear", "bicubic", "spline16", "lanczos",
                "hanning", "gaussian"]
MAP_CMAPS = ["auto", "viridis", "plasma", "inferno", "magma", "cividis",
             "turbo", "jet", "RdBu_r", "coolwarm", "seismic", "PuOr_r",
             "gray"]


def map_cmap(cmap, signed=False):
    """The colour map of a map: 'auto' -> viridis, or RdBu_r (blue - white
    - red, centred on 0) for a signed quantity."""
    if not cmap or cmap == "auto":
        return "RdBu_r" if signed else "viridis"
    return cmap


def map_shading(shading):
    """'nearest' (one flat cell per vector: no shading), 'gouraud' (colours
    interpolated between the vectors) or an image interpolation of
    matplotlib (bilinear, bicubic, spline16, lanczos, hanning, gaussian)."""
    return shading if shading in MAP_SHADINGS else "nearest"


def draw_map(ax, X, Z, V, shading="nearest", zorder=0, **kw):
    """A map of values V on the grid (X, Z) with a shading (map_shading):
    nearest / gouraud by pcolormesh, the image interpolations by imshow on
    the regular grid (the cells centred on the vectors). NaN shown empty.
    kw: cmap, vmin, vmax, ... Returns the mappable."""
    sh = map_shading(shading)
    V = np.ma.masked_invalid(np.asarray(V, float))
    if sh in ("nearest", "gouraud"):
        return ax.pcolormesh(X, Z, V, shading=sh, zorder=zorder, **kw)
    X, Z = np.asarray(X, float), np.asarray(Z, float)
    gx, gz = X[0, :], Z[:, 0]
    regular = (X.shape == V.shape and gx.size > 1 and gz.size > 1
               and np.allclose(X, gx[None, :]) and np.allclose(Z, gz[:, None]))
    if not regular:                              # not a grid: gouraud
        return ax.pcolormesh(X, Z, V, shading="gouraud", zorder=zorder, **kw)
    ix, iz = np.argsort(gx), np.argsort(gz)
    A = V[np.ix_(iz, ix)]
    gx, gz = gx[ix], gz[iz]
    hx = 0.5 * (gx[-1] - gx[0]) / (gx.size - 1)
    hz = 0.5 * (gz[-1] - gz[0]) / (gz.size - 1)
    im = ax.imshow(A, origin="lower", interpolation=sh, aspect="auto",
                   extent=(gx[0] - hx, gx[-1] + hx, gz[0] - hz, gz[-1] + hz),
                   zorder=zorder, **kw)
    return im


def _rect_shown(d, rect):
    """An ROI rectangle in m -> in the unit of the Maps and Quiver display."""
    (x1, z1), (x2, z2) = rect
    xs, ys = (1.0, 1.0) if d.calibrated else (1.0 / d.xs, 1.0 / d.ys)
    return ((x1 * xs, z1 * ys), (x2 * xs, z2 * ys))


def display_figure(d, frame=0, field=None, roi="Full", quiver=True,
                   quiver_colour="|u|", skip=1, background=None,
                   image=None, cmap="auto", clip=(1.0, 99.5),
                   scale=None, vec_scale=1.0, smoothing=0, u_max=None,
                   bg_max=None, rois_shown=None, lines_shown=None,
                   shading="nearest", fig=None):
    """The Velocity figure of Load data.

    frame       a frame index, 'mean' or 'rms'
    field       kept for compatibility (the velocity loaded is shown)
    roi         'Full', or an ROI field ('<velocity>_ROIxx', see
                PIVData.roi_fields): the velocity extracted in the ROI is
                shown (its own grid, nothing masked)
    quiver      draw the vectors (every `skip`-th one), coloured by |u|
                (quiver_colour '|u|', colour map `cmap`) or in one colour;
                vec_scale: arrow length -- the 99th-percentile |u| is drawn
                vec_scale vector spacings long (a key gives that speed)
    background  None, a quantity of QUANTITIES as a colour map -- 'velocity
                norm' (|u| of the field shown), 'U', 'V' (the velocity
                components), 'divergence', 'vorticity' (these four with a
                diverging map, centred on 0), 'kinetic energy' (of each
                frame, time mean / rms of it) -- computed on the field
                shown (the ROI field when one is chosen) with the mask
                applied; or 'image' (the image file `image`, grey,
                contrast stretched to the `clip` percentiles, placed on
                the PIV grid and rotated by ROTATE)
    smoothing   Gaussian smoothing of the background (sigma; 0 = none):
                in vector spacings for the velocity norm, in image pixels
                for an image
    rois_shown, lines_shown
                {name: colour} of the ROIs / lines drawn (those ticked in
                Define ROI, in their colours there); None: all of them, in
                the default colours (roi_colour)
    One colour scale for |u|: 0 .. 99th percentile over the view (u_max:
    a fixed value instead -- also the arrow length; bg_max: a fixed top of
    the background map; used by the movies)."""
    info = d.roi_fields().get(roi) if roi and roi != "Full" else None
    vw = RunView(d, d.calibrated, info["window"] if info else None)
    u, v, label = _field(d, frame, view=vw)
    X, Z = vw.x, vw.z
    norm = np.hypot(u, v)
    fin = norm[np.isfinite(norm)]
    vmax = (float(np.percentile(fin, 99)) if fin.size else 1.0) or 1.0
    if u_max:
        vmax = float(u_max)
    fig = fig or new_figure()
    fig.clear()
    ax = fig.add_subplot(111)
    clab = r"$|\mathbf{u}|$  (%s)" % vw.vunit
    mappable, notes = None, []
    sm = float(smoothing or 0)
    sh = map_shading(shading)
    cmap_s = map_cmap(cmap, True)                # signed maps
    cmap = map_cmap(cmap, False)                 # |u|, positive maps

    if background == "velocity norm":
        bg = smooth(norm, sm)
        mappable = draw_map(ax, X, Z, bg, sh, cmap=cmap, vmin=0.0,
                            vmax=vmax, zorder=0)
        notes.append("background: |u|" + (", smoothed (sigma %g)" % sm
                                           if sm else ""))
    elif background in QUANTITIES:
        q, _lab = frame_quantity(d, vw, frame, background)
        bg = smooth(np.asarray(q, float), sm)
        fq = np.abs(bg[np.isfinite(bg)])
        top = (float(np.percentile(fq, 99)) if fq.size else 1.0) or 1.0
        if bg_max:
            top = float(bg_max)
        if background in SIGNED:
            kw = dict(cmap=cmap_s, vmin=-top, vmax=top)
        else:
            kw = dict(cmap=cmap, vmin=0.0, vmax=top)
        mappable = draw_map(ax, X, Z, bg, sh, zorder=0, **kw)
        clab = "%s  (%s)" % (background, quantity_unit(vw, background))
        notes.append("background: %s%s" % (
            background, ", smoothed (sigma %g)" % sm if sm else ""))
    elif background == "image":
        if not image or not os.path.isfile(image):
            notes.append("(no background image chosen)")
        else:
            img, extent = place_background(d, read_image(image))
            if sm and img.ndim == 2:
                img = smooth(img, sm)
            vlo = vhi = None
            if clip and img.ndim == 2:
                good = img[np.isfinite(img)]
                if good.size:
                    vlo, vhi = np.percentile(good, clip)
            ax.imshow(img, cmap="gray", origin="upper", extent=extent,
                      vmin=vlo, vmax=vhi, zorder=0,
                      interpolation=sh if sh not in ("nearest", "gouraud")
                      else "nearest")
            notes.append("background: %s%s" % (
                os.path.basename(image),
                ", smoothed (sigma %g px)" % sm if sm else ""))

    if quiver:
        sk = max(1, int(skip))
        Xq, Zq = X[::sk, ::sk], Z[::sk, ::sk]
        U, V = u[::sk, ::sk], v[::sk, ::sk]
        N = np.hypot(U, V)
        bad = ~np.isfinite(N)
        U, V, N = (np.ma.masked_array(A, bad) for A in (U, V, N))
        sp = []
        if X.shape[1] > 1:
            sp.append(np.nanmedian(np.abs(np.diff(X, axis=1))))
        if X.shape[0] > 1:
            sp.append(np.nanmedian(np.abs(np.diff(Z, axis=0))))
        spacing = (float(np.nanmin(sp)) if sp else 1.0) * sk
        if scale is None:                        # vmax -> vec_scale spacings
            scale = vmax / (max(float(vec_scale), 1e-6) * spacing)
        kw = dict(pivot="mid", scale=scale, scale_units="xy", width=0.0025,
                  angles="xy", zorder=2)
        if quiver_colour == "|u|":
            q = ax.quiver(Xq, Zq, U, V, N, cmap=cmap, clim=(0.0, vmax), **kw)
            if mappable is None:
                mappable = q
        else:
            q = ax.quiver(Xq, Zq, U, V, color=quiver_colour, **kw)
        ax.quiverkey(q, 0.02, 1.015, vmax, "%.3g %s" % (vmax, vw.vunit),
                     labelpos="E", coordinates="axes", fontproperties={
                         "size": 8})
        notes.append("%s%s%s" % (d.default if info is None else roi,
                                 "" if sk == 1 else ", every %d vector" % sk,
                                 "" if float(vec_scale) == 1.0 else
                                 ", vector scale x%g" % float(vec_scale)))
    if mappable is not None:
        fig.colorbar(mappable, ax=ax, label=clab)

    # ROIs (those shown, in their colours): all of them on the full view;
    # the one shown otherwise
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=2.5, foreground="0.15", alpha=0.6)]
    for name, i in d.roi_fields().items():
        if info is not None and name != roi:
            continue
        if rois_shown is not None and i["roi"] not in rois_shown:
            continue
        c = (rois_shown or {}).get(i["roi"]) or \
            roi_colour("rect", i["number"] - 1)
        (xa, za), (xb, zb) = _rect_shown(d, i["rect"])
        rx, rz = sorted((xa, xb)), sorted((za, zb))
        ax.plot([rx[0], rx[1], rx[1], rx[0], rx[0]],
                [rz[0], rz[0], rz[1], rz[1], rz[0]], "--", color=c,
                lw=1.6, zorder=3)
        if info is None:
            ax.text(rx[0], rz[1], " %s" % i["roi"],
                    color=c, fontsize=8, va="bottom", zorder=3,
                    path_effects=halo)
    if info is None:                             # lines on the full view
        for k, (name, rect) in enumerate(
                parse_shapes(d.params.get("LINES"), "LINE")
                if d.can_calibrate else []):
            if lines_shown is not None and name not in lines_shown:
                continue
            c = (lines_shown or {}).get(name) or roi_colour("line", k)
            (xa, za), (xb, zb) = _rect_shown(d, rect)
            ax.plot([xa, xb], [za, zb], "-", color=c, lw=1.8, zorder=3)
            ax.text(xb, zb, " " + name, color=c, fontsize=8, zorder=3,
                    path_effects=halo)
    x0, x1 = np.nanmin(X), np.nanmax(X)
    z0, z1 = np.nanmin(Z), np.nanmax(Z)
    px, pz = 0.02 * (x1 - x0) or 1, 0.02 * (z1 - z0) or 1
    ax.set_xlim(x0 - px, x1 + px)
    ax.set_ylim(z0 - pz, z1 + pz)
    ax.set_aspect("equal")
    ax.set_xlabel("x (%s)" % vw.lunit)
    ax.set_ylabel("z (%s)" % vw.lunit)
    ax.set_title("%s%s\n%s\n%s" % (
        d.run, "" if info is None else "   [%s: ROI %s, %d x %d vectors]"
        % (roi, info["roi"], info["shape"][0], info["shape"][1]),
        label, "\n".join(notes)), fontsize=9)
    return fig


# --------------------------------------------------------------------------- #
#  Profiles (Processing > Profiles): a quantity along the lines (LINES)
# --------------------------------------------------------------------------- #
def _line_in_view(vw, rect):
    """A line given in m -> its ends in the unit of a view (m or px)."""
    (x1, z1), (x2, z2) = rect
    if vw.calibrated:
        return (x1, z1), (x2, z2)
    r = vw.run
    return (x1 / r.xs, z1 / r.ys), (x2 / r.xs, z2 / r.ys)


def sample_line(vw, field, rect, n=None):
    """(s, values, x, z): a map `field` (ny, nx) of a view sampled along a
    line (ends in m), by bilinear interpolation on the PIV grid -- NaN next
    to a NaN (mask, rejected vector) and outside the grid. s: distance from
    the first end, in the view's unit. n points (default: about one per
    vector spacing, at least 2)."""
    from scipy.interpolate import RegularGridInterpolator
    (xa, za), (xb, zb) = _line_in_view(vw, rect)
    X, Z = vw.x, vw.z
    gx, gz = X[0, :].astype(float), Z[:, 0].astype(float)
    F = np.asarray(field, float)
    ix, iz = np.argsort(gx), np.argsort(gz)        # increasing axes
    gx, gz, F = gx[ix], gz[iz], F[np.ix_(iz, ix)]
    length = float(np.hypot(xb - xa, zb - za))
    if n is None:
        sp = [np.median(np.diff(g)) for g in (gx, gz) if g.size > 1]
        sp = min(sp) if sp else 1.0
        n = int(np.ceil(length / sp)) + 1 if sp > 0 else 2
    n = max(int(n), 2)
    t = np.linspace(0.0, 1.0, n)
    x = xa + t * (xb - xa)
    z = za + t * (zb - za)
    if gx.size < 2 or gz.size < 2:
        return t * length, np.full(n, np.nan), x, z
    f = RegularGridInterpolator((gz, gx), F, method="linear",
                                bounds_error=False, fill_value=np.nan)
    return t * length, f(np.column_stack([z, x])), x, z


def _roi_view(d, calibrated, roi):
    """(view, ROI field info or None): the full field, or an ROI field
    ('<velocity>_ROIxx' of d.roi_fields())."""
    info = d.roi_fields().get(roi) if roi not in (None, "", "Full") \
        else None
    return RunView(d, calibrated, info["window"] if info else None), info


def _inside_view(vw, x, z):
    """Points (x, z) inside the grid of a view (an ROI's vectors)."""
    gx, gz = vw.x[0, :], vw.z[:, 0]
    ex = 1e-9 * max(float(np.ptp(gx)), 1.0)
    ez = 1e-9 * max(float(np.ptp(gz)), 1.0)
    return ((x >= gx.min() - ex) & (x <= gx.max() + ex)
            & (z >= gz.min() - ez) & (z <= gz.max() + ez))


def line_in_roi(d, rect, roi):
    """True when a line (in m) has points inside an ROI field (or roi is
    Full / None)."""
    vw, info = _roi_view(d, True, roi)
    if info is None:
        return True
    full = RunView(d, True, None)
    _s, _v, x, z = sample_line(full, np.zeros(full.x.shape), rect)
    return bool(_inside_view(vw, x, z).any())


def _sample_points(vw, F, x, z):
    """A map (ny, nx) or maps (ny, nx, nt) of a view interpolated
    (bilinear, NaN next to a NaN and outside the grid) at points (x, z)
    -> (npts,) or (npts, nt)."""
    from scipy.interpolate import RegularGridInterpolator
    gx, gz = vw.x[0, :].astype(float), vw.z[:, 0].astype(float)
    F = np.asarray(F, float)
    ix, iz = np.argsort(gx), np.argsort(gz)
    gx, gz, F = gx[ix], gz[iz], F[iz][:, ix]
    if gx.size < 2 or gz.size < 2:
        return np.full((len(x),) + F.shape[2:], np.nan)
    f = RegularGridInterpolator((gz, gx), F, method="linear",
                                bounds_error=False, fill_value=np.nan)
    return f(np.column_stack([z, x]))


# Define ROI > Apply: the quantities along each ticked line, in the full
# field and in each active ROI it crosses, every frame (Profiles uses them)
def line_quantity_name(q, line, roi):
    return "%s_%s_%s" % (q.replace(" ", "_"), line, roi or "Full")


def _line_entry_valid(d, e, rect, roi_rect, cal):
    return (e["rect"] == _rect_key(rect) and e["calibrated"] == cal
            and e["roi_rect"] == (None if roi_rect is None
                                  else _rect_key(roi_rect))
            and e["mask"] == _mask_key(d) and e["nt"] == d.nt)


def compute_line_quantities(d, lines, rois=None, calibrated=True,
                            quantities=ROI_QUANTITIES, progress=None,
                            prune=True):
    """For each line {name: ((x1, z1), (x2, z2)) in m} and each region --
    the full field and every ROI field (rois: the names of the active ROIs;
    None: all) that the line crosses --, the quantities (U, V, velocity
    norm, divergence, vorticity, kinetic energy) sampled along the line at
    every frame: d.line_quantities['<quantity>_<LINE>_<ROIxx or Full>'] =
    dict(data (npts, nt) float32, s, x, z, quantity, line, roi, unit,
    lunit, ...). In an ROI the quantity is the stored '<quantity>_ROIxx'
    (computed on the ROI alone) at the points of the line inside the ROI;
    in the full field it is computed on the full grid. Entries that exist
    and still fit (same line, ROI, mask, units) are kept; entries of lines
    or ROIs no longer defined are removed (prune; not when one line is
    calculated on its own). Returns (computed, kept, removed) names."""
    import warnings
    store = d.__dict__.setdefault("line_quantities", {})
    cal = bool(calibrated) and d.can_calibrate
    fields = {i["roi"]: i for i in d.roi_fields().values()} if cal else {}
    removed = []
    for name in (list(store) if prune else []):
        e = store[name]
        if e["line"] not in lines or (e["roi"] != "Full"
                                      and e["roi"] not in fields):
            del store[name]
            removed.append(name)
    regions = ["Full"] + [r for r in fields if rois is None or r in rois]
    computed, kept = [], []
    full = RunView(d, cal, None)
    uv_full = None
    todo = [(ln, rect, reg) for ln, rect in lines.items() for reg in regions]
    for k, (ln, rect, reg) in enumerate(todo):
        info = fields.get(reg)
        roi_rect = info["rect"] if info else None
        names = [line_quantity_name(q, ln, reg) for q in quantities]
        if all(n in store and _line_entry_valid(d, store[n], rect,
                                                roi_rect, cal)
               for n in names):
            kept += names
            continue
        s, _v, x, z = sample_line(full, np.zeros(full.x.shape), rect)
        vw = RunView(d, cal, info["window"]) if info else full
        if info is not None:
            keep = _inside_view(vw, x, z)
            if not keep.any():
                for n in names:                  # does not cross the ROI
                    store.pop(n, None)
                continue
            s, x, z = s[keep], x[keep], z[keep]
        for q, n in zip(quantities, names):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if info is not None and cal:
                    F = _stored_roi_quantity(d, vw, q)
                else:
                    F = None
                if F is None:
                    if info is None:
                        if uv_full is None:
                            uv_full = full.uv()
                        u, v = uv_full
                    else:
                        u, v = vw.uv()
                    F = quantity(vw, u, v, q)
                data = _sample_points(vw, F, x, z)
            store[n] = dict(data=np.asarray(data, np.float32), s=s, x=x, z=z,
                            quantity=q, line=ln, roi=reg,
                            rect=_rect_key(rect),
                            roi_rect=None if roi_rect is None
                            else _rect_key(roi_rect),
                            calibrated=cal, unit=quantity_unit(vw, q),
                            lunit=full.lunit, mask=_mask_key(d), nt=d.nt)
            computed.append(n)
        if progress is not None:
            progress(k + 1, len(todo))
    return computed, kept, removed


def _stored_line(d, rect, q, roi, cal):
    """The stored entry of a quantity along a line (by its coordinates) in
    a region (None / 'Full' or an ROI name), or None."""
    reg = roi or "Full"
    if reg != "Full":
        info = d.roi_fields().get(roi)
        if info is None:
            return None
        reg, roi_rect = info["roi"], info["rect"]
    else:
        roi_rect = None
    for e in getattr(d, "line_quantities", {}).values():
        if e["quantity"] == q and e["roi"] == reg and \
                _line_entry_valid(d, e, rect, roi_rect, cal):
            return e
    return None


def line_profile(d, rect, quantity_name, frame=0, calibrated=True,
                 _maps=None, roi=None, line=None):
    """dict(s, values, x, z, label, unit, lunit, inside) of a quantity
    (QUANTITIES) along a line of LINES (in m) for a frame, or its time
    'mean' / 'rms', on the full field (the mask applied: NaN in it) -- or,
    with an ROI field `roi`, the quantity of that ROI (Define ROI > Apply:
    '<quantity>_ROIxx', computed on the ROI alone) at the points of the
    line inside the ROI only (s: still the distance from the line's first
    end). inside: the number of points kept (0: the line does not cross
    the ROI). The quantities along the lines computed by Define ROI > Apply
    (d.line_quantities) are used when they exist; for a named line
    (`line`) that has none yet, they are calculated once and stored."""
    vw, info = _roi_view(d, calibrated, roi)
    if quantity_name in ROI_QUANTITIES:
        e = _stored_line(d, rect, quantity_name, roi, vw.calibrated)
        if e is None and line is not None:
            compute_line_quantities(d, {line: rect},
                                    rois=[info["roi"]] if info else [],
                                    calibrated=vw.calibrated, prune=False)
            e = _stored_line(d, rect, quantity_name, roi, vw.calibrated)
        if e is not None:
            import warnings
            data = np.asarray(e["data"], float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if frame == "mean":
                    val, label = np.nanmean(data, axis=1), "time mean"
                elif frame == "rms":
                    val, label = np.sqrt(np.nanmean(data * data, axis=1)), \
                        "rms"
                else:
                    val, label = data[:, int(frame)], d.frame_text(int(frame))
            return dict(s=e["s"], values=val, x=e["x"], z=e["z"],
                        label=label, lunit=vw.lunit,
                        unit=quantity_unit(vw, quantity_name),
                        inside=len(e["s"]))
        if info is not None and line is not None:   # does not cross it
            return dict(s=np.array([]), values=np.array([]),
                        x=np.array([]), z=np.array([]),
                        label=frame if isinstance(frame, str)
                        else d.frame_text(int(frame)), lunit=vw.lunit,
                        unit=quantity_unit(vw, quantity_name), inside=0)
    key = (quantity_name, str(frame), roi)
    if _maps is not None and key in _maps:
        q, label = _maps[key]
    else:
        q, label = frame_quantity(d, vw, frame, quantity_name)
        if _maps is not None:
            _maps[key] = (q, label)
    if info is None:
        s, val, x, z = sample_line(vw, q, rect)
        inside = len(s)
    else:                                        # spacing of the full grid
        full = RunView(d, vw.calibrated, None)
        s, _v, x, z = sample_line(full, np.zeros(full.x.shape), rect)
        n = len(s)
        _s, val, _x, _z = sample_line(vw, q, rect, n=n)
        keep = _inside_view(vw, x, z)
        s, val, x, z = s[keep], val[keep], x[keep], z[keep]
        inside = int(keep.sum())
    return dict(s=s, values=val, x=x, z=z, label=label, lunit=vw.lunit,
                unit=quantity_unit(vw, quantity_name), inside=inside)


def profile_figure(d, frame=0, lines=(), quantity="velocity norm",
                   calibrated=True, ylim=None, roi=None, fig=None):
    """The figure of Processing > Profiles: the quantity `quantity` along
    each line of `lines` [(name, ((x1, z1), (x2, z2)) in m, colour)] --
    x: the distance along the line from its first end, y: the quantity --
    for a frame, or its time 'mean' / 'rms'; mask applied. ylim: fixed
    y limits (the movies)."""
    fig = fig or new_figure()
    fig.clear()
    ax = fig.add_subplot(111)
    vw = RunView(d, calibrated, None)
    label, maps = (d.frame_text(int(frame)) if not isinstance(frame, str)
                   else frame), {}
    rinfo = _roi_view(d, calibrated, roi)[1]
    for name, rect, colour in lines:
        p = line_profile(d, rect, quantity, frame, vw.calibrated, maps, roi,
                         line=name)
        label = p["label"]
        ax.plot(p["s"], p["values"], "-", color=colour, lw=1.8, marker=".",
                ms=4, label=name if p["inside"] else
                "%s (not in the ROI)" % name)
    if not lines:
        ax.text(0.5, 0.5, "No line: draw lines in Pre-processing > Define "
                "ROI (and tick them)", ha="center", va="center",
                transform=ax.transAxes, color="0.4")
    if quantity in SIGNED:
        ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
    if lines:
        leg = ax.legend(loc="best", fontsize=8, framealpha=0.85)
        for t in leg.get_texts():            # white / light names readable
            t.set_color("black")
    ax.grid(True, color="0.9", lw=0.6)
    ax.set_xlabel("distance along the line (%s)" % vw.lunit)
    ax.set_ylabel("%s  (%s)" % (quantity, quantity_unit(vw, quantity)))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title("%s\n%s\n%s%s%s" % (
        d.run, label, quantity,
        ", in %s (computed on the ROI)" % rinfo["roi"]
        if rinfo else "", ", mask %s" % d.mask_name
        if d.mask_name else ""), fontsize=9)
    return fig


def profile_data(d, frame=0, lines=(), quantity="velocity norm",
                 calibrated=True, roi=None):
    """The data of the Profiles figure, for Save data: per line its ends,
    the points (x, z), the distance s and the values."""
    vw = RunView(d, calibrated, None)
    out = {"run": d.run, "quantity": quantity,
           "units": {"length": vw.lunit,
                     "quantity": quantity_unit(vw, quantity)},
           "mask": d.mask_name or "", "lines": {}}
    maps = {}
    out["roi"] = roi or "Full"
    for name, rect, _c in lines:
        p = line_profile(d, rect, quantity, frame, vw.calibrated, maps, roi,
                         line=name)
        out["frame"] = p["label"]
        out["lines"][name] = {"ends_m": np.asarray(rect, float),
                              "s": p["s"], "x": p["x"], "z": p["z"],
                              "values": np.asarray(p["values"], float)}
    if not isinstance(frame, str):
        out["frame_index"] = int(frame)
        if vw.calibrated:
            out["t_s"] = float(d.t[int(frame)])
    return out


def profile_limits(d, opts, frames, samples=60):
    """Fixed y limits of a profile movie: the 0.5 and 99.5 percentiles of
    the quantity along the lines over up to `samples` frames of the range
    (with a margin)."""
    idx = np.unique(np.linspace(frames[0], frames[-1],
                                min(samples, len(frames))).astype(int))
    vals = []
    for i in idx:
        maps = {}
        for _n, rect, _c in opts.get("lines") or ():
            p = line_profile(d, rect, opts.get("quantity", "velocity norm"),
                             int(i), opts.get("calibrated", True), maps,
                             opts.get("roi"), line=_n)
            v = np.asarray(p["values"], float)
            vals.append(v[np.isfinite(v)])
    v = np.concatenate(vals) if vals and sum(a.size for a in vals) else \
        np.array([0.0, 1.0])
    lo, hi = np.percentile(v, (0.5, 99.5))
    if opts.get("quantity") in SIGNED:
        hi = max(abs(lo), abs(hi))
        lo = -hi
    if not hi > lo:
        lo, hi = lo - 1.0, hi + 1.0
    pad = 0.05 * (hi - lo)
    return float(lo - pad), float(hi + pad)


# --------------------------------------------------------------------------- #
#  Time series (Processing > Time series): statistics over an ROI, per frame
# --------------------------------------------------------------------------- #
# name -> (ROI quantity, statistic, absolute value first, description)
TIME_SERIES = [
    ("<U>", "U", "mean", False, "mean over the ROI of U_ROIxx"),
    ("U_std", "U", "std", False,
     "standard deviation over the ROI of U_ROIxx"),
    ("<|U|>", "U", "mean", True, "mean over the ROI of |U_ROIxx|"),
    ("|U|_std", "U", "std", True,
     "standard deviation over the ROI of |U_ROIxx|"),
    ("<V>", "V", "mean", False, "mean over the ROI of V_ROIxx"),
    ("V_std", "V", "std", False,
     "standard deviation over the ROI of V_ROIxx"),
    ("<|V|>", "V", "mean", True, "mean over the ROI of |V_ROIxx|"),
    ("|V|_std", "V", "std", True,
     "standard deviation over the ROI of |V_ROIxx|"),
    ("<Vorticity>", "vorticity", "mean", False,
     "mean over the ROI of vorticity_ROIxx"),
    ("Vorticity_std", "vorticity", "std", False,
     "standard deviation over the ROI of vorticity_ROIxx"),
    ("<|Vorticity|>", "vorticity", "mean", True,
     "mean over the ROI of |vorticity_ROIxx|"),
    ("|Vorticity|_std", "vorticity", "std", True,
     "standard deviation over the ROI of |vorticity_ROIxx|"),
    ("<Kinetic energy>", "kinetic energy", "mean", False,
     "mean over the ROI of kinetic_energy_ROIxx"),
    ("Kinetic Energy_std", "kinetic energy", "std", False,
     "standard deviation over the ROI of kinetic_energy_ROIxx"),
]
TIME_SERIES_NAMES = [t[0] for t in TIME_SERIES]


def roi_time_series(d, field, name):
    """dict(t, values, n, unit, quantity): the time series `name`
    (TIME_SERIES) of an ROI field: for every frame, the mean or standard
    deviation over the ROI (the points that are not NaN: mask, rejected
    vectors) of the stored ROI quantity '<quantity>_ROIxx' (Define ROI >
    Apply; calculated once if it does not exist yet), or of its absolute
    value. Calibrated units (m/s, 1/s, m^2/s^2), t in s."""
    import warnings
    spec = {t[0]: t for t in TIME_SERIES}[name]
    _n, q, stat, absolute, _desc = spec
    data = roi_quantity(d, field, q)
    a = np.abs(data) if absolute else data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        val = (np.nanmean(a, axis=(0, 1)) if stat == "mean"
               else np.nanstd(a, axis=(0, 1)))
    n = np.sum(np.isfinite(a), axis=(0, 1))
    vw = RunView(d, True, d.roi_fields()[field]["window"])
    return dict(t=np.asarray(vw.t, float), values=np.asarray(val, float),
                n=np.asarray(n, int), quantity=q,
                unit=quantity_unit(vw, q).replace("$^2$", "²"))


def time_series_figure(d, rois=(), name="<U>", fig=None):
    """The figure of Processing > Time series: the time series `name` of
    each ROI of `rois` [(field, ROI name, colour)], against time (s)."""
    fig = fig or new_figure()
    fig.clear()
    ax = fig.add_subplot(111)
    unit = ""
    for field, roi, colour in rois:
        ts = roi_time_series(d, field, name)
        unit = ts["unit"]
        info = d.roi_fields()[field]
        ax.plot(ts["t"], ts["values"], "-", color=colour, lw=1.4,
                label=info["roi"])
    if not rois:
        ax.text(0.5, 0.5, "No ROI: tick ROIs in the table (Define ROI > "
                "Apply makes their fields)", ha="center", va="center",
                transform=ax.transAxes, color="0.4")
    else:
        leg = ax.legend(loc="best", fontsize=8, framealpha=0.85)
        for t in leg.get_texts():
            t.set_color("black")
    spec = {t[0]: t for t in TIME_SERIES}[name]
    if spec[1] in SIGNED and spec[2] == "mean" and not spec[3]:
        ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
    ax.grid(True, color="0.9", lw=0.6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("%s  (%s)" % (name, unit) if unit else name)
    ax.set_title("%s\n%s\n%s%s" % (
        d.run, name, spec[4], ", mask %s" % d.mask_name
        if d.mask_name else ""), fontsize=9)
    return fig


def time_series_data(d, rois=(), name="<U>"):
    """Save data of the Time series figure: per ROI t, the values and the
    number of points used per frame."""
    out = {"run": d.run, "time_series": name,
           "definition": {t[0]: t for t in TIME_SERIES}[name][4],
           "mask": d.mask_name or "", "rois": {}}
    for field, roi, _c in rois:
        ts = roi_time_series(d, field, name)
        out["units"] = {"time": "s", "value": ts["unit"]}
        out["rois"][field] = {"roi": roi, "t_s": ts["t"],
                              "values": ts["values"], "n_points": ts["n"]}
    return out


# --------------------------------------------------------------------------- #
#  Spectral analysis > FFT in time
# --------------------------------------------------------------------------- #
FFT_ORDERS = [("average", "Average over ROI first then spectral analysis"),
              ("fft", "Spectral analysis first then average over the ROI")]
# spectral function: key -> label
SPECTRAL_FUNCTIONS = [("fft", "FFT"), ("power", "Power spectrum"),
                      ("periodogram", "Periodogram")]
SPECTRAL_HELP = {
    "fft": "one-sided FFT, scaled as amplitudes (a sinusoid of amplitude A "
           "gives a peak of height A): amplitude, phase, real and "
           "imaginary parts",
    "power": "power spectral density by Welch's method: the record cut in "
             "segments (Hann window, 50 % overlap) whose periodograms are "
             "averaged -- smoother, coarser in frequency; (unit)\u00b2/Hz, "
             "its integral over f is the variance",
    "periodogram": "power spectral density of the whole record in one "
                   "piece (no window, no averaging): the finest frequency "
                   "resolution, noisy; (unit)\u00b2/Hz, its integral over f "
                   "is the variance",
}


def _quantity_cube(d, roi, q):
    """(cube (ny, nx, nt), view): a quantity for every frame on an ROI field
    (its stored '<quantity>_ROIxx', not calculated again) or on the full
    field ('Full': calculated once per mask and kept until the next Load)."""
    cal = d.can_calibrate
    if roi not in (None, "", "Full"):
        info = d.roi_fields()[roi]
        return roi_quantity(d, roi, q), RunView(d, True, info["window"])
    vw = RunView(d, cal, None)
    cache = d.__dict__.setdefault("_full_cubes", {})
    key = (q, _mask_key(d), cal)
    if key not in cache:
        import warnings
        u, v = vw.uv()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cache.clear()                        # one at a time: memory
            cache[key] = np.asarray(quantity(vw, u, v, q), np.float32)
    return cache[key], vw


def _fill_time(x):
    """A series with its NaN filled by linear interpolation in time (ends:
    the nearest value); None when more than half of it is NaN."""
    x = np.asarray(x, float)
    bad = ~np.isfinite(x)
    if not bad.any():
        return x
    if bad.sum() > 0.5 * x.size:
        return None
    k = np.arange(x.size)
    y = x.copy()
    y[bad] = np.interp(k[bad], k[~bad], x[~bad])
    return y


def _rfft_scaled(x, axis=-1):
    """One-sided FFT scaled as amplitudes: a sinusoid of amplitude A gives
    a peak of height A (|F| x 2/N; the mean, f = 0, x 1/N)."""
    n = x.shape[axis]
    F = np.fft.rfft(x, axis=axis)
    w = np.full(F.shape[axis], 2.0 / n)
    w[0] = 1.0 / n
    if n % 2 == 0:
        w[-1] = 1.0 / n                          # Nyquist
    shape = [1] * F.ndim
    shape[axis] = -1
    return F * w.reshape(shape)


def _detrend_kind(remove_mean, detrend):
    """scipy's detrend argument: 'linear' (detrend: the straight line --
    and so the mean -- removed), 'constant' (the mean removed) or False."""
    return "linear" if detrend else ("constant" if remove_mean else False)


def _prepare(x, remove_mean=False, detrend=False, axis=-1):
    """A series (or series along an axis) before its FFT: the linear trend
    removed (detrend, which also removes the mean), or the mean removed."""
    from scipy import signal
    if detrend:
        return signal.detrend(x, axis=axis, type="linear")
    if remove_mean:
        return x - np.mean(x, axis=axis, keepdims=True)
    return x


def _psd(x, fs, function, nperseg=None, axis=-1, remove_mean=True,
         detrend=False):
    """(f, power spectral density) of real series along an axis: Welch
    ('power': the trend / mean removed in each segment) or the raw
    periodogram ('periodogram': removed from the whole record), one-sided;
    remove_mean / detrend: see _detrend_kind."""
    from scipy import signal
    dk = _detrend_kind(remove_mean, detrend)
    if function == "power":
        n = x.shape[axis]
        seg = int(nperseg) if nperseg else max(
            16, 2 ** int(np.floor(np.log2(max(n // 4, 2)))))
        seg = min(seg, n)
        return signal.welch(x, fs=fs, window="hann", nperseg=seg,
                            noverlap=seg // 2, detrend=dk,
                            scaling="density", axis=axis)
    return signal.periodogram(x, fs=fs, window="boxcar", detrend=dk,
                              scaling="density", axis=axis)


def fft_in_time(d, roi, name, order="average", function="fft",
                nperseg=None, remove_mean=False, detrend=False):
    """Spectral analysis in time of a quantity of the time series list
    (TIME_SERIES) on an ROI field or the full field ('Full'). function:
    'fft' (amplitude, phase, real and imaginary parts), 'power' (power
    spectral density, Welch: segments of nperseg frames -- default a power
    of 2 near a quarter of the record --, Hann window, 50 % overlap) or
    'periodogram' (power spectral density of the whole record). For the
    two power spectra 'amp' is the density ((unit)^2/Hz) and phase, real,
    imag are None. remove_mean: the mean of each series removed first;
    detrend: its linear trend removed first (and so its mean too); for the
    power spectrum, in each segment.

    order 'average': the quantity is first averaged over the ROI (the time
      series: mean -- or standard deviation for the *_std ones -- over the
      points, of the quantity or its absolute value), then its FFT.
    order 'fft': the FFT of the quantity (or its absolute value) at each
      point of the ROI, then averaged over the ROI: real part, imaginary
      part = mean (or standard deviation for *_std) over the points of the
      real / imaginary parts; amplitude = mean (or std) of the amplitudes
      |F| of the points; phase = phase of the mean complex spectrum. The
      amplitude of each point is kept (FFT maps).

    NaN in time (rejected vectors) are filled by linear interpolation; a
    point (or the series) more than half NaN (e.g. masked) is left out.
    Spectrum: one-sided, f = 0 .. Nyquist, amplitude scaled so that a
    sinusoid of amplitude A gives a peak of height A.
    Returns dict(f, amp, phase, real, imag, unit, quantity, name, order,
    roi, n_points, n_left_out, dt, amp_points (ny, nx, nf) or None, x, z)."""
    import warnings
    spec = {t[0]: t for t in TIME_SERIES}[name]
    _n, q, stat, absolute, desc = spec
    cube, vw = _quantity_cube(d, roi, q)
    t = np.asarray(vw.t, float)
    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    nt = cube.shape[2]
    f = np.fft.rfftfreq(nt, dt)
    a = np.abs(cube) if absolute else cube
    out = dict(f=f, function=function, remove_mean=bool(remove_mean),
               detrend=bool(detrend),
               unit=quantity_unit(vw, q).replace("$^2$", "²"),
               quantity=q, name=name, desc=desc, order=order,
               roi=roi or "Full", dt=dt, amp_points=None,
               x=np.asarray(vw.x, float), z=np.asarray(vw.z, float),
               lunit=vw.lunit, funit="Hz" if vw.calibrated else "1/frame",
               tunit=vw.tunit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if order == "average":
            series = (np.nanmean(a, axis=(0, 1)) if stat == "mean"
                      else np.nanstd(a, axis=(0, 1)))
            s = _fill_time(series)
            if s is None:
                raise ValueError("the series over %s is more than half NaN"
                                 % (roi or "Full"))
            if function in ("power", "periodogram"):
                fp, Pxx = _psd(s, 1.0 / dt, function, nperseg,
                               remove_mean=remove_mean, detrend=detrend)
                out.update(f=fp, amp=Pxx, phase=None, real=None, imag=None,
                           n_points=int(np.isfinite(a).any(axis=2).sum()),
                           n_left_out=0)
                return out
            F = _rfft_scaled(_prepare(s, remove_mean, detrend))
            out.update(amp=np.abs(F), phase=np.angle(F), real=F.real,
                       imag=F.imag, n_points=int(np.isfinite(
                           a).any(axis=2).sum()), n_left_out=0)
            return out
        ny, nx = a.shape[:2]
        P = a.reshape(ny * nx, nt).astype(float)
        keep = np.ones(ny * nx, bool)
        for k in np.where(~np.isfinite(P).all(axis=1))[0]:
            y = _fill_time(P[k])
            if y is None:
                keep[k] = False
            else:
                P[k] = y
        if not keep.any():
            raise ValueError("no point of %s has enough valid values"
                             % (roi or "Full"))
        red = np.mean if stat == "mean" else np.std
        if function in ("power", "periodogram"):
            f, A = _psd(P[keep], 1.0 / dt, function, nperseg, axis=1,
                        remove_mean=remove_mean, detrend=detrend)
            out.update(f=f, amp=red(A, axis=0), phase=None, real=None,
                       imag=None)
        else:
            Fp = _rfft_scaled(_prepare(P[keep], remove_mean, detrend,
                                       axis=1), axis=1)
            A = np.abs(Fp)
            out.update(amp=red(A, axis=0), real=red(Fp.real, axis=0),
                       imag=red(Fp.imag, axis=0),
                       phase=np.angle(np.mean(Fp, axis=0)))
        out.update(n_points=int(keep.sum()), n_left_out=int((~keep).sum()))
        amp = np.full((ny * nx, f.size), np.nan, np.float32)
        amp[keep] = A
        out["amp_points"] = amp.reshape(ny, nx, f.size)
    return out


def _psd_unit(unit):
    """'m/s' -> '(m/s)\u00b2/Hz'."""
    u = unit if unit and " " not in unit and "/" not in unit else \
        "(%s)" % unit
    return "%s\u00b2/Hz" % u if unit else "1/Hz"


def band_mask(f, windows):
    """The frequencies used (bool over f): those of the range ticked in the
    list (one at a time), all frequencies when none is ticked. windows:
    [dict(fmin, fmax, on)] in Hz (or 1/frame)."""
    f = np.asarray(f, float)
    on = [w for w in windows if w.get("on")]
    if not on:
        return np.ones(f.shape, bool)
    m = np.zeros(f.shape, bool)
    eps = 1e-6 * (float(np.max(f)) if f.size else 1.0)  # ends included
    for w in on:
        m |= (f >= w["fmin"] - eps) & (f <= w["fmax"] + eps)
    return m


def _xmap(res, display):
    """(fn: f -> x, inverted, label, log x) of a spectrum's x axis. In time
    (FFT in time) f is in Hz: display 'frequency' (f) or 'period' (1/f).
    In space (FFT in space, res['kind'] == 'space') f is in cycles per
    unit length: display 'k' (wavenumber 2 pi f, rad/m), 'inv' (1/lambda
    = f, 1/m) or 'lambda' (wavelength 1/f, m)."""
    if res.get("kind") == "space":
        lu = res["lunit"]
        if display == "lambda":
            return (lambda f: 1.0 / np.asarray(f, float), True,
                    "wavelength λ (%s)" % lu, True)
        if display == "inv":
            return (lambda f: np.asarray(f, float), False,
                    "1/λ (1/%s)" % lu, False)
        return (lambda f: 2 * np.pi * np.asarray(f, float), False,
                "wavenumber %s (rad/%s)" % (res.get("kname", "k"), lu),
                False)
    if display == "period":
        return (lambda f: 1.0 / np.asarray(f, float), True,
                "period (%s)" % res["tunit"], True)
    return (lambda f: np.asarray(f, float), False,
            "frequency (%s)" % res["funit"], False)


def _fft_x(res, display):
    """(x values, keep, label, log) of the spectrum for a display (f = 0
    left out when the axis is inverted: period, wavelength)."""
    f = np.asarray(res["f"], float)
    fn, inv, lab, logx = _xmap(res, display)
    keep = f > 0 if inv else np.ones(f.shape, bool)
    return fn(f[keep]), keep, lab, logx


FFT_COLOURS = {"average": "#1f4e9c", "fft": "#d95f02", "space": "#1f4e9c"}


def fft_figure(d, res, display="frequency", windows=(), log_amp=False,
               fig=None, peaks=None, fmin=None, fmax=None):
    """The figure of FFT in time: amplitude, phase, real part, imaginary
    part (2 x 2, the same x axis); res: one result of fft_in_time or a list
    of them (both orders: one curve each, with a legend); the frequency
    ranges shaded (green: the one ticked, grey: the others); peaks: {order:
    find_spectral_peaks(...)} marked on the amplitude / density plot.
    fmin, fmax: the frequency range shown (None / 0: auto); in period
    display the periods 1/fmax .. 1/fmin. The y axes follow the values in
    that range (not the phase: -pi .. pi)."""
    fig = fig or new_figure()
    fig.clear()
    axs = fig.subplots(2, 2, sharex=True)
    results = [r for r in (res if isinstance(res, (list, tuple))
                           else [res]) if r is not None]
    res = results[0] if results else None
    if res is None:
        for ax in axs.ravel():
            ax.set_axis_off()
        axs[0, 0].text(0.5, 0.5, "Choose the ROI, the quantity and the "
                       "order, then Apply", transform=axs[0, 0].transAxes,
                       ha="left", va="center", color="0.4")
        return fig
    x, keep, xlab, logx = _fft_x(res, display)
    xfn, xinv = _xmap(res, display)[:2]
    u = res["unit"]
    psd = res.get("function", "fft") != "fft"
    if psd:                                      # one plot: the density
        fig.clear()
        axs = np.array([[fig.add_subplot(111)]])
        pu = res.get("psd_unit") or _psd_unit(u)
        if not res.get("psd_unit") and res["funit"] != "Hz":
            pu = pu.replace("Hz", res["funit"])
        panels = [(dict(SPECTRAL_FUNCTIONS)[res["function"]]
                   + " (power spectral density)", "amp", pu)]
    else:
        panels = [("amplitude", "amp", u), ("phase", "phase", "rad"),
                  ("real part", "real", u), ("imaginary part", "imag", u)]
    for ax, (title, key, unit) in zip(axs.ravel(), panels):
        for r in results:
            if r.get(key) is None:
                continue
            xr, kr, _l, _g = _fft_x(r, display)
            ax.plot(xr, r[key][kr], "-",
                    color=FFT_COLOURS.get(r["order"], "#1f4e9c"),
                    lw=1.1, marker=".", ms=3,
                    label={"average": "average then spectral analysis",
                           "fft": "spectral analysis then average"}.get(
                               r["order"], r.get("label", r["order"])))
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("%s (%s)" % (title, unit) if unit else title,
                      fontsize=8)
        ax.grid(True, color="0.9", lw=0.6)
        if logx:
            ax.set_xscale("log")
        for w in windows:
            a, b = w["fmin"], w["fmax"]
            if xinv:
                a, b = (float(xfn(b)) if b > 0 else np.inf), \
                    (float(xfn(a)) if a > 0 else np.inf)
            else:
                a, b = float(xfn(a)), float(xfn(b))
            ax.axvspan(a, b, color="#2ca02c" if w.get("on") else "0.6",
                       alpha=0.22 if w.get("on") else 0.12, lw=0)
    if len(results) > 1:                         # below the plots
        h, lab = axs[0, 0].get_legend_handles_labels()
        try:
            fig.legend(h, lab, loc="outside lower center", ncol=2,
                       fontsize=8, frameon=False)
        except Exception:                        # matplotlib < 3.7
            fig.legend(h, lab, loc="lower center", ncol=2, fontsize=8,
                       frameon=False)
    for r in results:                            # the peaks found
        pk = (peaks or {}).get(r["order"]) or []
        if not pk:
            continue
        ax = axs[0, 0]
        col = FFT_COLOURS.get(r["order"], "#1f4e9c")
        for p in pk:
            xp = float(xfn(p["f"]))
            ax.plot([xp], [p["value"]], "v", color=col,
                    mec="k", mew=0.6, ms=7, zorder=5)
            ax.annotate("%d" % p["rank"], (xp, p["value"]),
                        textcoords="offset points", xytext=(4, 6),
                        fontsize=7, color=col, zorder=6)
    if log_amp:
        axs[0, 0].set_yscale("log")
    _fft_xrange(axs, (xfn, xinv), fmin, fmax, skip_phase=not psd)
    if psd:
        axs[0, 0].set_xlabel(xlab)
        axs[0, 0].set_ylabel("power spectral density (%s)" % panels[0][2],
                             fontsize=8)
    else:
        axs[0, 1].set_ylim(-np.pi * 1.05, np.pi * 1.05)
        for ax in axs[1]:
            ax.set_xlabel(xlab)
    left = max(r.get("n_left_out") or 0 for r in results)
    short = {"average": "average then spectral analysis",
             "fft": "spectral analysis then average"}
    pre = ("detrended" if res.get("detrend") else "mean removed"
           if res.get("remove_mean") else "")
    fig.suptitle("%s\n%s   %s   %s%s   (%d %s%s)\n%s%s%s" % (
        d.run, res["roi"], res["name"],
        dict(SPECTRAL_FUNCTIONS)[res.get("function", "fft")],
        ", " + pre if pre else "",
        res["n_points"], res.get("points_word", "point(s)"),
        ", %d left out" % left if left else "",
        short[res["order"]] + " -- "
        if len(results) == 1 and res["order"] in short else "",
        res["desc"].replace("; ", "\n") if res.get("kind") == "space"
        else res["desc"], ", mask %s" % d.mask_name if d.mask_name else ""),
        fontsize=9)
    return fig


def find_spectral_peaks(res, n=3, windows=()):
    """The n dominant peaks of the amplitude (FFT) or density (power
    spectrum, periodogram) of a result of fft_in_time: the local maxima
    (scipy.signal.find_peaks; the ends count too), f = 0 left out, in the
    frequency range ticked (all frequencies if none), the n highest.
    Returns [dict(rank, f, period, value, index)] from the highest."""
    from scipy.signal import find_peaks
    f, a = np.asarray(res["f"], float), np.asarray(res["amp"], float)
    use = band_mask(f, windows) & (f > 0) & np.isfinite(a)
    if not use.any():
        return []
    y = np.where(use, a, -np.inf)
    pad = np.concatenate([[-np.inf], y, [-np.inf]])  # a maximum at an end
    idx, _ = find_peaks(pad)
    idx = idx - 1
    idx = idx[use[idx]]
    idx = idx[np.argsort(a[idx])[::-1]][:max(int(n), 0)]
    return [dict(rank=k + 1, f=float(f[i]), period=float(1.0 / f[i]),
                 value=float(a[i]), index=int(i))
            for k, i in enumerate(idx)]


def _fft_xrange(axs, display, fmin=None, fmax=None, skip_phase=True):
    """The x range of the FFT plots from fmin / fmax (in the unit of f: Hz,
    or cycles per length; None or 0: auto), and their y range from the
    values shown. display: 'period' / 'frequency', or (fn, inverted) of
    _xmap."""
    fmin = float(fmin) if fmin else None
    fmax = float(fmax) if fmax else None
    if fmin is not None and fmax is not None and fmax <= fmin:
        fmax = None                              # not a range: fmin only
    if fmin is None and fmax is None:
        return
    if isinstance(display, tuple):
        fn, inv = display
    else:
        inv = display == "period"
        fn = (lambda f: 1.0 / f) if inv else (lambda f: f)
    ax0 = axs.ravel()[0]
    lo, hi = ax0.get_xlim()
    if inv:
        if fmax is not None:
            lo = float(fn(fmax))
        if fmin is not None:
            hi = float(fn(fmin))
    else:
        if fmin is not None:
            lo = float(fn(fmin))
        if fmax is not None:
            hi = float(fn(fmax))
    if not hi > lo:
        return
    for k, ax in enumerate(axs.ravel()):
        ax.set_xlim(lo, hi)
        if skip_phase and k == 1:                # the phase: -pi .. pi
            continue
        ys = []
        for ln in ax.get_lines():
            x, y = (np.asarray(v, float) for v in ln.get_data())
            if x.size != y.size or x.size < 2:
                continue                         # markers, the 0 line
            m = (x >= lo) & (x <= hi) & np.isfinite(y)
            if ax.get_yscale() == "log":
                m &= y > 0
            ys.append(y[m])
        y = np.concatenate(ys) if ys else np.array([])
        if y.size:
            a, b = float(y.min()), float(y.max())
            if ax.get_yscale() == "log":
                ax.set_ylim(a / 1.5, b * 1.5)
            else:
                pad = 0.06 * (b - a) if b > a else (abs(a) or 1.0) * 0.1
                ax.set_ylim(a - pad, b + pad)


def fft_map_figure(d, res, windows=(), fig=None, shading="nearest",
                   cmap="auto", smoothing=0):
    """FFT maps (order 'fft'): at each point of the ROI, the FFT amplitude
    integrated over the frequency range used (band_mask): sum of |F| df."""
    fig = fig or new_figure()
    fig.clear()
    ax = fig.add_subplot(111)
    m = band_mask(res["f"], windows)
    df = res["f"][1] - res["f"][0] if res["f"].size > 1 else 1.0
    with np.errstate(all="ignore"):
        A = res["amp_points"]
        val = np.nansum(A[:, :, m], axis=2) * df
        val[~np.isfinite(A).any(axis=2)] = np.nan
    pc = draw_map(ax, res["x"], res["z"], smooth(val, smoothing), shading,
                  cmap=map_cmap(cmap))
    cb = fig.colorbar(pc, ax=ax)
    if res.get("function", "fft") == "fft":
        cb.set_label("integrated amplitude (%s %s)" % (res["unit"],
                                                      res["funit"]))
    else:
        cb.set_label("integrated power spectral density = variance in the "
                     "range (%s)" % _psd_unit(res["unit"])[:-3])
    ax.set_aspect("equal")
    ax.set_xlabel("x (%s)" % res["lunit"])
    ax.set_ylabel("z (%s)" % res["lunit"])
    fr = res["f"][m]
    band = ("%.4g .. %.4g %s, %d frequencies" % (fr.min(), fr.max(),
                                                  res["funit"], fr.size)
            if fr.size else "no frequency")
    used = ["W%d" % (k + 1) for k, w in enumerate(windows) if w.get("on")]
    ax.set_title("%s   %s   %s\n%s integrated over %s: %s" % (
        d.run, res["roi"], res["name"],
        "FFT amplitude" if res.get("function", "fft") == "fft" else
        dict(SPECTRAL_FUNCTIONS)[res["function"]],
        "range " + ", ".join(used) if used else "all frequencies", band),
        fontsize=9)
    return fig


def fft_data(d, res, windows=(), peaks=None):
    """Save data of the FFT figure (one group per order calculated, with
    its peaks found)."""
    results = [r for r in (res if isinstance(res, (list, tuple))
                           else [res]) if r is not None]
    out = {"run": d.run, "mask": d.mask_name or ""}
    for r in results:
        out[r["order"]] = {k: r[k] for k in (
            "f", "amp", "phase", "real", "imag", "unit", "name", "desc",
            "roi", "n_points", "dt", "function", "remove_mean", "detrend")
            if r.get(k) is not None}
        if r.get("function", "fft") != "fft":
            out[r["order"]]["psd"] = out[r["order"]].pop("amp")
            out[r["order"]]["psd_unit"] = _psd_unit(r["unit"])
        pk = (peaks or {}).get(r["order"])
        if pk:
            out[r["order"]]["peaks"] = {
                "rank": np.array([p["rank"] for p in pk]),
                "f": np.array([p["f"] for p in pk]),
                "period": np.array([p["period"] for p in pk]),
                "value": np.array([p["value"] for p in pk])}
    out["windows"] = {"W%d" % (k + 1): {"fmin": w["fmin"], "fmax": w["fmax"],
                                        "ticked": bool(w.get("on"))}
                      for k, w in enumerate(windows)}
    if results:
        out["frequencies_used"] = results[0]["f"][band_mask(
            results[0]["f"], windows)]
    return out


# --------------------------------------------------------------------------- #
#  Spectral analysis > FFT in space
# --------------------------------------------------------------------------- #
SPACE_DIRECTIONS = [("x", "along x (each row of the ROI)"),
                    ("z", "along z (each column of the ROI)"),
                    ("line", "along a line (Define ROI)")]
SPACE_WINDOWS = [("none", "none (rectangular)"), ("hann", "Hann")]


def _fill_rows(P):
    """Series along the last axis with their NaN filled by linear
    interpolation; a series more than half NaN becomes all NaN."""
    P = np.array(P, float, copy=True)
    flat = P.reshape(-1, P.shape[-1])
    bad = ~np.isfinite(flat)
    for k in np.where(bad.any(axis=1))[0]:
        y = _fill_time(flat[k])
        flat[k] = np.nan if y is None else y
    return flat.reshape(P.shape)


def space_series(d, roi, q, direction, line=None):
    """(P (nseries, nt, n), step, lunit, unit, view, text): the series of a
    quantity (ROI_QUANTITIES) in space for every frame -- along x: each row
    of the ROI (or the full field), along z: each column, along a line:
    the values stored along the line (Define ROI > Apply) in that region.
    step: the spacing of the points (m, or px uncalibrated)."""
    if direction == "line":
        if not line:
            raise ValueError("choose a line")
        lines = dict(parse_shapes(d.params.get("LINES"), "LINE"))
        if line not in lines:
            raise ValueError("%s is not a line of Define ROI" % line)
        cal = d.can_calibrate
        vw, info = _roi_view(d, cal, roi)
        e = _stored_line(d, lines[line], q, roi, vw.calibrated)
        if e is None:
            compute_line_quantities(d, {line: lines[line]},
                                    rois=[info["roi"]] if info else [],
                                    calibrated=vw.calibrated, prune=False)
            e = _stored_line(d, lines[line], q, roi, vw.calibrated)
        if e is None or len(e["s"]) < 4:
            raise ValueError("%s does not cross %s (or has fewer than 4 "
                             "points in it)" % (line, roi or "Full"))
        s = np.asarray(e["s"], float)
        step = float(np.median(np.diff(s)))
        P = np.asarray(e["data"], float).T[None, :, :]   # (1, nt, npts)
        return (P, step, e["lunit"], e["unit"], vw,
                "%s in %s, %d points every %.3g %s" % (
                    line, e["roi"], s.size, step, e["lunit"]))
    cube, vw = _quantity_cube(d, roi, q)
    x, z = np.asarray(vw.x, float), np.asarray(vw.z, float)
    if direction == "x":
        P = np.transpose(cube, (0, 2, 1))        # rows: (ny, nt, nx)
        g = x[0, :]
    else:
        P = np.transpose(cube, (1, 2, 0))        # columns: (nx, nt, ny)
        g = z[:, 0]
    if g.size < 4:
        raise ValueError("fewer than 4 points %s in %s" % (
            "along x" if direction == "x" else "along z", roi or "Full"))
    step = float(np.median(np.abs(np.diff(g))))
    if direction == "z" and g[0] > g[-1]:        # increasing z
        P = P[:, :, ::-1]
    return (P, step, vw.lunit, quantity_unit(vw, q).replace("$^2$", "²"),
            vw, "%d %s of %d points every %.3g %s" % (
                P.shape[0], "rows" if direction == "x" else "columns",
                P.shape[2], step, vw.lunit))


def fft_in_space(d, roi, q, direction="x", line=None, function="fft",
                 nperseg=None, remove_mean=True, detrend=False,
                 window="none", frame="mean"):
    """Spectral analysis in space of a quantity (U, V, velocity norm,
    divergence, vorticity, kinetic energy) on an ROI field or the full
    field ('Full'): along x (the FFT of each row), along z (of each
    column) or along a line of Define ROI (its stored values in that
    region), at every frame. NaN are filled by linear interpolation along
    the series; a series more than half NaN is left out. function: 'fft'
    (one-sided, amplitudes: a sinusoid of amplitude A gives a peak of
    height A; window 'hann': the amplitudes corrected by the window's
    mean), 'power' (Welch, nperseg points, Hann, 50 %) or 'periodogram'
    (window as chosen) -- power spectral density per cycle per length.
    The spectra of the series are averaged at each frame (amplitude: mean
    of |F|, real / imaginary parts: mean, phase: of the mean complex
    spectrum; density: mean); frame: an index, or 'mean' (the frames
    averaged the same way). f: cycles per length (k = 2 pi f, lambda =
    1/f). Returns a dict as fft_in_time (kind 'space') plus amp_t (nt, nk)
    and t, for the k-t diagram."""
    import warnings
    P, step, lunit, unit, vw, text = space_series(d, roi, q, direction,
                                                  line)
    nser, nt, n = P.shape
    P = _fill_rows(P)
    ok = np.isfinite(P).all(axis=2)              # (nser, nt)
    if not ok.any():
        raise ValueError("no series with enough valid values")
    fs = 1.0 / step
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        X = np.where(ok[:, :, None], P, 0.0)
        if function in ("power", "periodogram"):
            from scipy import signal
            dk = _detrend_kind(remove_mean, detrend)
            if function == "power":
                seg = int(nperseg) if nperseg else max(
                    8, 2 ** int(np.floor(np.log2(max(n // 2, 2)))))
                seg = min(seg, n)
                f, A = signal.welch(X, fs=fs, window="hann", nperseg=seg,
                                    noverlap=seg // 2, detrend=dk,
                                    scaling="density", axis=-1)
            else:
                f, A = signal.periodogram(
                    X, fs=fs, window="hann" if window == "hann" else
                    "boxcar", detrend=dk, scaling="density", axis=-1)
            A = np.where(ok[:, :, None], A, np.nan)
            amp_t = np.nanmean(A, axis=0)        # (nt, nk)
            Fm = None
        else:
            Y = _prepare(X, remove_mean, detrend, axis=-1)
            if window == "hann":
                w = np.hanning(n)
                Y = Y * w / w.mean()
            F = _rfft_scaled(Y, axis=-1)
            F = np.where(ok[:, :, None], F, np.nan)
            f = np.fft.rfftfreq(n, step)
            amp_t = np.nanmean(np.abs(F), axis=0)
            Fm = np.nanmean(F, axis=0)           # (nt, nk) complex
        if frame == "mean":
            amp = np.nanmean(amp_t, axis=0)
            Fs = np.nanmean(Fm, axis=0) if Fm is not None else None
            label = "mean over the %d frames" % nt
        else:
            i = int(frame)
            amp = amp_t[i]
            Fs = Fm[i] if Fm is not None else None
            label = d.frame_text(i)
    kname = {"x": "k_x", "z": "k_z"}.get(direction, "k")
    cal = vw.calibrated
    out = dict(kind="space", direction=direction, line=line, kname=kname,
               f=np.asarray(f, float), amp=np.asarray(amp, float),
               phase=None if Fs is None else np.angle(Fs),
               real=None if Fs is None else Fs.real,
               imag=None if Fs is None else Fs.imag,
               amp_t=np.asarray(amp_t, np.float32),
               t=np.asarray(vw.t, float), tunit=vw.tunit,
               unit=unit, quantity=q, name=q, roi=roi or "Full",
               desc="%s, %s; %s" % (dict(SPACE_DIRECTIONS)[direction]
                                    .split(" (")[0], text, label),
               frame=frame, frame_text=label, order="space",
               label="spectrum %s" % kname, function=function,
               remove_mean=bool(remove_mean), detrend=bool(detrend),
               window=window, step=step, lunit=lunit,
               funit="1/%s" % lunit,
               psd_unit="(%s)²·%s" % (unit, lunit) if unit
               else lunit,
               n_points=int(ok.any(axis=1).sum()),
               points_word="series" if direction != "line" else "line",
               n_left_out=int((~ok.any(axis=1)).sum()),
               calibrated=cal)
    return out


def kt_figure(d, res, display="k", kmin=None, kmax=None, log=True,
              cmap="viridis", fig=None):
    """The k-t diagram: the amplitude (FFT) or density (power spectrum,
    periodogram) of the spatial spectrum at every frame -- x: the
    wavenumber (or 1/lambda, or wavelength), y: the time (or frame)."""
    from matplotlib.colors import LogNorm
    fig = fig or new_figure()
    fig.clear()
    ax = fig.add_subplot(111)
    if res is None:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Choose the wavenumber, then Plot k-t diagram",
                transform=ax.transAxes, ha="center", va="center",
                color="0.4")
        return fig
    fn, inv, xlab, logx = _xmap(res, display)
    f = np.asarray(res["f"], float)
    keep = f > 0 if (inv or log) else np.ones(f.shape, bool)
    xk = fn(f[keep])
    A = np.asarray(res["amp_t"], float)[:, keep]
    t = np.asarray(res["t"], float)
    order = np.argsort(xk)
    xk, A = xk[order], A[:, order]
    good = A[np.isfinite(A) & (A > 0)] if log else A[np.isfinite(A)]
    norm = None
    if log and good.size:
        lo, hi = np.percentile(good, (1, 99.9))
        norm = LogNorm(vmin=max(lo, hi * 1e-8), vmax=hi)
    im = ax.pcolormesh(xk, t, np.where(np.isfinite(A), A, np.nan),
                       shading="nearest", cmap=cmap, norm=norm)
    psd = res["function"] != "fft"
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(("power spectral density (%s)" % res["psd_unit"]) if psd
                 else "amplitude (%s)" % res["unit"], fontsize=8)
    if logx:
        ax.set_xscale("log")
    lo = hi = None
    if kmin or kmax:                             # kmin / kmax in rad/length
        fmin = kmin / (2 * np.pi) if kmin else None
        fmax = kmax / (2 * np.pi) if kmax else None
        a = fn(fmax) if (inv and fmax) else (fn(fmin) if fmin else None)
        b = fn(fmin) if (inv and fmin) else (fn(fmax) if fmax else None)
        lo, hi = ax.get_xlim()
        ax.set_xlim(a if a is not None else lo, b if b is not None else hi)
    ax.set_xlabel(xlab)
    ax.set_ylabel("time (%s)" % res["tunit"])
    pre = ("detrended" if res.get("detrend") else "mean removed"
           if res.get("remove_mean") else "")
    ax.set_title("%s\n%s   %s   %s%s%s\n%s%s" % (
        d.run, res["roi"], res["name"],
        dict(SPECTRAL_FUNCTIONS)[res["function"]],
        ", " + pre if pre else "",
        ", Hann window" if res.get("window") == "hann" else "",
        res["desc"].split(";")[0],
        ", mask %s" % d.mask_name if d.mask_name else ""), fontsize=9)
    return fig


def fft_space_data(d, res, peaks=None):
    """Save data of FFT in space (and of the k-t diagram)."""
    if res is None:
        return {}
    psd = res["function"] != "fft"
    out = {"run": d.run, "mask": d.mask_name or "", "roi": res["roi"],
           "quantity": res["quantity"], "unit": res["unit"],
           "direction": res["direction"], "line": res.get("line") or "",
           "wavenumber_name": res["kname"],
           "function": res["function"], "window": res.get("window"),
           "remove_mean": res["remove_mean"], "detrend": res["detrend"],
           "frame": res["frame_text"], "spacing": res["step"],
           "length_unit": res["lunit"],
           "f_cycles_per_length": res["f"],
           "k_rad_per_length": 2 * np.pi * res["f"],
           "wavelength": np.where(res["f"] > 0, 1.0 / np.where(
               res["f"] > 0, res["f"], 1), np.inf),
           ("psd" if psd else "amplitude"): res["amp"],
           "kt_" + ("psd" if psd else "amplitude"): res["amp_t"],
           "t": res["t"], "time_unit": res["tunit"]}
    if psd:
        out["psd_unit"] = res["psd_unit"]
    else:
        out.update(phase=res["phase"], real=res["real"], imag=res["imag"])
    if peaks:
        out["peaks"] = {
            "rank": np.array([p["rank"] for p in peaks]),
            "k_rad_per_length": np.array([2 * np.pi * p["f"]
                                          for p in peaks]),
            "wavelength": np.array([p["period"] for p in peaks]),
            "value": np.array([p["value"] for p in peaks])}
    return out


def fft_map_data(d, res, windows=()):
    m = band_mask(res["f"], windows)
    df = res["f"][1] - res["f"][0] if res["f"].size > 1 else 1.0
    with np.errstate(all="ignore"):
        val = np.nansum(res["amp_points"][:, :, m], axis=2) * df
    return {"run": d.run, "roi": res["roi"], "name": res["name"],
            "x": res["x"], "z": res["z"], "integrated_amplitude": val,
            "frequencies_used": res["f"][m], "unit": res["unit"]}


# --------------------------------------------------------------------------- #
#  Movies (Processing > Maps and Quiver > Make Movie)
# --------------------------------------------------------------------------- #
def find_ffmpeg():
    """The ffmpeg executable: on the PATH, else the one of the Python
    package imageio-ffmpeg; None when there is none (movies are then
    animated GIFs)."""
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def movie_limits(d, opts, frames, samples=60):
    """(u_max, bg_max): fixed colour limits for a movie -- the 99th
    percentile of |u| and of |background quantity| over up to `samples`
    frames of the range (the ROI field, the mask applied)."""
    info = d.roi_fields().get(opts.get("roi")) \
        if opts.get("roi") not in (None, "Full") else None
    vw = RunView(d, d.calibrated, info["window"] if info else None)
    idx = np.unique(np.linspace(frames[0], frames[-1],
                                min(samples, len(frames))).astype(int))
    sp, bq = [], []
    for i in idx:
        u, v = vw.uv(int(i))
        n = np.hypot(u, v)
        sp.append(n[np.isfinite(n)])
        bg = opts.get("background")
        if bg in QUANTITIES and bg != "velocity norm":
            q = np.abs(np.asarray(frame_quantity(d, vw, int(i), bg)[0],
                                  float))
            bq.append(q[np.isfinite(q)])
    cat = lambda L: np.concatenate(L) if L and sum(a.size for a in L) else \
        np.array([1.0])
    u_max = float(np.percentile(cat(sp), 99)) or 1.0
    bg_max = float(np.percentile(cat(bq), 99)) if bq else None
    return u_max, bg_max


def make_movie(d, opts, frames, fps, path, size_in=(10, 7.5), dpi=128,
               progress=None, cancelled=None, on_frame=None, kind="maps"):
    """A movie of the Maps and Quiver figure over `frames` (frame indices),
    drawn with the options `opts` (display_figure) and fixed colour limits
    (movie_limits). With ffmpeg: an H.264 MP4 (yuv420p, +faststart: plays
    on macOS -- QuickTime -- and Windows); else an animated GIF (path's
    extension changed). progress(i, n) after each frame; cancelled() True
    stops it (the file is removed). on_frame(k, frame, rgb) receives each
    image of the movie (uint8 array h x w x 3), e.g. for a player.
    kind 'maps': the Maps and Quiver figure; 'profiles': the Profiles
    figure (opts of profile_figure, fixed y limits: profile_limits). Returns {path, frames, fps, size,
    format, seconds} or None when cancelled."""
    import subprocess
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    frames = [int(f) for f in frames]
    o = dict(opts)
    for k in ("frame", "mask", "field"):
        o.pop(k, None)
    u_max = bg_max = ylim = None
    if kind == "profiles":
        ylim = profile_limits(d, opts, frames)
        draw = lambda i: profile_figure(d, i, ylim=ylim, fig=fig, **o)
    else:
        u_max, bg_max = movie_limits(d, opts, frames)
        draw = lambda i: display_figure(d, i, u_max=u_max, bg_max=bg_max,
                                        fig=fig, **o)
    # even pixel size (H.264 / yuv420p)
    w = int(round(size_in[0] * dpi)) // 2 * 2
    h = int(round(size_in[1] * dpi)) // 2 * 2
    fig = new_figure((w / dpi, h / dpi))
    fig.set_dpi(dpi)
    canvas = FigureCanvasAgg(fig)
    ff = find_ffmpeg()
    gif = ff is None
    if gif:
        path = os.path.splitext(path)[0] + ".gif"
        images = []
    else:
        cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo",
               "-vcodec", "rawvideo", "-s", "%dx%d" % (w, h),
               "-pix_fmt", "rgba", "-r", "%g" % fps, "-i", "-", "-an",
               "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-preset",
               "medium", "-crf", "18", "-movflags", "+faststart", path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    try:
        for k, i in enumerate(frames):
            if cancelled is not None and cancelled():
                raise KeyboardInterrupt
            draw(i)
            canvas.draw()
            buf = np.asarray(canvas.buffer_rgba())
            if buf.shape[:2] != (h, w):          # layout rounding
                pad = np.zeros((h, w, 4), np.uint8)
                pad[..., 3] = 255
                hh, ww = min(h, buf.shape[0]), min(w, buf.shape[1])
                pad[:hh, :ww] = buf[:hh, :ww]
                buf = pad
            if gif:
                from PIL import Image
                images.append(Image.fromarray(buf[..., :3].copy()))
            else:
                proc.stdin.write(np.ascontiguousarray(buf).tobytes())
            if on_frame is not None:
                on_frame(k, i, buf[..., :3])
            if progress is not None:
                progress(k + 1, len(frames))
    except KeyboardInterrupt:
        if not gif:
            proc.stdin.close()
            proc.wait()
        if os.path.exists(path):
            os.remove(path)
        return None
    if gif:
        images[0].save(path, save_all=True, append_images=images[1:],
                       duration=int(round(1000.0 / fps)), loop=0)
    else:
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed: %s" % err.strip()[-800:])
    return {"path": path, "frames": len(frames), "fps": fps,
            "size": (w, h), "format": "GIF" if gif else "MP4 (H.264)",
            "seconds": len(frames) / float(fps), "first": frames[0],
            "last": frames[-1], "u_max": u_max, "bg_max": bg_max,
            "ylim": ylim, "kind": kind}


def display_data(d, frame=0, field=None, roi="Full"):
    """The data of the Velocity figure, for Save data: grid, the field
    shown (or its mean / rms), its norm (of the ROI field when one is
    shown) and the time base."""
    info = d.roi_fields().get(roi) if roi and roi != "Full" else None
    vw = RunView(d, d.calibrated, info["window"] if info else None)
    u, v, label = _field(d, frame, view=vw)
    out = {"run": d.run, "frame": label,
           "vector_field": roi if info else d.default,
           "units": {"length": vw.lunit, "velocity": vw.vunit,
                     "time": vw.tunit},
           "x": np.asarray(vw.x), "z": np.asarray(vw.z),
           "u": np.asarray(u, float), "v": np.asarray(v, float),
           "norm": np.hypot(np.asarray(u, float), np.asarray(v, float)),
           "dt_pulse_s": d.acq.dt, "camera_fps_Hz": d.acq.cam_fps,
           "piv_fps_Hz": d.acq.fps, "mask": d.mask_name or ""}
    for name in QUANTITIES:                      # the derived maps
        if name in ("velocity norm", "U", "V"):
            continue
        q, _lab = frame_quantity(d, vw, frame, name)
        out[name.replace(" ", "_")] = np.asarray(q, float)
    if info is not None:
        out["roi"] = info["roi"]
        out["roi_rectangle_m"] = np.asarray(info["rect"], float)
        out["roi_window_rows_cols"] = np.asarray(info["window"])
    if not isinstance(frame, str):
        out["frame_index"] = int(frame)
        if d.calibrated:
            out["t_s"] = float(d.t[int(frame)])
    return out


def selection_figure(d, source="frame", frame=0, image=None, fig=None,
                     cmap="auto", smoothing=0, shading="nearest"):
    """(fig, ax, view): the base figure of Define ROI and Mask, in the
    calibrated view when the run can be calibrated (else px): the current
    frame (|u| as a colour map with the vectors; masked vectors NaN) or an
    image placed on the PIV grid."""
    view = d.view()
    fig = fig or new_figure()
    fig.clear()
    ax = fig.add_subplot(111)
    x, z = view.x, view.z
    if source == "image" and image and os.path.isfile(image):
        img, extent = place_background(d, read_image(image), view.pix_scale)
        good = img[np.isfinite(img)] if img.ndim == 2 else img
        vlo, vhi = (np.percentile(good, (1.0, 99.5)) if img.ndim == 2
                    and good.size else (None, None))
        ax.imshow(img, cmap="gray", origin="upper", extent=extent,
                  vmin=vlo, vmax=vhi, interpolation="nearest", zorder=0)
        what = os.path.basename(image)
    else:
        u, v, what = _field(d, frame, view=view)
        sp = np.hypot(u, v)
        fin = sp[np.isfinite(sp)]
        vmax = (float(np.percentile(fin, 99)) if fin.size else 1.0) or 1.0
        mp = draw_map(ax, x, z, smooth(sp, smoothing), shading,
                      cmap=map_cmap(cmap), vmin=0, vmax=vmax,
                           zorder=0)
        fig.colorbar(mp, ax=ax, label=r"$|\mathbf{u}|$  (%s)" % view.vunit)
        sk = max(1, int(max(x.shape) / 40))
        ax.quiver(x[::sk, ::sk], z[::sk, ::sk], u[::sk, ::sk],
                  v[::sk, ::sk], color="k", pivot="mid", width=0.002,
                  angles="xy", zorder=1)
    px = 0.02 * (np.nanmax(x) - np.nanmin(x))
    pz = 0.02 * (np.nanmax(z) - np.nanmin(z))
    ax.set_xlim(np.nanmin(x) - px, np.nanmax(x) + px)
    ax.set_ylim(np.nanmin(z) - pz, np.nanmax(z) + pz)
    ax.set_aspect("equal")
    ax.set_xlabel("x (%s)" % view.lunit)
    ax.set_ylabel("z (%s)" % view.lunit)
    ax.set_title("%s\n%s" % (d.run, what), fontsize=9)
    return fig, ax, view


# a colour per ROI (rectangle) and per line, by their place in the lists;
# the two palettes share no colour
ROI_COLOURS = ["#e41a1c", "#ff7f00", "#00e5ff", "#ffffff", "#f781bf",
               "#a6d854", "#ffd92f", "#8c564b"]
LINE_COLOURS = ["#ff00ff", "#2b6cff", "#000000", "#4daf4a", "#984ea3",
                "#bcbd22", "#7f7f7f", "#fdbf6f"]


def roi_colour(kind, index):
    """The colour of the index-th ROI (kind 'rect') or line ('line')."""
    pal = ROI_COLOURS if kind == "rect" else LINE_COLOURS
    return pal[index % len(pal)]


def roi_figure(d, source="frame", frame=0, image=None, rois=(), lines=(),
               selected=None, cmap="auto", fig=None, shading="nearest",
               smoothing=0):
    """The figure of Pre-processing > Define ROI (calibrated view, m): the
    current frame or an image, with the ROIs (dashed rectangles) and the
    lines (segments) given as [(name, ((x1, z1), (x2, z2))[, colour])] --
    each in its own colour (roi_colour by default); the `selected` name is
    drawn thicker."""
    import matplotlib.patheffects as pe
    fig, ax, view = selection_figure(d, source, frame, image, fig, cmap,
                                     smoothing=smoothing, shading=shading)
    halo = [pe.withStroke(linewidth=2.5, foreground="0.15", alpha=0.6)]
    for k, item in enumerate(rois):
        name, ((xa, za), (xb, zb)) = item[:2]
        c = item[2] if len(item) > 2 else roi_colour("rect", k)
        rx, rz = sorted((xa, xb)), sorted((za, zb))
        lw = 2.6 if name == selected else 1.6
        ax.plot([rx[0], rx[1], rx[1], rx[0], rx[0]],
                [rz[0], rz[0], rz[1], rz[1], rz[0]], "--", color=c,
                lw=lw, zorder=3)
        ax.text(rx[0], rz[1], " " + name, color=c, fontsize=9,
                va="bottom", zorder=3, path_effects=halo)
    for k, item in enumerate(lines):
        name, ((xa, za), (xb, zb)) = item[:2]
        c = item[2] if len(item) > 2 else roi_colour("line", k)
        lw = 3.0 if name == selected else 1.8
        ax.plot([xa, xb], [za, zb], "-o", color=c, lw=lw, ms=4,
                zorder=3)
        ax.text(xb, zb, " " + name, color=c, fontsize=9, zorder=3,
                path_effects=halo)
    return fig


def area_outline(kind, pts):
    """(x, z) of the outline of an area (rect, circle, poly)."""
    p = np.asarray(pts, float)
    if kind == "rect":
        (x0, z0), (x1, z1) = p[:2]
        return [x0, x1, x1, x0, x0], [z0, z0, z1, z1, z0]
    if kind == "circle":
        (xc, zc), (xe, ze) = p[:2]
        r = np.hypot(xe - xc, ze - zc)
        t = np.linspace(0, 2 * np.pi, 121)
        return xc + r * np.cos(t), zc + r * np.sin(t)
    if len(p) == 0:
        return [], []
    q = np.vstack([p, p[:1]]) if len(p) > 2 else p
    return q[:, 0], q[:, 1]


def mask_figure(d, source="frame", frame=0, image=None, areas=(),
                masks=(), selected=None, cmap="auto", fig=None,
                shading="nearest", smoothing=0):
    """The figure of Pre-processing > Mask: the current frame or an image,
    the selection areas [(name, kind, points in the view's unit, ticked)]
    (ticked: orange, unticked: grey dotted) and the masks to show
    [(name, mask (ny, nx), colour)] -- each shaded in its colour."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    fig, ax, view = selection_figure(d, source, frame, image, fig, cmap,
                                     smoothing=smoothing, shading=shading)
    handles = []
    for name, m, colour in masks:
        if m is None or not np.any(m):
            continue
        ax.pcolormesh(view.x, view.z, np.ma.masked_where(~m, m.astype(float)),
                      shading="auto", cmap=ListedColormap([colour]),
                      vmin=0, vmax=1, alpha=0.5, zorder=1.5)
        handles.append(Patch(color=colour, alpha=0.5, label="%s (%d)" % (
            name, int(m.sum()))))
    if handles:
        ax.legend(handles=handles, loc="upper left", fontsize=8,
                  framealpha=0.8)
    mask = None
    for name, kind, pts, ticked in areas:
        xo, zo = area_outline(kind, pts)
        lw = 2.8 if name == selected else 1.6
        ax.plot(xo, zo, "-" if ticked else ":", lw=lw, zorder=3,
                color="orange" if ticked else "0.5")
        if len(xo):
            ax.text(np.max(xo), np.max(zo), " " + name, fontsize=9,
                    color="orange" if ticked else "0.5", zorder=3)
    ax.set_title(ax.get_title() + "\nmask applied: %s" % (
        "%s (%d vectors)" % (d.mask_name, int(d.mask.sum()))
        if d.mask is not None else "none"), fontsize=9)
    return fig


# --------------------------------------------------------------------------- #
#  Calibration
# --------------------------------------------------------------------------- #
def find_calibration_images(run_dir, root=None):
    """Candidate calibration images (as find_background_images), the files
    whose name holds 'calib' first."""
    paths = find_background_images(run_dir, root)
    return sorted(paths, key=lambda p: "calib" not in
                  os.path.basename(p).lower())


def rotate_image_px(img, rotate):
    """An image rotated into the analysis frame, in pixel coordinates (as
    Calibration_PIV.py): a 90-deg rotation keeps pixel distances, so the
    measured length does not depend on ROTATE."""
    k = {0: 0, 180: 2, -90: 1, 90: 3}.get(int(rotate))
    if k is None:
        raise ValueError("ROTATE must be 0, 90, 180 or -90, got %r" % rotate)
    return np.rot90(img, k) if k else img


def rotate_points_px(points, rot_from, rot_to, shape):
    """Points (x, y) in px of an image shown rotated by rot_from, moved to
    the same image shown rotated by rot_to (rotate_image_px); shape: (rows,
    columns) of the image NOT rotated."""
    K = {0: 0, 180: 2, -90: 1, 90: 3}
    k0, k1 = K[int(rot_from)], K[int(rot_to)]
    h, w = int(shape[0]), int(shape[1])
    if k0 % 2:
        h, w = w, h                              # the shape as shown
    out = []
    for x, y in points:
        x, y = float(x), float(y)
        hh, ww = h, w
        for _ in range((k1 - k0) % 4):           # np.rot90 once (CCW)
            x, y = y, ww - 1 - x
            hh, ww = ww, hh
        out.append((x, y))
    return out


def segment_length(points):
    """Length [px] of the segment between its two points [(x, y), (x, y)]
    (0 when it is not complete)."""
    p = np.asarray(points, float)
    if len(p) < 2:
        return 0.0
    return float(np.hypot(*(p[1] - p[0])))


def calibration_figure(path, rotate=0, clip=(1.0, 99.5), segment=None,
                       result=None, fig=None):
    """The calibration image in pixels (rotated by ROTATE, grey, contrast
    stretched to the `clip` percentiles); optionally the validated segment
    and its calibration text."""
    img = rotate_image_px(read_image(path), rotate)
    fig = fig or new_figure()
    fig.clear()
    ax = fig.add_subplot(111)
    vlo = vhi = None
    if clip and img.ndim == 2:
        good = img[np.isfinite(img)]
        if good.size:
            vlo, vhi = np.percentile(good, clip)
    ax.imshow(img, cmap="gray", origin="upper", vmin=vlo, vmax=vhi,
              interpolation="nearest")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    title = "%s   (ROTATE = %d deg)" % (os.path.basename(path), int(rotate))
    if segment is not None and len(segment) >= 2:
        s = np.asarray(segment, float)
        ax.plot(s[:, 0], s[:, 1], "-o", color="red", lw=1.8, ms=4)
    if result:
        title += "\n" + result
    ax.set_title(title, fontsize=10)
    return fig


def sections_data(sections):
    """A run-information table [(title, [(quantity, value)])] as a nested
    dict, for Save data."""
    import re as _re
    clean = lambda s: _re.sub(r"[^0-9A-Za-z]+", "_", str(s)).strip("_") or "x"
    return {clean(t): {clean(k): str(v) for k, v in rows}
            for t, rows in sections}
