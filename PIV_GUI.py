#!/usr/bin/env python
"""PIVlab processing -- desktop GUI (Qt + matplotlib), version 1.

Start it from this folder with the Anaconda interpreter:

    ~/anaconda3/bin/python PIV_GUI.py

Window (laid out as the DTS processing GUI)
------
  menus      File (Load data / New session / Load session / Save session /
             Save variables),
             Calibration (Calibration, Define ROI),
             Processing (Divergence, Vorticity, Energy, Velocity),
             Spectral analysis (FFT in time, FFT in space, 2D FFT,
             f-Filter, k-Filter) -- in the menu bar AND repeated as a
             toolbar of drop-down buttons at the top of the window and of
             every separate figure window. Choosing an entry opens its
             parameter panel on the left and brings its figures forward.
  left       the parameters of the current step
  centre     the ticked figures, as tabs (matplotlib toolbar: zoom, pan,
             cursor readout, save image); closing a tab unticks it
  right      the list of figures with tick boxes, grouped by step

List of figures: tick = shown as a tab; click a name = bring it forward;
double click = show it in the central panel (back from its separate
window). Right click (list, figure or table): Focus, Open in separate
window, Save data... (npz, NetCDF, HDF5, txt, csv, MATLAB -- the DTS GUI's
gui_export), Update (redraw with the current parameters).

V1 implements File > Load data (one run at a time) and the sessions; the
other steps show an empty panel. The work is done by piv_gui_lib.py.
"""

import json
import os
import sys
import time

import numpy as np
import traceback

import matplotlib
try:
    matplotlib.use("QtAgg")
except Exception:                                   # older matplotlib
    matplotlib.use("Qt5Agg")
from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets
try:
    from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg,
                                                   NavigationToolbar2QT)
except ImportError:
    from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg,
                                                    NavigationToolbar2QT)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import piv_gui_lib as lib  # noqa: E402

Signal = getattr(QtCore, "Signal", None) or getattr(QtCore, "pyqtSignal")
MENU_STYLE = ("QMenuBar { font-size: 15px; font-weight: bold; } "
              "QMenu { font-size: 14px; font-weight: bold; }")
APP_NAME = "PIVlab processing"
CALIBRATED, UNCALIBRATED = "calibrated", "uncalibrated"
def json_filter(label, *exts):
    """A file-dialog filter for files with a double extension (e.g.
    .pivcalib.json): the pattern is *.json -- the macOS native dialog greys
    out the files of a pattern such as *.pivcalib.json --, the extension
    named in the label."""
    return "%s \u2014 %s (*.json);;All files (*)" % (
        label, " / ".join(exts))


def with_double_ext(path, ext):
    """The path ending with the double extension ext (.pivcalib.json, ...):
    'name' or 'name.json' (what a save dialog filtering *.json gives) ->
    'name' + ext; the Load data lists find the files by it."""
    if path.lower().endswith(ext):
        return path
    if path.lower().endswith(".json"):
        path = path[:-5]
    if path.lower().endswith(ext[:-5]):          # 'name.pivcalib'
        return path + ".json"
    return path + ext


SESSION_EXT = json_filter("PIV session", ".pivsession.json",
                          ".session.json")
# the files of the steps, loaded by Load data (no parameter file)
CAL_EXT = ".pivcalib.json"
ROI_EXT = ".pivroi.json"
MASK_EXT = ".pivmask.json"



def _qexec(obj):
    """obj.exec (PySide6, PyQt6, recent PyQt5), else obj.exec_ (old PyQt5) --
    exec_ is deprecated in PySide6 (DeprecationWarning)."""
    f = getattr(obj, "exec", None)
    return f if callable(f) else obj.exec_

def E(path):
    """A Qt enum by dotted path, scoped (Qt6) or not (Qt5):
    E('Qt.Orientation.Horizontal')."""
    parts = path.split(".")
    if parts[0] == "Qt":
        obj = QtCore.Qt
    else:
        obj = getattr(QtWidgets, parts[0], None) or getattr(QtGui, parts[0])
    try:
        cur = obj
        for p in parts[1:]:
            cur = getattr(cur, p)
        return cur
    except AttributeError:
        return getattr(obj, parts[-1])


# --------------------------------------------------------------------------- #
#  The processing steps (menu entries)
# --------------------------------------------------------------------------- #
# (menu, [(step id, label)])
MENUS = [
    ("File", [("load", "Load data")]),          # + Load/Save session
    ("Pre-processing", [("calibration", "Calibration"),
                        ("roi", "Define ROI"), ("mask", "Mask")]),
    ("Processing", [("maps", "Maps and Quiver"),
                    ("timeseries", "Time series"),
                    ("profiles", "Profiles")]),
    ("Spectral analysis", [("fft_time", "FFT in time"),
                           ("fft_space", "FFT in space"),
                           ("fft_2d", "2D FFT"), ("f_filter", "f-Filter"),
                           ("k_filter", "k-Filter")]),
    ("Advanced", [("python_script", "Python script")]),
]
STEP_LABEL = {sid: lab for _m, items in MENUS for sid, lab in items}


# --------------------------------------------------------------------------- #
#  Figures and tables (the pages of the centre)
# --------------------------------------------------------------------------- #
def _shrinkable(widget):
    """No minimum width from this widget (a toolbar shows its extension
    arrow, a label is cut): the central area can be made narrow."""
    widget.setMinimumWidth(0)
    widget.setSizePolicy(E("QSizePolicy.Policy.Ignored"),
                         E("QSizePolicy.Policy.Preferred"))


# --------------------------------------------------------------------------- #
#  Sessions: the state of every widget of a panel (tick boxes, drop-down
#  menus, numbers, sliders, text fields), by attribute name
# --------------------------------------------------------------------------- #
def _widget_value(w):
    if isinstance(w, (QtWidgets.QCheckBox, QtWidgets.QRadioButton)):
        return ["check", bool(w.isChecked())]
    if isinstance(w, QtWidgets.QComboBox):
        return ["combo", w.currentText()]
    if isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
        return ["spin", w.value()]
    if isinstance(w, QtWidgets.QSlider):
        return ["slider", int(w.value())]
    if isinstance(w, QtWidgets.QLineEdit):
        return ["text", w.text()]
    return None


def widget_states(obj, exclude=()):
    """{name: [kind, value]} of the widgets that are attributes of a panel
    (and of the dicts of widgets: 'spins.start', 'order_box.fft', ...)."""
    out = {}
    for name, w in list(vars(obj).items()):
        if name.startswith("_") or name in exclude:
            continue
        items = w.items() if isinstance(w, dict) else [(None, w)]
        for k, ww in items:
            if not isinstance(ww, QtWidgets.QWidget):
                continue
            v = _widget_value(ww)
            if v is not None:
                out[name if k is None else "%s.%s" % (name, k)] = v
    return out


def restore_widget_states(obj, states, exclude=()):
    """Set the widgets of a panel back (signals on: the panel reacts as
    if the user had set them). Returns the number set."""
    n = 0
    for key, (kind, value) in (states or {}).items():
        name, _dot, sub = key.partition(".")
        if name in exclude:
            continue
        w = getattr(obj, name, None)
        if isinstance(w, dict):
            w = w.get(sub)
        if not isinstance(w, QtWidgets.QWidget):
            continue
        try:
            if kind == "check" and isinstance(
                    w, (QtWidgets.QCheckBox, QtWidgets.QRadioButton)):
                if w.isChecked() != bool(value):
                    w.setChecked(bool(value))
            elif kind == "combo" and isinstance(w, QtWidgets.QComboBox):
                if w.findText(value) >= 0 and w.currentText() != value:
                    w.setCurrentText(value)
            elif kind == "spin" and isinstance(
                    w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                v = int(value) if isinstance(w, QtWidgets.QSpinBox) \
                    else float(value)
                if w.value() != v:
                    w.setValue(v)
            elif kind == "slider" and isinstance(w, QtWidgets.QSlider):
                if w.value() != int(value):
                    w.setValue(int(value))
            elif kind == "text" and isinstance(w, QtWidgets.QLineEdit):
                if w.text() != value:
                    w.setText(value)
            else:
                continue
            n += 1
        except Exception:
            continue
    return n



class _Page(object):
    """What a figure page and a table page share.

    data_fn()   -> nested dict written by Save data (None: the data drawn
                   on the figure, read back by gui_export.figure_data)
    update_fn() -> redraws the page with the current parameters (Update)
    """

    def _init_page(self, manager, step, name, data_fn, update_fn):
        self.manager, self.step, self.name = manager, step, name
        self.data_fn, self.update_fn = data_fn, update_fn
        self.float_window = None
        self.drawing = False            # a segment is being drawn on it

    @property
    def title(self):
        return "%s -- %s" % (STEP_LABEL.get(self.step, self.step), self.name)


class FigurePage(QtWidgets.QWidget, _Page):
    """One matplotlib figure: canvas + toolbar."""

    def __init__(self, manager, step, name, figure, data_fn=None,
                 update_fn=None):
        QtWidgets.QWidget.__init__(self)
        self._init_page(manager, step, name, data_fn, update_fn)
        self.figure = figure
        self.canvas = FigureCanvasQTAgg(figure)
        # as in the DTS GUI: the figure never shrinks below its own size
        # (75 px per inch); when the area is smaller, scroll bars appear --
        # so the central area itself has no minimum size
        self.canvas.setMinimumSize(int(figure.get_figwidth() * 75),
                                   int(figure.get_figheight() * 75))
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        _shrinkable(self.toolbar)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(E("QFrame.Shape.NoFrame"))
        self.scroll.setWidget(self.canvas)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.scroll, 1)
        self.canvas.setContextMenuPolicy(
            E("Qt.ContextMenuPolicy.CustomContextMenu"))
        self.canvas.customContextMenuRequested.connect(
            lambda pos: self.manager.context_menu(
                self, self.canvas.mapToGlobal(pos)))

    def redraw(self):
        self.canvas.draw_idle()


class TablePage(QtWidgets.QWidget, _Page):
    """A table in the figure area (e.g. the run information)."""

    def __init__(self, manager, step, name, sections, data_fn=None,
                 update_fn=None):
        QtWidgets.QWidget.__init__(self)
        self._init_page(manager, step, name, data_fn, update_fn)
        self.figure = None
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["quantity", "value"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(E(
            "QAbstractItemView.EditTrigger.NoEditTriggers"))
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(True)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.table)
        self.table.setContextMenuPolicy(
            E("Qt.ContextMenuPolicy.CustomContextMenu"))
        self.table.customContextMenuRequested.connect(
            lambda pos: self.manager.context_menu(
                self, self.table.viewport().mapToGlobal(pos)))
        self.set_sections(sections)

    def set_sections(self, sections):
        """sections: [(title, [(quantity, value)])]."""
        self.sections = sections
        n = sum(1 + len(rows) for _t, rows in sections)
        t = self.table
        t.clearSpans()
        t.setRowCount(n)
        r = 0
        for title, rows in sections:
            it = QtWidgets.QTableWidgetItem(title)
            f = it.font(); f.setBold(True); it.setFont(f)
            it.setBackground(QtGui.QColor("#dde7ef"))
            t.setItem(r, 0, it)
            t.setSpan(r, 0, 1, 2)
            r += 1
            for k, v in rows:
                t.setItem(r, 0, QtWidgets.QTableWidgetItem(str(k)))
                t.setItem(r, 1, QtWidgets.QTableWidgetItem(str(v)))
                r += 1
        t.resizeColumnToContents(0)
        t.setColumnWidth(0, min(t.columnWidth(0) + 20, 380))

    def redraw(self):
        pass


class ParamTablePage(QtWidgets.QWidget, _Page):
    """The parameter file used for the loaded run, as an editable table:
    one row per key (value as JSON, its meaning as tooltip and in a third
    column). The Update button at the top writes the edited values to the
    file and loads the run again. Edited cells are highlighted until then."""

    EDITED = QtGui.QColor("#fff2b3")

    def __init__(self, manager, step, name, path, params, on_update):
        QtWidgets.QWidget.__init__(self)
        self._init_page(manager, step, name, None, None)
        self.figure = None
        self.on_update = on_update
        self.update_fn = self.apply
        self.data_fn = lambda: dict(self.params)
        top = QtWidgets.QHBoxLayout()
        self.update_btn = QtWidgets.QPushButton("Update")
        self.update_btn.setMinimumHeight(30)
        self.update_btn.setToolTip("write the values to the parameter file "
                                   "and load the run again")
        f = self.update_btn.font(); f.setBold(True)
        self.update_btn.setFont(f)
        self.path_label = QtWidgets.QLabel()
        self.path_label.setTextInteractionFlags(
            E("Qt.TextInteractionFlag.TextSelectableByMouse"))
        _shrinkable(self.path_label)
        top.addWidget(self.update_btn)
        top.addWidget(self.path_label, 1)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["parameter", "value (JSON)",
                                              "meaning"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setToolTip("double-click a value to edit it (JSON: null, "
                              "numbers, true/false, \"text\", [..]); then "
                              "Update")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 0)
        lay.addLayout(top)
        lay.addWidget(self.table)
        self.table.setContextMenuPolicy(
            E("Qt.ContextMenuPolicy.CustomContextMenu"))
        self.table.customContextMenuRequested.connect(
            lambda pos: self.manager.context_menu(
                self, self.table.viewport().mapToGlobal(pos)))
        self.update_btn.clicked.connect(self.apply)
        self.table.itemChanged.connect(self._edited)
        self.set_params(path, params)

    def set_params(self, path, params):
        self.path, self.params = path, dict(params)
        self.path_label.setText(path)
        self.path_label.setToolTip(path)
        t = self.table
        t.blockSignals(True)
        t.setRowCount(len(params))
        ro = E("Qt.ItemFlag.ItemIsEnabled") | E("Qt.ItemFlag.ItemIsSelectable")
        for r, (k, v) in enumerate(params.items()):
            help_ = lib.PARAM_HELP.get(k, "")
            key = QtWidgets.QTableWidgetItem(k)
            key.setFlags(ro)
            key.setToolTip(help_)
            f = key.font(); f.setBold(True); key.setFont(f)
            val = QtWidgets.QTableWidgetItem(_json_text(v))
            val.setToolTip(help_)
            mean = QtWidgets.QTableWidgetItem(help_)
            mean.setFlags(ro)
            t.setItem(r, 0, key)
            t.setItem(r, 1, val)
            t.setItem(r, 2, mean)
        t.resizeColumnToContents(0)
        t.setColumnWidth(1, max(260, min(t.columnWidth(1), 420)))
        t.blockSignals(False)

    def _edited(self, item):
        if item.column() == 1:
            item.setBackground(self.EDITED)

    def values(self):
        out = {}
        for r in range(self.table.rowCount()):
            k = self.table.item(r, 0).text()
            out[k] = _parse_json_field(k, self.table.item(r, 1).text())
        return out

    def apply(self):
        try:
            vals = self.values()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Parameter file", str(exc))
            return
        self.on_update(self.path, vals)

    def redraw(self):
        pass


def _png_of(rgb):
    """An RGB image (uint8 h x w x 3) as PNG bytes (QImage: usable outside
    the GUI thread)."""
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    raw = rgb.tobytes()
    img = QtGui.QImage(raw, w, h, 3 * w,
                       E("QImage.Format.Format_RGB888")).copy()
    ba = QtCore.QByteArray()
    buf = QtCore.QBuffer(ba)
    io = QtCore.QIODevice
    buf.open(getattr(getattr(io, "OpenModeFlag", io), "WriteOnly"))
    img.save(buf, "PNG")
    buf.close()
    return bytes(ba)


class MoviePage(QtWidgets.QWidget, _Page):
    """The movie just made, played in the figure area: the images of the
    movie (kept in memory as PNG), Play / Pause, first frame, a frame
    slider, the playing speed (fps, the movie's by default) and Loop. It
    plays the images themselves, so it needs no video codec."""

    def __init__(self, manager, step, name, images, info):
        QtWidgets.QWidget.__init__(self)
        self._init_page(manager, step, name, None, None)
        self.figure = None
        self.screen = QtWidgets.QLabel()
        self.screen.setAlignment(E("Qt.AlignmentFlag.AlignCenter"))
        self.screen.setMinimumSize(120, 90)
        self.screen.setSizePolicy(E("QSizePolicy.Policy.Ignored"),
                                  E("QSizePolicy.Policy.Ignored"))
        self.screen.setStyleSheet("background: white;")
        self.screen.setContextMenuPolicy(
            E("Qt.ContextMenuPolicy.CustomContextMenu"))
        self.screen.customContextMenuRequested.connect(
            lambda pos: self.manager.context_menu(
                self, self.screen.mapToGlobal(pos)))
        st = self.style()
        self.first_btn = QtWidgets.QToolButton()
        self.first_btn.setIcon(st.standardIcon(
            E("QStyle.StandardPixmap.SP_MediaSkipBackward")))
        self.first_btn.setToolTip("first frame")
        self.play_btn = QtWidgets.QToolButton()
        self.play_btn.setToolTip("play / pause")
        self.slider = QtWidgets.QSlider(E("Qt.Orientation.Horizontal"))
        self.slider.setToolTip("frame of the movie")
        self.where = QtWidgets.QLabel("")
        self.where.setMinimumWidth(230)
        self.fps = QtWidgets.QSpinBox()
        self.fps.setRange(1, 120)
        self.fps.setSuffix(" fps")
        self.fps.setToolTip("playing speed (the movie file keeps its own "
                            "fps)")
        self.loop = QtWidgets.QCheckBox("Loop")
        self.loop.setChecked(True)
        bar = QtWidgets.QHBoxLayout()
        for w in (self.first_btn, self.play_btn):
            bar.addWidget(w)
        bar.addWidget(self.slider, 1)
        bar.addWidget(self.where)
        bar.addWidget(self.fps)
        bar.addWidget(self.loop)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.screen, 1)
        lay.addLayout(bar)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.first_btn.clicked.connect(lambda: self.slider.setValue(0))
        self.play_btn.clicked.connect(self.toggle)
        self.slider.valueChanged.connect(self.show_frame)
        self.fps.valueChanged.connect(self._speed)
        self._pix = None
        self.set_movie(images, info)

    def set_movie(self, images, info, play=True):
        """images: PNG bytes per frame; info: make_movie's dict (+ 'times':
        the record time of each frame, or None)."""
        self.pause()
        self.images, self.info = list(images), dict(info)
        self.data_fn = lambda: {k: v for k, v in self.info.items()
                                if k != "times"}
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(len(self.images) - 1, 0))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.fps.blockSignals(True)
        self.fps.setValue(int(info.get("fps", 10)))
        self.fps.blockSignals(False)
        self.show_frame(0)
        if play and len(self.images) > 1:
            self.play()

    # ---- playing ---------------------------------------------------------- #
    def playing(self):
        return self.timer.isActive()

    def play(self):
        if not self.images:
            return
        if self.slider.value() >= len(self.images) - 1:
            self.slider.setValue(0)
        self.timer.start(int(round(1000.0 / self.fps.value())))
        self._icon()

    def pause(self):
        self.timer.stop()
        self._icon()

    def toggle(self):
        self.pause() if self.playing() else self.play()

    def _icon(self):
        self.play_btn.setIcon(self.style().standardIcon(E(
            "QStyle.StandardPixmap.SP_MediaPause" if self.playing()
            else "QStyle.StandardPixmap.SP_MediaPlay")))

    def _speed(self, v):
        if self.playing():
            self.timer.start(int(round(1000.0 / v)))

    def _tick(self):
        k = self.slider.value() + 1
        if k >= len(self.images):
            if not self.loop.isChecked():
                self.pause()
                return
            k = 0
        self.slider.setValue(k)

    # ---- drawing ---------------------------------------------------------- #
    def show_frame(self, k):
        if not self.images:
            self._pix = None
            self.screen.clear()
            self.where.setText("")
            return
        k = int(min(max(k, 0), len(self.images) - 1))
        pix = QtGui.QPixmap()
        pix.loadFromData(self.images[k], "PNG")
        self._pix = pix
        self._fit()
        f = self.info.get("first", 0) + k
        txt = "%d / %d   (frame %d" % (k + 1, len(self.images), f)
        times = self.info.get("times")
        if times is not None and k < len(times):
            txt += ", t = %.3f s" % times[k]
        self.where.setText(txt + ")")

    def _fit(self):
        if self._pix is None:
            return
        self.screen.setPixmap(self._pix.scaled(
            self.screen.size(), E("Qt.AspectRatioMode.KeepAspectRatio"),
            E("Qt.TransformationMode.SmoothTransformation")))

    def resizeEvent(self, ev):
        QtWidgets.QWidget.resizeEvent(self, ev)
        self._fit()

    def hideEvent(self, ev):                    # not shown: no playing
        self.pause()
        QtWidgets.QWidget.hideEvent(self, ev)

    def redraw(self):
        self._fit()


class FloatWindow(QtWidgets.QMainWindow):
    """A figure in its own window. The drop-down menus are repeated at its
    top (toolbar); closing the window brings the figure back to the
    centre."""

    def __init__(self, main, page):
        super().__init__(main)
        self.main, self.page = main, page
        self.setWindowFlag(E("Qt.WindowType.Window"), True)
        self.setWindowTitle("%s -- %s" % (APP_NAME, page.title))
        main.add_menu_toolbar(self)
        self.setCentralWidget(page)
        page.show()
        self.resize(950, 760)

    def closeEvent(self, ev):
        if self.page is not None:
            page, self.page = self.page, None
            self.main.figures.dock(page, from_close=True)
        ev.accept()


class FigureManager(object):
    """The centre (tabs) and the list of figures on the right, as in the
    DTS GUI:

    * every figure has a tick box; ticked figures are shown as tabs in the
      centre, unticked ones are hidden (closing a tab unticks it); a step
      heading ticks / unticks all its figures;
    * a click on a name brings the figure forward (ticking it); a double
      click shows it in the centre, bringing it back from its separate
      window if it has one;
    * right click: Focus, Open in separate window, Save data..., Update.
    """

    def __init__(self, main):
        self.main = main
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        self.empty = QtWidgets.QLabel("No figure yet -- File > Load data")
        self.empty.setAlignment(E("Qt.AlignmentFlag.AlignCenter"))
        self.empty.setStyleSheet("color: gray; font-size: 14px;")
        self.empty.setWordWrap(True)
        self.center = QtWidgets.QStackedWidget()
        self.center.addWidget(self.empty)
        self.center.addWidget(self.tabs)
        # no minimum size for the figures (as in the DTS GUI): the area can
        # be narrowed from the left and from the right; a larger figure
        # gets scroll bars
        self.center.setMinimumSize(120, 100)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Figures"])
        self.tree.setToolTip("Ticked figures are shown as tabs. Click a "
                             "name to bring it forward, double-click to "
                             "show it in the centre, right-click for more.")
        self.tree.setContextMenuPolicy(
            E("Qt.ContextMenuPolicy.CustomContextMenu"))
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.itemClicked.connect(self._item_clicked)
        self.tree.itemDoubleClicked.connect(self._item_double)
        self.pages = []                          # in order of creation
        self._items = {}                         # page -> tree item
        self._groups = {}                        # step -> tree item
        self._syncing = False
        self._just_toggled = False

    # ---- creating pages ------------------------------------------------ #
    def add(self, step, name, figure, data_fn=None, update_fn=None):
        """Add a figure, or replace the figure of an existing page (same
        step and name); returns the page. A new page is ticked (shown)."""
        old = self.find(step, name)
        if old is not None:
            old.figure = figure
            old.canvas.figure = figure
            figure.set_canvas(old.canvas)
            old.data_fn, old.update_fn = data_fn, update_fn
            old.canvas.draw_idle()
            return old
        return self._register(FigurePage(self, step, name, figure, data_fn,
                                         update_fn))

    def add_table(self, step, name, sections, data_fn=None, update_fn=None):
        old = self.find(step, name)
        if old is not None:
            old.set_sections(sections)
            old.data_fn, old.update_fn = data_fn, update_fn
            return old
        return self._register(TablePage(self, step, name, sections, data_fn,
                                        update_fn))

    def _register(self, page):
        self._syncing = True
        step = page.step
        grp = self._groups.get(step)
        if grp is None:
            grp = QtWidgets.QTreeWidgetItem([STEP_LABEL.get(step, step)])
            f = grp.font(0); f.setBold(True); grp.setFont(0, f)
            grp.setFlags(E("Qt.ItemFlag.ItemIsEnabled")
                         | E("Qt.ItemFlag.ItemIsUserCheckable")
                         | E("Qt.ItemFlag.ItemIsAutoTristate"))
            self.tree.addTopLevelItem(grp)
            self._groups[step] = grp
        it = QtWidgets.QTreeWidgetItem([page.name])
        it.setFlags(E("Qt.ItemFlag.ItemIsEnabled")
                    | E("Qt.ItemFlag.ItemIsSelectable")
                    | E("Qt.ItemFlag.ItemIsUserCheckable"))
        it.setCheckState(0, E("Qt.CheckState.Checked"))
        grp.addChild(it)
        grp.setExpanded(True)
        self.pages.append(page)
        self._items[page] = it
        self._syncing = False
        self.refresh_tabs()
        return page

    def add_param_table(self, step, name, path, params, on_update):
        old = self.find(step, name)
        if old is not None:
            old.set_params(path, params)
            return old
        return self._register(ParamTablePage(self, step, name, path, params,
                                             on_update))

    def add_movie(self, step, name, images, info):
        """The movie player page (MoviePage), new or with a new movie;
        brought forward."""
        old = self.find(step, name)
        if old is not None:
            old.set_movie(images, info)
            self.focus(old)
            return old
        page = self._register(MoviePage(self, step, name, images, info))
        self.focus(page)
        return page

    def remove(self, page):
        """A page removed (from the list, the tabs or its window)."""
        if page not in self._items:
            return
        if page.float_window is not None:
            win, page.float_window = page.float_window, None
            win.page = None
            win.takeCentralWidget()
            win.close()
        i = self.tabs.indexOf(page)
        if i >= 0:
            self.tabs.removeTab(i)
        it = self._items.pop(page)
        self.pages.remove(page)
        grp = it.parent()
        self._syncing = True
        if grp is not None:
            grp.removeChild(it)
            if grp.childCount() == 0:
                self.tree.takeTopLevelItem(
                    self.tree.indexOfTopLevelItem(grp))
                for k, g in list(self._groups.items()):
                    if g is grp:
                        del self._groups[k]
        self._syncing = False
        page.deleteLater()
        self.refresh_tabs()

    def find(self, step, name):
        for p in self.pages:
            if p.step == step and p.name == name:
                return p
        return None

    def page_of_item(self, item):
        for p, it in self._items.items():
            if it is item:
                return p
        return None

    def is_ticked(self, page):
        return self._items[page].checkState(0) == E("Qt.CheckState.Checked")

    def _tick(self, page, on=True):
        self._syncing = True
        self._items[page].setCheckState(0, E("Qt.CheckState.Checked") if on
                                        else E("Qt.CheckState.Unchecked"))
        self._syncing = False

    # ---- the tabs ------------------------------------------------------- #
    def refresh_tabs(self, current=None):
        """Tabs = the ticked figures not in a separate window, in the
        order of the list."""
        wanted = [p for p in self.pages
                  if self.is_ticked(p) and p.float_window is None]
        keep = current or self.tabs.currentWidget()
        for i in reversed(range(self.tabs.count())):
            if self.tabs.widget(i) not in wanted:
                self.tabs.removeTab(i)
        for pos, p in enumerate(wanted):
            if self.tabs.indexOf(p) != pos:
                if self.tabs.indexOf(p) >= 0:
                    self.tabs.removeTab(self.tabs.indexOf(p))
                self.tabs.insertTab(pos, p, p.name)
                self.tabs.setTabToolTip(pos, p.title)
        if keep is not None and self.tabs.indexOf(keep) >= 0:
            self.tabs.setCurrentWidget(keep)
        self.center.setCurrentWidget(self.tabs if wanted else self.empty)
        if not wanted:
            self.empty.setText("No figure shown -- tick figures in the list "
                               "on the right" if self.pages else
                               "No figure yet -- File > Load data")
        for p in self.pages:
            self._items[p].setText(0, p.name + ("   [window]"
                                                if p.float_window else ""))

    def _close_tab(self, index):
        page = self.tabs.widget(index)
        if page in self._items:
            self._tick(page, False)
            self.refresh_tabs()

    # ---- showing --------------------------------------------------------- #
    def focus(self, page):
        """Tick the figure and bring it forward (its window if it has one,
        else its tab)."""
        self._tick(page, True)
        if page.float_window is not None:
            page.float_window.show()
            page.float_window.raise_()
            page.float_window.activateWindow()
            self.refresh_tabs()
        else:
            self.refresh_tabs(current=page)
        self.tree.setCurrentItem(self._items[page])

    def show_in_centre(self, page):
        """Double click: the figure in the central panel (back from its
        separate window if it has one)."""
        if page.float_window is not None:
            self.dock(page)
        self.focus(page)

    def focus_step(self, step):
        """The first ticked figure of a step comes forward."""
        for s, g in self._groups.items():
            g.setExpanded(True)
        for p in self.pages:
            if p.step == step and self.is_ticked(p):
                self.focus(p)
                return

    # ---- separate windows ----------------------------------------------- #
    def pop_out(self, page):
        if page.float_window is not None:
            self.focus(page)
            return
        self._tick(page, True)
        i = self.tabs.indexOf(page)
        if i >= 0:
            self.tabs.removeTab(i)
        page.float_window = FloatWindow(self.main, page)
        page.float_window.show()
        self.refresh_tabs()

    def dock(self, page, from_close=False):
        win = page.float_window
        if win is None:
            return
        page.float_window = None
        if not from_close:
            win.page = None
            win.takeCentralWidget()
            win.close()
        page.setParent(None)
        self.refresh_tabs(current=page)

    def close_all_windows(self):
        for p in self.pages:
            if p.float_window is not None:
                win, p.float_window = p.float_window, None
                win.page = None
                win.close()

    # ---- Save data / Update ------------------------------------------- #
    last_save_folder = ""

    def save_data(self, page):
        """Save the data of a figure (as the DTS GUI): the user picks the
        file and the format (npz, NetCDF, HDF5, txt, csv, MATLAB)."""
        import gui_export
        try:
            tree = (page.data_fn() if page.data_fn is not None
                    else gui_export.figure_data(page.figure))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self.main, "Save data", str(exc))
            return []
        start = os.path.join(FigureManager.last_save_folder
                             or self.main.last_dir(),
                             gui_export._clean(page.name))
        path, flt = QtWidgets.QFileDialog.getSaveFileName(
            self.main, "Save the data of '%s'" % page.name, start,
            gui_export.file_filter())
        if not path:
            return []
        fmt = gui_export.format_of(path, flt)
        try:
            files = gui_export.write(path, fmt, gui_export.to_tree(tree),
                                     title=page.title)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self.main, "Save data",
                                          "Could not save the data:\n%s" % exc)
            return []
        FigureManager.last_save_folder = os.path.dirname(files[0])
        self.main.status("Data of '%s' saved (%s): %s"
                         % (page.name, fmt, ", ".join(files)), log=True)
        return files

    def update(self, page):
        """Redraw a figure with the current parameters of its step."""
        if page.update_fn is None:
            self.main.status("'%s' has nothing to update" % page.name)
            return
        try:
            page.update_fn()
            self.main.status("'%s' updated" % page.name, log=True)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self.main, "Update", str(exc))

    # ---- menus ---------------------------------------------------------- #
    def context_menu(self, page, global_pos):
        if getattr(page, "drawing", False):      # right click = end of a
            return                               # segment, not the menu
        m = QtWidgets.QMenu(self.main)
        a_focus = m.addAction("Focus")
        if page.float_window is None:
            a_win = m.addAction("Open in separate window")
        else:
            a_win = m.addAction("Show in the central panel")
        m.addSeparator()
        a_save = m.addAction("Save data...")
        a_upd = m.addAction("Update")
        a_upd.setEnabled(page.update_fn is not None)
        chosen = _qexec(m)(global_pos)
        if chosen is a_focus:
            self.focus(page)
        elif chosen is a_win:
            if page.float_window is None:
                self.pop_out(page)
            else:
                self.show_in_centre(page)
        elif chosen is a_save:
            self.save_data(page)
        elif chosen is a_upd:
            self.update(page)

    def _tree_menu(self, pos):
        it = self.tree.itemAt(pos)
        page = self.page_of_item(it) if it is not None else None
        if page is not None:
            self.context_menu(page, self.tree.viewport().mapToGlobal(pos))

    # ---- list events ---------------------------------------------------- #
    def _item_changed(self, it, _col):
        if self._syncing:
            return
        self._just_toggled = True              # this click was on the box
        QtCore.QTimer.singleShot(0, self._clear_toggle)
        page = self.page_of_item(it)
        if page is not None and self.is_ticked(page):
            self.refresh_tabs(current=page)      # just ticked: forward
        else:
            self.refresh_tabs()

    def _clear_toggle(self):
        self._just_toggled = False

    def _item_clicked(self, it, _col):
        toggled, self._just_toggled = self._just_toggled, False
        page = self.page_of_item(it)
        if page is None:                         # a step heading
            for s, g in self._groups.items():
                if g is it and not toggled:
                    self.main.set_step(s)
            return
        if toggled:                              # the box was clicked
            return
        self.focus(page)
        if page.step != self.main.step:
            self.main.set_step(page.step, focus_figures=False)

    def _item_double(self, it, _col):
        page = self.page_of_item(it)
        if page is not None:
            self.show_in_centre(page)


# --------------------------------------------------------------------------- #
#  Background loading (a large .mat takes a while)
# --------------------------------------------------------------------------- #
class TimeBaseDialog(QtWidgets.QDialog):
    """No acquisition log (skipped): the pulse separation and the camera
    fps, entered by hand."""

    def __init__(self, parent, run, pulse_sep, camera_fps):
        super().__init__(parent)
        self.setWindowTitle("Time base of %s" % run)
        lay = QtWidgets.QVBoxLayout(self)
        lab = QtWidgets.QLabel(
            "<b>%s</b> has no acquisition log: enter the time base." % run)
        lab.setWordWrap(True)
        lay.addWidget(lab)
        f = QtWidgets.QFormLayout()
        self.dt = QtWidgets.QLineEdit("%.6g" % pulse_sep)
        self.cam = QtWidgets.QLineEdit("%.6g" % camera_fps)
        self.dt.setToolTip("time between frames A and B of a PIV pair [s]")
        self.cam.setToolTip("camera frame rate [Hz]; PIV fields at camera "
                            "fps / frames per PIV field")
        f.addRow("Pulse separation in s", self.dt)
        f.addRow("Camera FPS in Hz", self.cam)
        lay.addLayout(f)
        bb = QtWidgets.QDialogButtonBox(
            E("QDialogButtonBox.StandardButton.Ok")
            | E("QDialogButtonBox.StandardButton.Cancel"))
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.values = None

    def _ok(self):
        try:
            dt = float(self.dt.text().replace(",", "."))
            fps = float(self.cam.text().replace(",", "."))
            if dt <= 0 or fps <= 0:
                raise ValueError
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "Time base", "Enter two positive numbers.")
            return
        self.values = (dt, fps)
        self.accept()


def _exec(dlg):
    return _qexec(dlg)()


class LoadThread(QtCore.QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, run_dir, params, calibrate=True, velocity=None,
                 time_base=None, piv_path=None, log_path=None):
        super().__init__()
        self.run_dir, self.params = run_dir, params
        self.calibrate, self.velocity = calibrate, velocity
        self.time_base = time_base
        self.piv_path, self.log_path = piv_path, log_path

    def run(self):
        try:
            self.done.emit(lib.load_run(self.run_dir, self.params,
                                        self.calibrate, self.velocity,
                                        self.time_base, self.piv_path,
                                        self.log_path))
        except Exception as exc:
            self.failed.emit("%s\n\n%s" % (exc, traceback.format_exc()))


# --------------------------------------------------------------------------- #
#  Left panels
# --------------------------------------------------------------------------- #
def _json_text(v):
    return json.dumps(v)


def _parse_json_field(key, txt):
    txt = txt.strip()
    try:
        return json.loads(txt)
    except ValueError:
        if txt.lower() == "none":
            return None
        raise ValueError("%s: %r is not a JSON value (use null, numbers, "
                         "true/false, \"text\", [..])" % (key, txt))


class ParamDialog(QtWidgets.QDialog):
    """A separate window with every field of a parameter file.

    New file: all fields start at their default values and the file name
    defaults to param_GUI.json in the root folder. Existing file: the
    file's values (Save overwrites it; Save as... changes the name)."""

    def __init__(self, parent, root, path=None):
        super().__init__(parent)
        self.root = root
        self.saved_path = None
        new = path is None
        self.setWindowTitle("New parameter file" if new else
                            "Parameters -- %s" % os.path.basename(path))
        params = lib.default_params() if new else lib.read_param_file(path)
        lay = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            ("Default values. Edit them, then <b>Save</b>." if new else
             "Values of <b>%s</b>." % os.path.basename(path))
            + "<br>Values are JSON: <tt>null</tt>, numbers, "
              "<tt>true</tt>/<tt>false</tt>, <tt>\"text\"</tt>, lists "
              "<tt>[..]</tt>. Hover a name for its meaning.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        form = QtWidgets.QFormLayout()
        self.name = QtWidgets.QLineEdit(
            lib.PARAM_FILENAME if new else os.path.basename(path))
        self.name.setToolTip("file name, in the root folder (param_GUI*.json)")
        form.addRow("file name", self.name)
        self.edits = {}
        for k, v in params.items():
            e = QtWidgets.QLineEdit(_json_text(v))
            e.setMinimumWidth(320)
            lab = QtWidgets.QLabel(k)
            lab.setToolTip(lib.PARAM_HELP.get(k, ""))
            e.setToolTip(lib.PARAM_HELP.get(k, ""))
            form.addRow(lab, e)
            self.edits[k] = e
        lay.addLayout(form)
        btns = QtWidgets.QDialogButtonBox()
        self.save_btn = btns.addButton(
            "Save", E("QDialogButtonBox.ButtonRole.AcceptRole"))
        btns.addButton(E("QDialogButtonBox.StandardButton.Cancel"))
        reset = btns.addButton("Default values",
                               E("QDialogButtonBox.ButtonRole.ResetRole"))
        btns.accepted.connect(self.save)
        btns.rejected.connect(self.reject)
        reset.clicked.connect(self.reset)
        lay.addWidget(btns)

    def reset(self):
        for k, v in lib.default_params().items():
            if k in self.edits:
                self.edits[k].setText(_json_text(v))

    def values(self):
        return {k: _parse_json_field(k, e.text())
                for k, e in self.edits.items()}

    def save(self):
        name = self.name.text().strip()
        if not name.endswith(".json"):
            name += ".json"
        if not name.startswith("param_GUI"):
            QtWidgets.QMessageBox.warning(
                self, "Parameters", "The file name must start with "
                "'param_GUI' (e.g. param_GUI.json, param_GUI_k6.json) so "
                "the GUI lists it.")
            return
        try:
            vals = self.values()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Parameters", str(exc))
            return
        path = os.path.join(self.root, name)
        self.saved_path = lib.write_param_file(path, vals)
        self.accept()


CREATE_ITEM = "Create a default one..."
BACKGROUNDS = lib.QUANTITIES + ["image"]   # velocity norm, U, V,
                                            # divergence, vorticity,
                                            # kinetic energy
NONE_ITEM = "(no run folder)"


SMOOTH_TIP = ("Gaussian filter of the map (NaN -- mask, rejected vectors "
              "-- left out and kept NaN); sigma in vector spacings (in "
              "image pixels for an image background)")
SHADING_TIP = ("unticked: one flat cell per vector; ticked: gouraud (the "
               "colours interpolated between the vectors) or an image "
               "interpolation (bilinear, bicubic, spline16, lanczos, "
               "hanning, gaussian)")
CMAP_TIP = ("auto: viridis, or blue-white-red centred on 0 for a signed "
            "quantity (U, V, divergence, vorticity)")


def _smooth_spin():
    sp = QtWidgets.QDoubleSpinBox()
    sp.setRange(0.5, 50.0)
    sp.setDecimals(1)
    sp.setSingleStep(0.5)
    sp.setValue(1.0)
    sp.setPrefix("sigma ")
    sp.setSuffix(" vector spacings")
    sp.setToolTip(SMOOTH_TIP)
    return sp


class MapStyleBox(QtWidgets.QGroupBox):
    """Map display (every step that draws a map): a tickable Smoothing (a
    Gaussian filter of the map, sigma in vector spacings), a tickable
    Shading (gouraud or an image interpolation; unticked: one flat cell
    per vector) and the Colour map (auto: viridis, or blue-white-red
    centred on 0 for a signed quantity). changed() when one changes."""

    changed = Signal()

    def __init__(self, title="Map display", parent=None):
        super().__init__(title, parent)
        g = QtWidgets.QGridLayout(self)
        self.smooth_on = QtWidgets.QCheckBox("Smoothing")
        self.smooth_on.setToolTip(SMOOTH_TIP)
        self.smooth = _smooth_spin()
        self.shading_on = QtWidgets.QCheckBox("Shading")
        self.shading_on.setToolTip(SHADING_TIP)
        self.shading = QtWidgets.QComboBox()
        self.shading.addItems(lib.MAP_SHADINGS)
        self.shading.setToolTip(SHADING_TIP)
        self.cmap = QtWidgets.QComboBox()
        self.cmap.addItems(lib.MAP_CMAPS)
        self.cmap.setToolTip(CMAP_TIP)
        g.addWidget(self.smooth_on, 0, 0)
        g.addWidget(self.smooth, 0, 1)
        g.addWidget(self.shading_on, 1, 0)
        g.addWidget(self.shading, 1, 1)
        g.addWidget(QtWidgets.QLabel("Colour map"), 2, 0)
        g.addWidget(self.cmap, 2, 1)
        g.setColumnStretch(1, 1)
        for w in (self.smooth_on, self.shading_on):
            w.toggled.connect(self._enable)
            w.toggled.connect(lambda *_: self.changed.emit())
        self.smooth.valueChanged.connect(
            lambda *_: self.changed.emit() if self.smooth_on.isChecked()
            else None)
        self.shading.currentIndexChanged.connect(
            lambda *_: self.changed.emit() if self.shading_on.isChecked()
            else None)
        self.cmap.currentIndexChanged.connect(lambda *_: self.changed.emit())
        self._enable()

    def _enable(self, *_):
        self.smooth.setEnabled(self.smooth_on.isChecked())
        self.shading.setEnabled(self.shading_on.isChecked())

    def options(self):
        return dict(
            shading=(self.shading.currentText()
                     if self.shading_on.isChecked() else "nearest"),
            smoothing=(self.smooth.value()
                       if self.smooth_on.isChecked() else 0),
            cmap=self.cmap.currentText())

    def get_state(self):
        return dict(smooth_on=self.smooth_on.isChecked(),
                    smoothing=self.smooth.value(),
                    shading_on=self.shading_on.isChecked(),
                    shading=self.shading.currentText(),
                    cmap=self.cmap.currentText())

    def set_state(self, s):
        if not s:
            return
        sh = s.get("shading")
        if sh and self.shading.findText(sh) >= 0:
            self.shading.setCurrentText(sh)
        self.shading_on.setChecked(bool(s.get(
            "shading_on", sh not in (None, "nearest"))))
        if s.get("smoothing"):
            self.smooth.setValue(float(s["smoothing"]))
        self.smooth_on.setChecked(bool(s.get("smooth_on", False)))
        if s.get("cmap") and self.cmap.findText(s["cmap"]) >= 0:
            self.cmap.setCurrentText(s["cmap"])


class RoiCombo(QtWidgets.QComboBox):
    """The ROI drop-down menu of Maps and Quiver and Profiles: Full, then
    the ROIs ticked in Pre-processing > Define ROI (ROI01, ROI02, ...: their
    field '<velocity>_ROIxx'), each with its colour square of Define ROI.
    The data of an item is the field name ('Full' for the full field)."""

    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main
        self._role = E("Qt.ItemDataRole.UserRole")
        self.addItem("Full", "Full")

    def refill(self):
        """The items again (a new run, Define ROI changed); the field shown
        kept when it is still in the list, Full otherwise."""
        d = self.main.data
        keep = self.currentData(self._role) or "Full"
        rp = getattr(self.main, "roi_panel", None)
        shown = rp.shown("rect") if rp is not None else None
        was = self.blockSignals(True)
        self.clear()
        self.addItem("Full", "Full")
        for name, info in (d.roi_fields().items() if d is not None else ()):
            if shown is not None and info["roi"] not in shown:
                continue                         # not ticked in Define ROI
            c = (shown or {}).get(info["roi"]) or \
                lib.roi_colour("rect", info["number"] - 1)
            pm = QtGui.QPixmap(12, 12)
            pm.fill(QtGui.QColor(c))
            p = QtGui.QPainter(pm)
            p.setPen(QtGui.QColor("#555555"))
            p.drawRect(0, 0, 11, 11)
            p.end()
            self.addItem(QtGui.QIcon(pm), "%s  (%s)" % (info["roi"], name),
                         name)
        i = self.findData(keep, self._role)
        self.setCurrentIndex(max(i, 0))
        self.blockSignals(was)
        if i < 0 and keep != "Full" and not was:
            self.currentIndexChanged.emit(0)


class RoiTable(QtWidgets.QTreeWidget):
    """A replica of the ROI table of Pre-processing > Define ROI (ROI, x,
    z; colour square; tick box) for the processing steps.

    mode 'select' (Maps and Quiver, Profiles): a first row *Full*, then the
    ROIs; the row selected is the field shown (an ROI that is ticked and
    applied -- it has a field '<velocity>_ROIxx'); a tick here is the tick
    of Define ROI (and the other way round).
    mode 'tick' (Time series): the ROIs only; the ticks are this table's
    own (which ROIs are active), ticked at first as in Define ROI.
    The combo-box calls used by the panels are kept: currentData(role),
    findData(value, role), setCurrentIndex(i), count(), itemText(i),
    itemData(i, role), currentIndexChanged(int); ticksChanged()."""

    currentIndexChanged = Signal(int)
    ticksChanged = Signal()

    def __init__(self, main, mode="select", parent=None):
        super().__init__(parent)
        self.main, self.mode = main, mode
        self._role = E("Qt.ItemDataRole.UserRole")
        self.setColumnCount(3)
        self.setHeaderLabels(["ROI", "x (m)", "z (m)"])
        self.setRootIsDecorated(False)
        self.setUniformRowHeights(True)
        self._own_ticks = {}                     # mode 'tick': name -> on
        self._filling = False
        self.currentItemChanged.connect(self._current)
        self.itemChanged.connect(self._ticked)

    # ---- the rows --------------------------------------------------------- #
    def refill(self):
        """The rows again from Define ROI's table and the run's ROI
        fields; the selection (field) kept when it is still possible."""
        d = self.main.data
        rp = getattr(self.main, "roi_panel", None)
        keep = self.currentData(self._role) or "Full"
        fields = {i["roi"]: n for n, i in d.roi_fields().items()} \
            if d is not None else {}
        self._filling = True
        self.blockSignals(True)
        self.clear()
        if self.mode == "select":
            full = QtWidgets.QTreeWidgetItem(["Full", "whole field", ""])
            full.setData(0, self._role, "Full")
            self.addTopLevelItem(full)
        src = rp.trees["rect"] if rp is not None else None
        rows = []
        if src is not None and src.topLevelItemCount():
            for i in range(src.topLevelItemCount()):
                it = src.topLevelItem(i)
                rows.append((it.text(0), [it.text(1), it.text(2)],
                             it.icon(0), it.checkState(0)
                             == E("Qt.CheckState.Checked")))
        elif d is not None:                      # Define ROI not filled yet
            for name, info in d.roi_fields().items():
                (x1, z1), (x2, z2) = info["rect"]
                rows.append((info["roi"], [
                    "%.4g .. %.4g" % tuple(sorted((x1, x2))),
                    "%.4g .. %.4g" % tuple(sorted((z1, z2)))],
                    None, True))
        for k, (name, cols, icon, on) in enumerate(rows):
            if self.mode == "tick":
                if not on:
                    continue                     # not active in Define ROI
                on = self._own_ticks.setdefault(name, on)
            it = QtWidgets.QTreeWidgetItem([name] + cols)
            if icon is None or icon.isNull():
                pm = QtGui.QPixmap(12, 12)
                pm.fill(QtGui.QColor(lib.roi_colour("rect", k)))
                icon = QtGui.QIcon(pm)
            it.setIcon(0, icon)
            field = fields.get(name)
            it.setData(0, self._role, field)
            it.setData(1, self._role, name)
            it.setData(2, self._role, lib.roi_colour("rect", k))
            flags = (E("Qt.ItemFlag.ItemIsEnabled")
                     | E("Qt.ItemFlag.ItemIsUserCheckable"))
            if self.mode == "select" and on and field:
                flags |= E("Qt.ItemFlag.ItemIsSelectable")
            it.setFlags(flags)
            it.setCheckState(0, E("Qt.CheckState.Checked") if on
                             else E("Qt.CheckState.Unchecked"))
            if not field:
                it.setForeground(0, QtGui.QBrush(QtGui.QColor("#888888")))
                it.setToolTip(0, "not applied yet (Define ROI > Apply): no "
                                 "ROI field")
            elif self.mode == "select":
                it.setToolTip(0, "%s -- select the row to show this ROI "
                                 "field (ticked ROIs only)" % field)
            self.addTopLevelItem(it)
        for c in range(3):
            self.resizeColumnToContents(c)
        n = max(self.topLevelItemCount(), 2)     # all rows seen (up to 8)
        rh = self.sizeHintForRow(0) if self.topLevelItemCount() else 18
        self.setFixedHeight(min(self.header().sizeHint().height()
                                + min(n, 8) * max(rh, 16) + 6, 220))
        i = self.findData(keep, self._role)
        if i < 0 or not (self.topLevelItem(i).flags()
                         & E("Qt.ItemFlag.ItemIsSelectable")):
            i = 0 if self.mode == "select" else -1
        if i >= 0:
            self.setCurrentItem(self.topLevelItem(i))
        self.blockSignals(False)
        self._filling = False
        if self.mode == "select" and self.currentData(self._role) != keep:
            self.currentIndexChanged.emit(max(i, 0))

    def _current(self, cur, prev):
        if self._filling or cur is None:
            return
        self.currentIndexChanged.emit(self.indexOfTopLevelItem(cur))

    def _ticked(self, it, col):
        if self._filling or col != 0:
            return
        name = it.data(1, self._role)
        on = it.checkState(0) == E("Qt.CheckState.Checked")
        if name is None:
            return
        if self.mode == "tick":
            self._own_ticks[name] = on
            self.ticksChanged.emit()
            return
        # the tick of Define ROI -- after this signal (the tables are
        # filled again when Define ROI's tick changes)
        QtCore.QTimer.singleShot(0, lambda: self._to_source(name, on))

    def _to_source(self, name, on):
        rp = getattr(self.main, "roi_panel", None)
        if rp is not None:
            src = rp.trees["rect"]
            for i in range(src.topLevelItemCount()):
                if src.topLevelItem(i).text(0) == name:
                    src.topLevelItem(i).setCheckState(
                        0, E("Qt.CheckState.Checked") if on
                        else E("Qt.CheckState.Unchecked"))
                    return
        self.refill()

    # ---- combo-box calls -------------------------------------------------- #
    def currentData(self, role=None):
        it = self.currentItem()
        return it.data(0, self._role) if it is not None else None

    def count(self):
        return self.topLevelItemCount()

    def itemText(self, i):
        return self.topLevelItem(i).text(0)

    def itemData(self, i, role=None):
        return self.topLevelItem(i).data(0, self._role)

    def findData(self, value, role=None):
        for i in range(self.topLevelItemCount()):
            if self.topLevelItem(i).data(0, self._role) == value \
                    and value is not None:
                return i
        return -1

    def setCurrentIndex(self, i):
        if 0 <= i < self.topLevelItemCount() and (
                self.topLevelItem(i).flags()
                & E("Qt.ItemFlag.ItemIsSelectable")):
            self.setCurrentItem(self.topLevelItem(i))

    def currentIndex(self):
        it = self.currentItem()
        return self.indexOfTopLevelItem(it) if it is not None else -1

    def ticked_fields(self):
        """[(field, ROI name, colour)] of the rows ticked with a field."""
        out = []
        for i in range(self.topLevelItemCount()):
            it = self.topLevelItem(i)
            f = it.data(0, self._role)
            if f in (None, "Full"):
                continue
            if it.checkState(0) == E("Qt.CheckState.Checked"):
                out.append((f, it.text(0), it.data(2, self._role)))
        return out


class SetupFileRow(object):
    """Load data: a tick box, a file selector (the files of this kind found
    in the run's PostProcessing folder, the run folder and the root folder,
    plus Browse...) and Browse... -- for the calibration, ROI and mask
    files saved by their steps."""

    def __init__(self, panel, label, ext, tip):
        self.panel, self.ext, self.label = panel, ext, label
        self._role = E("Qt.ItemDataRole.UserRole")
        self._extra = []
        self.check = QtWidgets.QCheckBox(label)
        self.check.setToolTip(tip + " Off: no file is loaded.")
        self.combo = QtWidgets.QComboBox()
        self.combo.setSizeAdjustPolicy(E(
            "QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon"))
        self.combo.setToolTip("*%s files of the root folder (where Save "
                              "proposes them), of the run's PostProcessing "
                              "folder and of the run folder; Browse... for "
                              "any other" % ext)
        self.browse_btn = QtWidgets.QPushButton("Browse...")
        self.browse_btn.setMaximumWidth(90)
        self.browse_btn.clicked.connect(self.browse)
        self.check.toggled.connect(lambda on: self._enable())
        self._enable()

    def add_to(self, grid, row):
        grid.addWidget(self.check, row, 0)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.combo, 1)
        h.addWidget(self.browse_btn)
        grid.addLayout(h, row, 1, 1, 2)

    def _enable(self):
        on = self.check.isChecked()
        self.combo.setEnabled(on)
        self.browse_btn.setEnabled(on)

    def found(self):
        lp = self.panel
        dirs = []
        if lp.root_dir:                          # the default folder
            dirs.append(lp.root_dir)
        if lp.run_dir:
            dirs += [os.path.join(lp.run_dir, lib.POSTPROC_DIR), lp.run_dir]
        out = []
        for d in dirs:
            try:
                names = sorted(f for f in os.listdir(d)
                               if f.endswith(self.ext)
                               and not f.startswith("."))
            except OSError:
                continue
            out += [os.path.join(d, f) for f in names]
        return out

    def fill(self, select=None):
        keep = select or self.combo.currentData(self._role)
        found = self.found()
        paths = found + [p for p in self._extra if p not in found]
        lp = self.panel
        cb = self.combo
        cb.blockSignals(True)
        cb.clear()
        for p in paths:
            where = os.path.dirname(p)
            if lp.run_dir and where == os.path.join(lp.run_dir,
                                                    lib.POSTPROC_DIR):
                where = "run/PostProcessing"
            elif lp.run_dir and where == lp.run_dir:
                where = "run"
            elif lp.root_dir and where == lp.root_dir:
                where = "root"
            cb.addItem("%s   (%s)" % (os.path.basename(p), where), p)
            cb.setItemData(cb.count() - 1, p, E("Qt.ItemDataRole.ToolTipRole"))
        if not paths:
            cb.addItem("(no %s file found -- Browse...)" % self.ext, None)
        if keep in paths:
            cb.setCurrentIndex(paths.index(keep))
        cb.blockSignals(False)

    def browse(self):
        lp = self.panel
        start = (os.path.dirname(self.path or "") or lp.root_dir
                 or lp.run_dir or os.path.expanduser("~"))
        path, _f = QtWidgets.QFileDialog.getOpenFileName(
            lp, self.label, start, json_filter("%s file" % self.label,
                                               self.ext))
        if path:
            self.select(path)

    def select(self, path):
        if path and path not in self._extra:
            self._extra.append(path)
        self.fill(select=path)

    @property
    def path(self):
        """The file selected (None when none)."""
        return self.combo.currentData(self._role)

    @property
    def active(self):
        """The file to load: ticked and selected (None otherwise)."""
        p = self.path
        return p if self.check.isChecked() and p else None

    def get_state(self):
        return dict(on=self.check.isChecked(), path=self.path)

    def set_state(self, st):
        st = st or {}
        if st.get("path"):
            self.select(st["path"])
        self.check.setChecked(bool(st.get("on")))


def read_setup_file(path, kind):
    """A calibration / ROI / mask file (JSON) read; ValueError when it is
    not one of that kind."""
    with open(path) as fh:
        rec = json.load(fh)
    if not isinstance(rec, dict) or (rec.get("kind") and
                                     kind not in str(rec.get("kind"))):
        raise ValueError("not a %s file" % kind)
    return rec


class LoadPanel(QtWidgets.QWidget):
    """File > Load data (single run): root folder, run, the calibration /
    ROI / mask files to load with it (no parameter file), display
    options."""

    def __init__(self, main):
        super().__init__()
        self.main = main
        lay = QtWidgets.QVBoxLayout(self)
        role = E("Qt.ItemDataRole.UserRole")
        self._role = role

        # --- root folder ------------------------------------------------ #
        box = QtWidgets.QGroupBox("Root folder")
        bl = QtWidgets.QHBoxLayout(box)
        self.root = QtWidgets.QLineEdit()
        self.root.setPlaceholderText("folder holding the run folders")
        browse = QtWidgets.QPushButton("Browse...")
        bl.addWidget(self.root, 1)
        bl.addWidget(browse)
        lay.addWidget(box)
        browse.clicked.connect(self.browse_root)
        self.root.editingFinished.connect(self.refresh)

        # --- run + parameters ------------------------------------------- #
        box = QtWidgets.QGroupBox("Run")
        g = QtWidgets.QGridLayout(box)
        self.run_combo = QtWidgets.QComboBox()
        self.run_combo.setSizeAdjustPolicy(E(
            "QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon"))
        g.addWidget(QtWidgets.QLabel("run folder"), 0, 0)
        g.addWidget(self.run_combo, 0, 1, 1, 2)
        self.cal_file = SetupFileRow(
            self, "Calibration", CAL_EXT,
            "On: load this calibration file (saved by Pre-processing > "
            "Calibration > Save) and run the Calibration step from it: "
            "image, segment, length in mm, time base, then Apply (the run "
            "in m, m/s, s).")
        self.roi_file = SetupFileRow(
            self, "ROI", ROI_EXT,
            "On: load this ROI file (saved by Pre-processing > Define ROI > "
            "Save) and run the Define ROI step from it: the ROIs and lines, "
            "their ticks, then Apply (the ROI quantities of the active "
            "ROIs). Needs the calibration (the ROIs are in m).")
        self.mask_file = SetupFileRow(
            self, "Mask", MASK_EXT,
            "On: load this mask file (saved by Pre-processing > Mask > "
            "Save) and run the Mask step from it: the masks, the selection "
            "areas, the masks shown and the mask applied.")
        for k, row in enumerate((self.cal_file, self.roi_file,
                                 self.mask_file)):
            row.add_to(g, 1 + k)
        self.velocity = QtWidgets.QComboBox()
        self.velocity.addItems([lib.VALIDATED, lib.UNVALIDATED,
                                lib.REJECTED_NAN])
        self.velocity.setToolTip(
            "the velocity to load: Validated (PIVlab filtered velocity), "
            "Unvalidated (original velocity), or Unvalidated with the "
            "vectors PIVlab rejected set to NaN (default: Validated). Only "
            "this velocity is read and kept in memory; another choice is "
            "read at the next Load")
        g.addWidget(QtWidgets.QLabel("velocity"), 4, 0)
        g.addWidget(self.velocity, 4, 1, 1, 2)
        g.setColumnStretch(1, 1)
        lay.addWidget(box)
        self.load_btn = QtWidgets.QPushButton("Load")
        self.load_btn.setDefault(True)
        self.load_btn.setMinimumHeight(34)
        self.load_btn.setToolTip("read the run's PIV file and acquisition "
                                 "log; show the run information and the "
                                 "parameters (the maps are in Processing > "
                                 "Maps and Quiver)")
        lay.addWidget(self.load_btn)
        self.run_combo.currentIndexChanged.connect(
            lambda *_: self._fill_files())
        self.load_btn.clicked.connect(self._user_load)

        # --- Display ---------------------------------------------------- #
        box = QtWidgets.QGroupBox("Display")
        g = QtWidgets.QGridLayout(box)
        g.setColumnStretch(1, 1)
        indent = 18                                   # sub-options
        r = 0
        # Velocity shown: calibrated (m, m/s, s) or uncalibrated (px,
        # px/frame, frames); calibrated is listed only when XSCALE / YSCALE
        # are set
        self.units = QtWidgets.QComboBox()
        self.units.addItem(UNCALIBRATED)
        self.units.setToolTip(
            "velocity shown: calibrated (m, m/s, s) or uncalibrated (px, "
            "px/frame, frame numbers). 'calibrated' is listed only when the "
            "parameter file has XSCALE and YSCALE; the Calibration box of "
            "the Run sets the choice at Load. The run is kept once in px "
            "and px/frame, so switching costs no memory")
        g.addWidget(QtWidgets.QLabel("Velocity"), r, 0)
        g.addWidget(self.units, r, 1, 1, 2); r += 1
        # Frame: single frame (slider) / time mean / rms
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["single frame", "time mean", "rms"])
        self.frame = QtWidgets.QSlider(E("Qt.Orientation.Horizontal"))
        self.frame_spin = QtWidgets.QSpinBox()
        g.addWidget(QtWidgets.QLabel("Frame"), r, 0)
        g.addWidget(self.mode, r, 1, 1, 2); r += 1
        g.addWidget(self.frame, r, 1)
        g.addWidget(self.frame_spin, r, 2); r += 1
        self.frame_info = QtWidgets.QLabel("")
        self.frame_info.setToolTip("frame number and, when the run is "
                                   "calibrated, its time = frame x time "
                                   "between PIV fields (frames per field / "
                                   "camera fps)")
        g.addWidget(self.frame_info, r, 1, 1, 2); r += 1
        # ROI: Full + the ROIs ticked in Define ROI
        self.roi = RoiCombo(main)
        self.roi.setToolTip(
            "Full field, or an ROI ticked in Define ROI (ROI01, ROI02, ...): "
            "its field -- the velocity extracted in the ROI, "
            "'<velocity>_ROIxx' -- shown on its own grid, with its "
            "quantities of Define ROI > Apply")
        g.addWidget(QtWidgets.QLabel("ROI"), r, 0)
        g.addWidget(self.roi, r, 1, 1, 2); r += 1
        # Mask: the mask applied (velocity NaN in it for every step)
        self.mask_combo = QtWidgets.QComboBox()
        self.mask_combo.addItem("None", None)
        self.mask_combo.setToolTip(
            "the mask applied: the velocity in it is NaN for every step "
            "(the display, the ROI fields, the later steps). The masks are "
            "built in Pre-processing > Mask")
        g.addWidget(QtWidgets.QLabel("Mask"), r, 0)
        g.addWidget(self.mask_combo, r, 1, 1, 2); r += 1
        # Quiver (tick) / vector field / colour + every n vectors
        self.quiver = QtWidgets.QCheckBox("Quiver")
        self.quiver.setChecked(True)
        f = self.quiver.font(); f.setBold(True); self.quiver.setFont(f)
        g.addWidget(self.quiver, r, 0, 1, 3); r += 1
        self.qcolour = QtWidgets.QComboBox()
        self.qcolour.addItems(lib.QUIVER_COLOURS)
        self.qcolour.setCurrentText("black")
        self.qcolour.setToolTip("arrow colour: |u| (colour map below) or "
                                "one colour. Set by default when the "
                                "background changes: black without "
                                "background or on the velocity norm, white "
                                "on an image; any colour can then be "
                                "chosen")
        self.skip = QtWidgets.QSpinBox()
        self.skip.setRange(1, 50)
        self.skip.setValue(1)
        self.skip.setPrefix("every ")
        self.skip.setSuffix(" vector(s)")
        qlab = QtWidgets.QLabel("Quiver")
        qlab.setContentsMargins(indent, 0, 0, 0)
        g.addWidget(qlab, r, 0)
        g.addWidget(self.qcolour, r, 1)
        g.addWidget(self.skip, r, 2); r += 1
        self.vscale = QtWidgets.QDoubleSpinBox()
        self.vscale.setRange(0.05, 50.0)
        self.vscale.setDecimals(2)
        self.vscale.setSingleStep(0.25)
        self.vscale.setValue(1.0)
        self.vscale.setPrefix("x ")
        self.vscale.setToolTip(
            "length of the vectors on the figure: the 99th-percentile "
            "speed is drawn this many vector spacings long (a key at the "
            "top right gives that speed)")
        vlab = QtWidgets.QLabel("Vector scale")
        vlab.setContentsMargins(indent, 0, 0, 0)
        g.addWidget(vlab, r, 0)
        g.addWidget(self.vscale, r, 1, 1, 2); r += 1
        self._quiver_widgets = (qlab, self.qcolour, self.skip, vlab,
                                self.vscale)
        # Background (tick) / velocity norm or image (+ image selector)
        self.bkg_on = QtWidgets.QCheckBox("Background")
        self.bkg_on.setFont(f)
        g.addWidget(self.bkg_on, r, 0, 1, 3); r += 1
        self.bkg = QtWidgets.QComboBox()
        self.bkg.addItems(BACKGROUNDS)
        self.bkg.setToolTip(
            "a colour map of the velocity norm |u|, the divergence du/dx + "
            "dv/dz, the vorticity dv/dx - du/dz (diverging maps, centred on "
            "0) or the kinetic energy (u^2 + v^2)/2 -- computed on the ROI "
            "field selected, with the mask selected applied; or a camera "
            "image (grey) under the vectors")
        g.addWidget(self.bkg, r, 1, 1, 2); r += 1
        self.image_combo = QtWidgets.QComboBox()
        self.image_combo.setToolTip(
            "background image: image files (tif, png, jpg, bmp) and "
            "background*.mat of the run folder and of the root folder; "
            "Browse... for any other file")
        self.image_combo.setSizeAdjustPolicy(E(
            "QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon"))
        self.image_browse = QtWidgets.QPushButton("Browse...")
        self.image_browse.setMaximumWidth(90)
        self.image_label = QtWidgets.QLabel("image")
        self.image_label.setContentsMargins(indent, 0, 0, 0)
        g.addWidget(self.image_label, r, 0)
        ib = QtWidgets.QHBoxLayout()
        ib.addWidget(self.image_combo, 1)
        ib.addWidget(self.image_browse)
        g.addLayout(ib, r, 1, 1, 2); r += 1
        # map display: tickable Smoothing (Gaussian filter) and Shading,
        # colour map (auto: viridis for |u| and positive maps,
        # blue-white-red for the signed ones)
        self.smooth_on = QtWidgets.QCheckBox("Smoothing")
        self.smooth_on.setToolTip(SMOOTH_TIP)
        self.smooth_label = self.smooth_on
        self.smooth = _smooth_spin()
        g.addWidget(self.smooth_on, r, 0)
        g.addWidget(self.smooth, r, 1, 1, 2); r += 1
        self.shading_on = QtWidgets.QCheckBox("Shading")
        self.shading_on.setToolTip(SHADING_TIP)
        self.shading = QtWidgets.QComboBox()
        self.shading.addItems(lib.MAP_SHADINGS)
        self.shading.setToolTip(SHADING_TIP)
        g.addWidget(self.shading_on, r, 0)
        g.addWidget(self.shading, r, 1, 1, 2); r += 1
        self.cmap = QtWidgets.QComboBox()
        self.cmap.addItems(lib.MAP_CMAPS)
        self.cmap.setToolTip(
            "colour map of the background map and of the arrows coloured "
            "by |u|; auto: viridis, or blue-white-red centred on 0 for U, "
            "V, divergence, vorticity")
        g.addWidget(QtWidgets.QLabel("Colour map"), r, 0)
        g.addWidget(self.cmap, r, 1, 1, 2); r += 1
        # the Display box is the panel of Processing > Maps and Quiver
        # (MapsPanel); Load data only loads
        lay.addStretch(1)
        self.display_box = box
        box.setEnabled(False)
        self.frame.valueChanged.connect(self.frame_spin.setValue)
        self.frame.valueChanged.connect(lambda *_: self.update_frame_info())
        self.mode.currentIndexChanged.connect(
            lambda *_: self.update_frame_info())
        self.frame_spin.valueChanged.connect(self.frame.setValue)
        self.frame.sliderReleased.connect(self.main.draw_field)
        self.frame.valueChanged.connect(
            lambda _v: (None if self.frame.isSliderDown()
                        else self.main.draw_field()))
        redraw = lambda *_: self.main.draw_field()
        for w in (self.mode, self.cmap, self.roi, self.qcolour):
            w.currentIndexChanged.connect(redraw)
        self.shading.currentIndexChanged.connect(
            lambda *_: redraw() if self.shading_on.isChecked() else None)
        self.velocity.currentIndexChanged.connect(
            lambda *_: self._velocity_changed())
        self.units.currentIndexChanged.connect(
            lambda *_: self._units_changed())
        self.mask_combo.currentIndexChanged.connect(
            lambda *_: self._mask_selected())
        self.skip.valueChanged.connect(redraw)
        self.vscale.valueChanged.connect(redraw)
        self.smooth.valueChanged.connect(
            lambda *_: redraw() if self.smooth_on.isChecked() else None)
        self.smooth_on.toggled.connect(lambda *_: (
            self._enable_background_rows(), redraw()))
        self.shading_on.toggled.connect(lambda *_: (
            self.shading.setEnabled(self.shading_on.isChecked()), redraw()))
        self.shading.setEnabled(False)
        self.quiver.toggled.connect(lambda on: (self._enable_quiver(on),
                                                redraw()))
        self.bkg_on.toggled.connect(lambda *_: self._bkg_changed())
        self.bkg.currentIndexChanged.connect(lambda *_: self._bkg_changed())
        self.image_combo.activated.connect(redraw)
        self.image_browse.clicked.connect(self.browse_image)
        self.mode.currentIndexChanged.connect(
            lambda i: [w.setEnabled(i == 0) for w in (self.frame,
                                                      self.frame_spin)])
        self._extra_images = []
        self._enable_background_rows()
        self._pending_frame = None

    # ----------------------------------------------------------------- #
    @property
    def root_dir(self):
        r = self.root.text().strip()
        return r if r and os.path.isdir(r) else None

    # no parameter file any more (the calibration, ROI and mask files of
    # their steps instead): kept for the callers
    param_file = None

    def setup_rows(self):
        return (("calibration", self.cal_file), ("roi", self.roi_file),
                ("mask", self.mask_file))

    def _fill_files(self):
        for _k, row in self.setup_rows():
            row.fill()

    @property
    def run_dir(self):
        name = self.run_combo.currentData(self._role)
        return os.path.join(self.root_dir, name) if (name and self.root_dir) \
            else None

    def browse_root(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Root folder", self.root.text() or os.path.expanduser("~"))
        if d:
            self.root.setText(d)
            self.refresh()

    def refresh(self, select_run=None, select_param=None):
        """Re-read the root folder: the run folders, the calibration / ROI
        / mask files."""
        root = self.root_dir
        self._fill_runs(select_run)
        self._fill_files()
        self.load_btn.setEnabled(bool(self.run_dir)
                                 and self.main._thread is None)
        if root is None and self.root.text().strip():
            self.main.status("Not a folder: %s" % self.root.text().strip())
        elif root:
            msg = ("%d run folder(s) in %s"
                   % (self.run_combo.count()
                      if self.run_combo.currentData(self._role) else 0,
                      root))
            # in the log only when it changes (refresh runs often)
            self.main.status(msg, log=msg != getattr(self, "_last_scan", None))
            self._last_scan = msg

    def _piv_filename(self):
        return lib.DEFAULT_PARAMS["PIV_FILENAME"]

    def _fill_runs(self, select=None):
        cb = self.run_combo
        keep = select or cb.currentData(self._role)
        cb.clear()
        if not self.root_dir:
            return
        runs = lib.list_runs(self.root_dir, self._piv_filename())
        if not runs:
            cb.addItem(NONE_ITEM, None)
            return
        model = cb.model()
        for i, (name, has) in enumerate(runs):
            # a run without the PIV file stays selectable: Load asks for
            # the .mat (or uses the one recorded in run_information.json)
            cb.addItem(name if has is not False else
                       "%s   (no %s)" % (name, self._piv_filename()), name)
        names = [n for n, _h in runs]
        usable = [n for n, h in runs if h is not False] or names
        if keep in usable:
            cb.setCurrentIndex(names.index(keep))
        elif usable:
            cb.setCurrentIndex(names.index(usable[0]))

    def load(self):
        """Load: the run with the default parameters, plus -- each when
        ticked -- the calibration file (XSCALE / YSCALE, rotation, time
        base), the ROI file (PTS_ROI, LINES) and the mask file (masks, mask
        applied, selection areas); once the run is read, the three steps
        are run again from their files (MainWindow._apply_setup_files)."""
        rd = self.run_dir
        if not rd:
            self.main.status("Choose a run folder first")
            return
        P = lib.default_params()
        pending = {}
        for kind, row in self.setup_rows():
            path = row.active
            if row.check.isChecked() and not path:
                self.main.write_log("WARNING: %s ticked but no file "
                                    "selected: not loaded" % row.label)
            if not path:
                continue
            try:
                rec = read_setup_file(path, kind)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self, "Load data", "%s file %s: %s" % (row.label, path,
                                                           exc))
                return
            pending[kind] = (path, rec)
        if "calibration" in pending:
            rec = pending["calibration"][1]
            xs, ys = rec.get("XSCALE") or rec.get("applied_m_per_px"), \
                rec.get("YSCALE") or rec.get("applied_m_per_px")
            if xs:
                P.update(XSCALE=float(xs), YSCALE=float(ys or xs))
            if rec.get("rotate_deg") is not None:
                P["ROTATE"] = int(rec["rotate_deg"])
            P["CALIB_INFO"] = CalibrationPanel.calib_info_of(rec)
            self.main.calib_panel.time_base_from(rec, log_ok=None)
        if "roi" in pending:
            rec = pending["roi"][1]
            P.update(PTS_ROI=rec.get("PTS_ROI"), LINES=rec.get("LINES"))
        if "mask" in pending:
            rec = pending["mask"][1]
            P.update(MASKS=rec.get("MASKS"),
                     MASK_APPLIED=rec.get("MASK_APPLIED"),
                     MASK_SELECTION=rec.get("MASK_SELECTION"))
        self.main._pending_setup = pending
        self.main.setup_files = {k: p for k, (p, _r) in pending.items()}
        self.main.write_log("Load data: %s" % ("; ".join(
            "%s file %s" % (k, p) for k, (p, _r) in pending.items())
            or "no calibration, ROI or mask file"))
        self.main.load_run(rd, P, None, calibrate=lib.is_calibrated(P))

    def _velocity_changed(self):
        """Another velocity chosen once a run is loaded: only the velocity
        loaded is in memory, so it is read at the next Load."""
        d = self.main.data
        name = self.velocity.currentText()
        if d is None or name == d.default:
            return
        if name not in d.available:
            self.main.status("%s is not in %s" % (name, os.path.basename(
                d.piv_path or "")), log=True)
            return
        self.main.status("%s chosen: press Load to read it (%s is the "
                         "velocity in memory)" % (name, d.default), 0)

    def _units_changed(self):
        """Display > Velocity: the run shown calibrated or uncalibrated
        (no reading, no copy: the calibration is applied when drawing)."""
        d = self.main.data
        if d is None:
            return
        want = self.units.currentText() == CALIBRATED
        if want == d.calibrated:
            return
        d.set_calibrated(want)
        self._fill_rois(d)
        self.update_frame_info()
        self.main.write_log("Display: %s (%s, %s, %s)" % (
            self.units.currentText(), d.lunit, d.vunit, d.tunit))
        self.main.show_info(d)
        self.main.calib_panel.refresh()
        self.main.draw_field()
        self.main.profiles_panel.sync()
        self.main.timeseries_panel.sync()
        self.main.fft_time_panel.sync()
        self.main.fft_space_panel.sync()

    def fill_masks(self, d):
        """Mask list: None + the masks of the run (Pre-processing > Mask);
        the one applied is selected."""
        cb = self.mask_combo
        cb.blockSignals(True)
        cb.clear()
        cb.addItem("None", None)
        for name, m in (d.masks.items() if d is not None else []):
            pm = QtGui.QPixmap(12, 12)
            pm.fill(QtGui.QColor(m["colour"]))
            n = int(np.sum(m["mask"])) if m["mask"] is not None else 0
            cb.addItem(QtGui.QIcon(pm), "%s  (%d vectors)" % (name, n), name)
        i = cb.findData(d.mask_name if d is not None else None, self._role)
        cb.setCurrentIndex(max(i, 0))
        cb.blockSignals(False)
        pp = getattr(self.main, "profiles_panel", None)
        if pp is not None:                       # Profiles shows the same
            pp._mirror(pp.mask_combo, cb)

    def _mask_selected(self):
        """Display > Mask: the mask applied to the loaded run (velocity
        NaN in it for every step), saved in the run information."""
        d = self.main.data
        name = self.mask_combo.currentData(self._role)
        if d is None or name == d.mask_name:
            return
        d.apply_mask(name)
        self.main.status("Mask applied: %s" % (
            "%s, %d vectors -> velocity NaN" % (name, int(d.mask.sum()))
            if d.mask is not None else "none"), log=True)
        self.main.show_info(d)
        self.main.draw_field()
        self.main.mask_panel._state()
        self.main.mask_panel.draw(keep_view=True)
        self.main.profiles_panel.sync()
        self.main.timeseries_panel.sync()
        self.main.fft_time_panel.sync()
        self.main.fft_space_panel.sync()

    def _fill_rois(self, d):
        """ROI list: Full + one field per ROI of the parameter file ticked
        in Pre-processing > Define ROI -- the velocity extracted in the
        ROI, '<velocity>_ROIxx' (ROIs are in m: none when the run cannot be
        calibrated); each with its colour of Define ROI."""
        cb = self.roi
        keep = getattr(self, "_pending_roi", None)
        cb.blockSignals(True)
        cb.refill()
        if keep:
            i = cb.findData(keep, self._role)
            if i < 0:                            # same ROI, other velocity
                i = next((k for k in range(cb.count()) if str(
                    cb.itemData(k, self._role))[-6:] == str(keep)[-6:]
                    and "_ROI" in str(keep)), -1)
            if i >= 0:
                cb.setCurrentIndex(i)
        cb.blockSignals(False)
        self._pending_roi = None

    # ---- background image ------------------------------------------- #
    def _enable_quiver(self, on):
        for w in self._quiver_widgets:
            w.setEnabled(on)

    def _enable_background_rows(self):
        on = self.bkg_on.isChecked()
        img = self.bkg.currentText() == "image"
        self.bkg.setEnabled(on)
        self.smooth_on.setEnabled(on)
        self.smooth.setEnabled(on and self.smooth_on.isChecked())
        self.smooth.setSuffix(" px" if img else " vector spacings")
        for w in (self.image_label, self.image_combo, self.image_browse):
            w.setVisible(img)
            w.setEnabled(on)

    def default_quiver_colour(self):
        """Black without a background or on the velocity norm, white on an
        image."""
        img = self.bkg_on.isChecked() and self.bkg.currentText() == "image"
        return "white" if img else "black"

    def _bkg_changed(self):
        self._enable_background_rows()
        self.qcolour.blockSignals(True)          # redrawn once, below
        self.qcolour.setCurrentText(self.default_quiver_colour())
        self.qcolour.blockSignals(False)
        if self.bkg_on.isChecked() and self.bkg.currentText() == "image" \
                and self.image_combo.currentData(self._role) is None:
            self.fill_images()
            if self.image_combo.currentData(self._role) is None:
                self.main.status("No image in the run or root folder: "
                                 "Browse... for one", log=True)
        self.main.draw_field()

    def fill_images(self, select=None):
        """The candidate background images: run folder, root folder, then
        any file chosen with Browse..."""
        keep = select or self.image_combo.currentData(self._role)
        found = lib.find_background_images(self.run_dir, self.root_dir)
        paths = found + [p for p in self._extra_images if p not in found]
        cb = self.image_combo
        cb.blockSignals(True)
        cb.clear()
        for p in paths:
            where = ("run" if self.run_dir and os.path.dirname(p)
                     == self.run_dir else "root" if os.path.dirname(p)
                     == self.root_dir else os.path.dirname(p))
            cb.addItem("%s   (%s)" % (os.path.basename(p), where), p)
            cb.setItemData(cb.count() - 1, p,
                           E("Qt.ItemDataRole.ToolTipRole"))
        if not paths:
            cb.addItem("(no image found -- Browse...)", None)
        if keep in paths:
            cb.setCurrentIndex(paths.index(keep))
        cb.blockSignals(False)

    def browse_image(self):
        start = self.run_dir or self.root_dir or os.path.expanduser("~")
        path, _f = QtWidgets.QFileDialog.getOpenFileName(
            self, "Background image", start,
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.mat);;"
            "All files (*)")
        if path:
            if path not in self._extra_images:
                self._extra_images.append(path)
            self.fill_images(select=path)
            self.main.draw_field()

    def update_frame_info(self):
        d = self.main.data
        if d is None:
            self.frame_info.setText("")
        elif self.mode.currentIndex() != 0:
            self.frame_info.setText("all %d frames%s" % (
                d.nt, "  (%.4g s)" % (d.nt * d.dt_field) if d.calibrated
                else ""))
        else:
            self.frame_info.setText(d.frame_text(
                min(self.frame.value(), d.nt - 1)))

    def data_loaded(self, d):
        self.frame.setRange(0, d.nt - 1)
        self.frame_spin.setRange(0, d.nt - 1)
        self.display_box.setEnabled(True)
        self.fill_images()
        # the velocities not in the file are greyed out; the one loaded
        # is shown
        model = self.velocity.model()
        for i in range(self.velocity.count()):
            it = model.item(i)
            if it is not None:
                it.setEnabled(self.velocity.itemText(i) in d.available)
        self.velocity.blockSignals(True)
        self.velocity.setCurrentText(d.default)
        self.velocity.blockSignals(False)
        # Display > Velocity: calibrated only when XSCALE / YSCALE are set
        units = [CALIBRATED, UNCALIBRATED] if d.can_calibrate \
            else [UNCALIBRATED]
        pend = getattr(self, "_pending_units", None)
        if pend in units:                        # from a session
            d.set_calibrated(pend == CALIBRATED)
        self._pending_units = None
        self.units.blockSignals(True)
        self.units.clear()
        self.units.addItems(units)
        self.units.setCurrentText(CALIBRATED if d.calibrated
                                  else UNCALIBRATED)
        self.units.blockSignals(False)
        self._fill_rois(d)
        pend = getattr(self, "_pending_mask", None)
        if pend and pend in d.masks and pend != d.mask_name:
            d.apply_mask(pend)                   # from a session
        self._pending_mask = None
        self.fill_masks(d)
        self.update_frame_info()

    def display_options(self):
        frame = {1: "mean", 2: "rms"}.get(self.mode.currentIndex(),
                                         self.frame.value())
        d = self.main.data                       # the velocity in memory
        return dict(frame=frame, field=d.default if d is not None else None,
                    roi=self.roi.currentData(self._role) or "Full",
                    quiver=self.quiver.isChecked(),
                    quiver_colour=self.qcolour.currentText(),
                    skip=self.skip.value(),
                    vec_scale=self.vscale.value(),
                    smoothing=(self.smooth.value()
                               if self.smooth_on.isChecked() else 0),
                    background=(self.bkg.currentText()
                                if self.bkg_on.isChecked() else None),
                    image=self.image_combo.currentData(self._role),
                    cmap=self.cmap.currentText(),
                    shading=(self.shading.currentText()
                             if self.shading_on.isChecked() else "nearest"),
                    rois_shown=self._shown("rect"),
                    lines_shown=self._shown("line"))

    def _shown(self, kind):
        """The ROIs / lines ticked in Define ROI ({name: colour}; None:
        all)."""
        rp = getattr(self.main, "roi_panel", None)
        return rp.shown(kind) if rp is not None else None

    def get_state(self):
        return dict(root=self.root.text(),
                    run=self.run_combo.currentData(self._role),
                    files={k: row.get_state()
                           for k, row in self.setup_rows()},
                    mode=self.mode.currentIndex(), frame=self.frame.value(),
                    cmap=self.cmap.currentText(), skip=self.skip.value(),
                    shading=self.shading.currentText(),
                    shading_on=self.shading_on.isChecked(),
                    smooth_on=self.smooth_on.isChecked(),
                    field=self.velocity.currentText(),
                    roi=self.roi.currentData(self._role),
                    vec_scale=self.vscale.value(),
                    smoothing=self.smooth.value(),
                    mask=self.mask_combo.currentData(self._role),
                    quiver=self.quiver.isChecked(),
                    quiver_colour=self.qcolour.currentText(),
                    units=self.units.currentText(),
                    background_on=self.bkg_on.isChecked(),
                    background=self.bkg.currentText(),
                    image=self.image_combo.currentData(self._role))

    def _user_load(self):
        """The Load button: a new start -- the steps performed are counted
        again from the Load (sessions)."""
        self.main.journal = []
        self.load()

    def set_state(self, s):
        self.root.setText(s.get("root") or "")
        self.refresh(select_run=s.get("run"))
        files = s.get("files") or {}
        for k, row in self.setup_rows():
            row.set_state(files.get(k))
        if s.get("mode") is not None:
            self.mode.setCurrentIndex(int(s["mode"]))
        if s.get("cmap") and self.cmap.findText(s["cmap"]) >= 0:
            self.cmap.setCurrentText(s["cmap"])
        if s.get("shading") and self.shading.findText(s["shading"]) >= 0:
            self.shading.setCurrentText(s["shading"])
        self.shading_on.setChecked(bool(s.get(
            "shading_on", s.get("shading") not in (None, "nearest"))))
        if s.get("image") and os.path.isfile(s["image"]):
            self._extra_images = [s["image"]]
            self.fill_images(select=s["image"])
        if s.get("background") in BACKGROUNDS:
            self.bkg.setCurrentText(s["background"])
        if s.get("background_on") is not None:
            self.bkg_on.setChecked(bool(s["background_on"]))
        if s.get("quiver") is not None:
            self.quiver.setChecked(bool(s["quiver"]))
        if s.get("quiver_colour") in lib.QUIVER_COLOURS:
            self.qcolour.setCurrentText(s["quiver_colour"])
        field = lib.OLD_FIELD_NAMES.get(s.get("field"), s.get("field"))
        if field in (lib.VALIDATED, lib.UNVALIDATED, lib.REJECTED_NAN):
            self.velocity.blockSignals(True)
            self.velocity.setCurrentText(field)
            self.velocity.blockSignals(False)
        self._pending_roi = s.get("roi")
        if s.get("skip"):
            self.skip.setValue(int(s["skip"]))
        if s.get("vec_scale"):
            self.vscale.setValue(float(s["vec_scale"]))
        if s.get("smoothing"):
            self.smooth.setValue(float(s["smoothing"]))
        self.smooth_on.setChecked(bool(s.get(
            "smooth_on", bool(s.get("smoothing")))))
        self._pending_frame = s.get("frame")
        self._pending_units = s.get("units")
        self._pending_mask = s.get("mask")


class MovieThread(QtCore.QThread):
    """lib.make_movie outside the GUI thread."""

    progress = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, d, opts, frames, fps, path, size_in, kind="maps"):
        super().__init__()
        self.args = (d, opts, frames, fps, path, size_in)
        self.kind = kind
        self.stop = False

    def run(self):
        d, opts, frames, fps, path, size_in = self.args
        try:
            images = []
            info = lib.make_movie(
                d, opts, frames, fps, path, size_in, kind=self.kind,
                progress=lambda i, n: self.progress.emit(i, n),
                cancelled=lambda: self.stop,
                on_frame=lambda k, f, rgb: images.append(_png_of(rgb)))
            if info is not None:
                info["images"] = images          # for the movie player
            self.done.emit(info)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MovieBox(QtWidgets.QGroupBox):
    """Make Movie, below the options of a Processing step (Maps and Quiver,
    Profiles): start / end frame and fps (sliders + numbers), Create movie
    (Cancel while it runs), Reset, Save; the movie just made is played in
    the step's movie figure (MoviePage). The step's panel gives:
    STEP, FIG (its figure), MOVIE (the player's name), KIND (for
    lib.make_movie), movie_options() (the options of its figure),
    movie_note(opts) (for the log) and movie_stem(opts) (file name)."""

    def __init__(self, main, panel):
        super().__init__("Make Movie")
        self.main, self.panel = main, panel
        self.thread = None
        self.movie = None                        # the movie created
        box = self
        g = QtWidgets.QGridLayout(box)
        g.setColumnStretch(1, 1)
        self.sliders, self.spins = {}, {}
        for r, (key, lab, lo, hi, val, tip) in enumerate((
                ("start", "Start frame", 0, 0, 0, "first frame of the movie"),
                ("end", "End frame", 0, 0, 0, "last frame of the movie"),
                ("fps", "FPS", 1, 60, 10, "frames per second of the movie"))):
            sl = QtWidgets.QSlider(E("Qt.Orientation.Horizontal"))
            sp = QtWidgets.QSpinBox()
            for w in (sl, sp):
                w.setRange(lo, hi)
                w.setValue(val)
                w.setToolTip(tip)
            sp.setMinimumWidth(70)
            sl.valueChanged.connect(sp.setValue)
            sp.valueChanged.connect(sl.setValue)
            sp.valueChanged.connect(lambda _v, k=key: self._changed(k))
            g.addWidget(QtWidgets.QLabel(lab), r, 0)
            g.addWidget(sl, r, 1)
            g.addWidget(sp, r, 2)
            self.sliders[key], self.spins[key] = sl, sp
        self.length = QtWidgets.QLabel("")
        g.addWidget(self.length, 3, 0, 1, 3)
        row = QtWidgets.QHBoxLayout()
        self.create_btn = QtWidgets.QPushButton("Create movie")
        self.reset_btn = QtWidgets.QPushButton("Reset")
        self.save_btn = QtWidgets.QPushButton("Save")
        self.create_btn.setToolTip(
            "draw the figure with the settings of this step for every "
            "frame from the start to the end frame, with fixed colour "
            "limits (maps) or y limits (profiles), "
            "and make the movie (MP4 H.264, plays on macOS and Windows; "
            "an animated GIF when ffmpeg is not available)")
        self.reset_btn.setToolTip("start, end and FPS back to the whole "
                                  "record at 10 fps; the movie not saved is "
                                  "dropped")
        self.save_btn.setToolTip("save the movie created (default: the "
                                 "run's PostProcessing folder)")
        for b in (self.create_btn, self.reset_btn, self.save_btn):
            b.setMinimumHeight(32)
            row.addWidget(b)
        self.create_btn.clicked.connect(self.create)
        self.reset_btn.clicked.connect(self.reset)
        self.save_btn.clicked.connect(self.save)
        g.addLayout(row, 4, 0, 1, 3)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setVisible(False)
        g.addWidget(self.bar, 5, 0, 1, 3)
        self.movie_lab = QtWidgets.QLabel("")
        self.movie_lab.setWordWrap(True)
        g.addWidget(self.movie_lab, 6, 0, 1, 3)
        self.movie_box = box
        self._state()

    # ---- frames ----------------------------------------------------------- #
    def _changed(self, key):
        st, en = self.spins["start"], self.spins["end"]
        if key == "start" and st.value() > en.value():
            en.setValue(st.value())
        if key == "end" and en.value() < st.value():
            st.setValue(en.value())
        self._length()

    def _length(self):
        d = self.main.data
        n = self.spins["end"].value() - self.spins["start"].value() + 1
        fps = self.spins["fps"].value()
        txt = "%d frames -> %.1f s of movie at %d fps" % (n, n / float(fps),
                                                         fps)
        if d is not None and d.calibrated:
            txt += " (record: %.3f .. %.3f s)" % (
                d.t[self.spins["start"].value()], d.t[self.spins["end"].value()])
        self.length.setText(txt)

    def data_loaded(self, d):
        """A new run: the frame range is its whole record; a movie not
        saved is dropped."""
        self._drop()
        for k in ("start", "end"):
            for w in (self.sliders[k], self.spins[k]):
                w.blockSignals(True)
                w.setRange(0, d.nt - 1)
                w.blockSignals(False)
        self.spins["start"].setValue(0)
        self.spins["end"].setValue(d.nt - 1)
        self.sliders["start"].setValue(0)
        self.sliders["end"].setValue(d.nt - 1)
        self._length()
        self._state()

    def _state(self):
        d = self.main.data
        busy = self.thread is not None
        self.movie_box.setEnabled(d is not None)
        self.create_btn.setText("Cancel" if busy else "Create movie")
        self.reset_btn.setEnabled(not busy)
        self.save_btn.setEnabled(not busy and self.movie is not None)
        for k in self.spins:
            self.spins[k].setEnabled(not busy)
            self.sliders[k].setEnabled(not busy)

    # ---- movie ------------------------------------------------------------ #
    def create(self):
        if self.thread is not None:              # Cancel
            self.thread.stop = True
            self.main.status("Cancelling the movie...")
            return
        d = self.main.data
        if d is None:
            return
        self._drop(keep_player=True)
        opts = self.panel.movie_options()
        frames = list(range(self.spins["start"].value(),
                            self.spins["end"].value() + 1))
        fps = self.spins["fps"].value()
        # the shape of the figure on screen, 1280 px wide (10 in at 128 dpi)
        page = self.main.figures.find(self.panel.STEP, self.panel.FIG)
        wi, hi = tuple(page.figure.get_size_inches()) if page else (8, 6)
        aspect = min(max(hi / float(wi or 1), 0.4), 1.5)
        size = (10.0, 10.0 * aspect)
        import tempfile
        fdesc, path = tempfile.mkstemp(prefix="piv_movie_", suffix=".mp4")
        os.close(fdesc)
        self._tmp = path                         # removed if not the movie
        self._opts = opts
        self.thread = MovieThread(d, opts, frames, fps, path, size,
                                  self.panel.KIND)
        self.thread.progress.connect(self._progress)
        self.thread.done.connect(self._done)
        self.thread.failed.connect(self._failed)
        self.thread.finished.connect(self._finished)
        self.bar.setRange(0, len(frames))
        self.bar.setValue(0)
        self.bar.setVisible(True)
        self.movie_lab.setText("Making the movie...")
        self.main.write_log("Movie (%s): frames %d..%d at %d fps, %s%s" % (
            self.panel.FIG, frames[0], frames[-1], fps,
            self.panel.movie_note(opts),
            ", mask %s" % d.mask_name if d.mask_name else ""))
        self.thread.start()
        self._state()

    def _progress(self, i, n):
        self.bar.setValue(i)
        self.main.status("Movie: frame %d / %d" % (i, n), 0)

    def _done(self, info):
        if info is None:
            self._drop()                         # and the old player
            self.movie_lab.setText("Cancelled.")
            self.main.status("Movie cancelled", log=True)
            return
        images = info.pop("images", [])
        self.movie = info
        self.show_movie(images)
        self.movie_lab.setText(
            "<b>Movie ready</b> (not saved yet): %d frames (%d .. %d), "
            "%.1f s at %d fps, %d x %d, %s. Save to keep it." % (
                info["frames"], info["first"], info["last"],
                info["seconds"], info["fps"], info["size"][0],
                info["size"][1], info["format"]))
        self.main.status("Movie ready: %d frames, %s -- Save to keep it" % (
            info["frames"], info["format"]), log=True)

    def show_movie(self, images):
        """The figure 'Movie': the movie just made, playing."""
        info, d = dict(self.movie), self.main.data
        info["times"] = None
        try:
            if d is not None and d.calibrated:
                info["times"] = [float(d.t[f]) for f in
                                 range(info["first"], info["last"] + 1)]
        except Exception:
            pass
        info["file"] = info["path"]
        self.main.figures.add_movie(self.panel.STEP, self.panel.MOVIE, images, info)

    def _failed(self, msg):
        self._drop()
        self.movie_lab.setText("The movie failed (see the log).")
        self.main.write_log("ERROR while making the movie:\n%s"
                            % msg.rstrip(), error=True)

    def _finished(self):
        tmp = getattr(self, "_tmp", None)        # placeholder not used
        if tmp and os.path.exists(tmp) and not (
                self.movie and self.movie["path"] == tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass
        self._tmp = None
        self.thread = None
        self.bar.setVisible(False)
        self._state()

    def _drop(self, keep_player=False):
        """The movie created and not saved: removed, with its player (the
        figure 'Movie') unless keep_player (a new movie is being made)."""
        if self.movie and os.path.isfile(self.movie["path"]):
            try:
                os.remove(self.movie["path"])
            except Exception:
                pass
        self.movie = None
        self.movie_lab.setText("")
        page = self.main.figures.find(self.panel.STEP, self.panel.MOVIE)
        if page is not None:
            if keep_player:                      # replaced by the new one
                page.pause()
            else:
                self.main.figures.remove(page)   # its movie is gone

    def default_name(self):
        d, o, m = self.main.data, getattr(self, "_opts", {}), self.movie
        return "%s_%s_f%d-%d_%dfps%s" % (
            d.run, self.panel.movie_stem(o), m["first"], m["last"], m["fps"],
            os.path.splitext(m["path"])[1])

    def save(self):
        if not self.movie:
            return
        d = self.main.data
        ext = os.path.splitext(self.movie["path"])[1]
        start = os.path.join(d.post_dir or d.run_dir, self.default_name())
        path, _f = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save the movie", start,
            "MP4 movie (*.mp4)" if ext == ".mp4" else "GIF (*.gif)")
        if not path:
            return
        if not path.lower().endswith(ext):
            path += ext
        import shutil
        try:
            shutil.copyfile(self.movie["path"], path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save the movie", str(exc))
            return
        self.movie_lab.setText("<b>Saved:</b> %s" % path)
        self.main.status("Movie saved: %s" % path, log=True)

    def reset(self):
        d = self.main.data
        self._drop()
        self.spins["fps"].setValue(10)
        if d is not None:
            self.spins["start"].setValue(0)
            self.spins["end"].setValue(d.nt - 1)
        self.main.status("Make Movie reset")
        self._state()

    def close(self):
        """The window closes: a movie being made is stopped, a movie not
        saved is removed."""
        if self.thread is not None:
            self.thread.stop = True
            self.thread.wait(10000)
        self._drop()


class MapsPanel(QtWidgets.QWidget):
    """Processing > Maps and Quiver: the Display box (velocity calibrated
    or not, ROI field, mask, frame, quiver, background map, colour map) and
    the figure 'Maps and Quiver'. The box belongs to the LoadPanel object
    (its options are read from there); it is shown here. Below it: Make
    Movie (MovieBox) -- the figure, with these settings, for every frame
    from a start to an end frame."""

    STEP = "maps"
    FIG = "Maps and Quiver"
    MOVIE = "Movie"
    KIND = "maps"

    def __init__(self, main):
        super().__init__()
        self.main = main
        lay = QtWidgets.QVBoxLayout(self)
        self.note = QtWidgets.QLabel("Load a run first (File > Load data).")
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        lay.addWidget(main.load_panel.display_box)
        self.mv = MovieBox(main, self)
        lay.addWidget(self.mv)
        lay.addStretch(1)

    def movie_options(self):
        return self.main.load_panel.display_options()

    def movie_note(self, o):
        return "%s%s" % (o.get("roi"), ", background %s" % o["background"]
                         if o.get("background") else "")

    def movie_stem(self, o):
        what = (o.get("background") or "quiver").replace(" ", "_")
        roi = "" if o.get("roi") in (None, "Full") else "_" + o["roi"]
        return what + roi

    def data_loaded(self, d):
        self.mv.data_loaded(d)

    def close(self):
        self.mv.close()

    def open(self):
        self.note.setVisible(self.main.data is None)
        if self.main.data is not None:
            self.main.draw_field(force=True)


class ProfilesPanel(QtWidgets.QWidget):
    """Processing > Profiles: the layout of Maps and Quiver, with Lines in
    place of ROI and no quiver / background: a quantity (Extracted
    quantity: the quantities of the maps) sampled along the lines of
    Define ROI (LINES, ticked ones), plotted against the distance along the
    line. Velocity (calibrated or not) and Mask are those of the run (the
    same as in Maps and Quiver); the frame is the step's own. Below: Make
    Movie (MovieBox) -- the profiles for every frame of a range, with fixed
    y limits."""

    STEP = "profiles"
    FIG = "Profiles"
    MOVIE = "Profile movie"
    KIND = "profiles"
    ALL = "__all__"

    def __init__(self, main):
        super().__init__()
        self.main = main
        self._role = E("Qt.ItemDataRole.UserRole")
        lay = QtWidgets.QVBoxLayout(self)
        self.note = QtWidgets.QLabel("Load a run first (File > Load data).")
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        box = QtWidgets.QGroupBox("Display")
        g = QtWidgets.QGridLayout(box)
        g.setColumnStretch(1, 1)
        r = 0
        self.units = QtWidgets.QComboBox()
        self.units.setToolTip(
            "velocity shown: calibrated (m, m/s, s) or uncalibrated (px, "
            "px/frame, frame numbers) -- the same choice as in Maps and "
            "Quiver")
        g.addWidget(QtWidgets.QLabel("Velocity"), r, 0)
        g.addWidget(self.units, r, 1, 1, 2); r += 1
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["single frame", "time mean", "rms"])
        self.frame = QtWidgets.QSlider(E("Qt.Orientation.Horizontal"))
        self.frame_spin = QtWidgets.QSpinBox()
        g.addWidget(QtWidgets.QLabel("Frame"), r, 0)
        g.addWidget(self.mode, r, 1, 1, 2); r += 1
        g.addWidget(self.frame, r, 1)
        g.addWidget(self.frame_spin, r, 2); r += 1
        self.frame_info = QtWidgets.QLabel("")
        g.addWidget(self.frame_info, r, 1, 1, 2); r += 1
        self.line = QtWidgets.QComboBox()
        self.line.setToolTip(
            "the lines of Pre-processing > Define ROI (LINES) ticked there, "
            "each in its colour; All lines: every one on the same figure")
        g.addWidget(QtWidgets.QLabel("Lines"), r, 0)
        g.addWidget(self.line, r, 1, 1, 2); r += 1
        self.roi = RoiCombo(main)
        self.roi.setToolTip(
            "Full: the quantity of the full domain along the whole lines; an "
            "ROI (ticked in Define ROI): the quantity calculated on that ROI "
            "(Define ROI > Apply: <quantity>_ROIxx) at the points of the "
            "lines inside the ROI only")
        g.addWidget(QtWidgets.QLabel("ROI"), r, 0)
        g.addWidget(self.roi, r, 1, 1, 2); r += 1
        self.mask_combo = QtWidgets.QComboBox()
        self.mask_combo.setToolTip(
            "the mask applied to the run (velocity NaN in it) -- the same "
            "choice as in Maps and Quiver")
        g.addWidget(QtWidgets.QLabel("Mask"), r, 0)
        g.addWidget(self.mask_combo, r, 1, 1, 2); r += 1
        self.quantity = QtWidgets.QComboBox()
        self.quantity.addItems(lib.QUANTITIES)
        self.quantity.setToolTip(
            "the quantity extracted along the lines: velocity norm |u|, U, "
            "V, divergence, vorticity or kinetic energy -- computed on the "
            "full field with the mask applied, then interpolated "
            "(bilinear) along the line, about one point per vector "
            "spacing")
        g.addWidget(QtWidgets.QLabel("Extracted quantity"), r, 0)
        g.addWidget(self.quantity, r, 1, 1, 2); r += 1
        self.box = box
        box.setEnabled(False)
        lay.addWidget(box)
        self.mv = MovieBox(main, self)
        lay.addWidget(self.mv)
        lay.addStretch(1)

        self.frame.valueChanged.connect(self.frame_spin.setValue)
        self.frame_spin.valueChanged.connect(self.frame.setValue)
        self.frame.valueChanged.connect(lambda *_: self._frame_info())
        self.frame.sliderReleased.connect(self.draw)
        self.frame.valueChanged.connect(
            lambda _v: None if self.frame.isSliderDown() else self.draw())
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.line.currentIndexChanged.connect(lambda *_: self._selected())
        self.roi.currentIndexChanged.connect(lambda *_: self._selected())
        self.quantity.currentIndexChanged.connect(lambda *_: self.draw())
        self.units.currentIndexChanged.connect(
            lambda i: self._to_source(self.main.load_panel.units, i))
        self.mask_combo.currentIndexChanged.connect(
            lambda i: self._to_source(self.main.load_panel.mask_combo, i))

    # ---- the run's choices (Velocity, Mask), shared with Maps --------------- #
    def _mirror(self, cb, src):
        cb.blockSignals(True)
        cb.clear()
        for i in range(src.count()):
            cb.addItem(src.itemIcon(i), src.itemText(i),
                       src.itemData(i, self._role))
        cb.setCurrentIndex(src.currentIndex())
        cb.blockSignals(False)

    def _to_source(self, src, i):
        """A choice here is made in Maps and Quiver's box (it applies it to
        the run; sync() then brings it back here)."""
        if 0 <= i < src.count() and src.currentIndex() != i:
            src.setCurrentIndex(i)

    def sync(self, redraw=True):
        """The run's Velocity and Mask choices and the lines (ticked in
        Define ROI) again; the figure redrawn if it is there."""
        lp = self.main.load_panel
        self._mirror(self.units, lp.units)
        self._mirror(self.mask_combo, lp.mask_combo)
        self._fill_lines()
        self._fill_rois()
        self._frame_info()
        if redraw:
            self.draw()

    def _fill_rois(self):
        """The replica of Define ROI's table (Full + the ROIs)."""
        keep = self.roi.currentData(self._role) or "Full"
        self.roi.blockSignals(True)
        self.roi.refill()
        self.roi.blockSignals(False)
        if self.roi.currentData(self._role) != keep:
            self._warned = None

    def _selected(self):
        """A line or an ROI chosen: lines that do not cross the ROI are
        reported (once per choice), then the figure."""
        self._check_lines()
        self.draw()

    def _check_lines(self):
        d = self.main.data
        roi = self.roi.currentData(self._role)
        if d is None or roi in (None, "Full"):
            return
        o = self.options()
        key = (roi, tuple(n for n, _r, _c in o["lines"]))
        if getattr(self, "_warned", None) == key:
            return
        self._warned = key
        info = d.roi_fields().get(roi)
        miss = [n for n, rect, _c in o["lines"]
                if not lib.line_in_roi(d, rect, roi)]
        if miss and info is not None:
            msg = "\n".join("The selected line %s does not intersect the "
                            "selected ROI %s." % (n, info["roi"])
                            for n in miss)
            self.main.write_log(msg)
            QtWidgets.QMessageBox.information(self, "Profiles", msg)

    def lines(self):
        """[(name, rect in m, colour)] of the ticked lines of the run."""
        d = self.main.data
        if d is None or not d.can_calibrate:
            return []
        rp = getattr(self.main, "roi_panel", None)
        shown = rp.shown("line") if rp is not None else None
        out = []
        for k, (name, rect) in enumerate(
                lib.parse_shapes(d.params.get("LINES"), "LINE")):
            if shown is not None and name not in shown:
                continue
            c = (shown or {}).get(name) or lib.roi_colour("line", k)
            out.append((name, rect, c))
        return out

    def _fill_lines(self):
        cb = self.line
        keep = cb.currentData(self._role) or self.ALL
        lines = self.lines()
        cb.blockSignals(True)
        cb.clear()
        if lines:
            cb.addItem("All lines", self.ALL)
        for name, _rect, c in lines:
            pm = QtGui.QPixmap(12, 12)
            pm.fill(QtGui.QColor(c))
            p = QtGui.QPainter(pm)
            p.setPen(QtGui.QColor("#555555"))
            p.drawRect(0, 0, 11, 11)
            p.end()
            cb.addItem(QtGui.QIcon(pm), name, name)
        if not lines:
            cb.addItem("(no line: Pre-processing > Define ROI)", None)
        i = cb.findData(keep, self._role)
        cb.setCurrentIndex(max(i, 0))
        cb.blockSignals(False)

    # ---- frame ------------------------------------------------------------- #
    def _mode_changed(self, i):
        for w in (self.frame, self.frame_spin):
            w.setEnabled(i == 0)
        self._frame_info()
        self.draw()

    def _frame_info(self):
        d = self.main.data
        if d is None:
            self.frame_info.setText("")
        elif self.mode.currentIndex() != 0:
            self.frame_info.setText("all %d frames" % d.nt)
        else:
            self.frame_info.setText(d.frame_text(
                min(self.frame.value(), d.nt - 1)))

    def data_loaded(self, d):
        for w in (self.frame, self.frame_spin):
            w.blockSignals(True)
            w.setRange(0, d.nt - 1)
            w.blockSignals(False)
        self.box.setEnabled(True)
        self.mv.data_loaded(d)
        self.sync(redraw=True)

    # ---- the figure -------------------------------------------------------- #
    def options(self):
        """The options of the Profiles figure (lib.profile_figure)."""
        d = self.main.data
        sel = self.line.currentData(self._role)
        lines = self.lines()
        if sel != self.ALL:
            lines = [l for l in lines if l[0] == sel]
        frame = {1: "mean", 2: "rms"}.get(self.mode.currentIndex(),
                                         self.frame.value())
        roi = self.roi.currentData(self._role)
        return dict(frame=frame, lines=lines,
                    quantity=self.quantity.currentText(),
                    calibrated=bool(d is not None and d.calibrated),
                    roi=None if roi in (None, "Full") else roi)

    def draw(self, force=False):
        d = self.main.data
        if d is None:
            return None
        page = self.main.figures.find(self.STEP, self.FIG)
        if page is None and not force and self.main.step != self.STEP:
            return None                          # not shown yet
        opts = self.options()
        try:
            if page is None:
                fig = lib.profile_figure(d, fig=lib.new_figure(), **opts)
                page = self.main.figures.add(
                    self.STEP, self.FIG, fig, data_fn=self._data,
                    update_fn=lambda: self.draw(force=True))
            else:
                lib.profile_figure(d, fig=page.figure, **opts)
                page.redraw()
        except Exception as exc:
            self.main.status("Figure error: %s" % exc, log=True)
            return None
        return page

    def _data(self):
        return lib.profile_data(self.main.data, **self.options())

    # ---- Make Movie ---------------------------------------------------------- #
    def movie_options(self):
        return self.options()

    def movie_note(self, o):
        return "%s along %s%s" % (o["quantity"], ", ".join(
            n for n, _r, _c in o["lines"]) or "no line",
            " in %s" % o["roi"] if o.get("roi") else "")

    def movie_stem(self, o):
        sel = self.line.currentData(self._role)
        what = "all_lines" if sel == self.ALL else str(sel)
        return "profile_%s_%s%s" % (o["quantity"].replace(" ", "_"), what,
                                    "_" + o["roi"] if o.get("roi") else "")

    def close(self):
        self.mv.close()

    def open(self):
        self.note.setVisible(self.main.data is None)
        if self.main.data is not None:
            self.sync(redraw=False)
            self.draw(force=True)

    # ---- session ------------------------------------------------------------ #
    def get_state(self):
        return dict(mode=self.mode.currentIndex(), frame=self.frame.value(),
                    line=self.line.currentData(self._role),
                    roi=self.roi.currentData(self._role),
                    quantity=self.quantity.currentText())

    def set_state(self, s):
        if s.get("mode") is not None:
            self.mode.setCurrentIndex(int(s["mode"]))
        if s.get("quantity"):
            self.quantity.setCurrentText(s["quantity"])
        self._pending = s

    def apply_pending(self):
        """After the Load of a session: frame and line."""
        s = getattr(self, "_pending", None)
        if not s:
            return
        self._pending = None
        if s.get("frame") is not None:
            self.frame.setValue(int(s["frame"]))
        i = self.line.findData(s.get("line"), self._role)
        if i >= 0:
            self.line.setCurrentIndex(i)
        i = self.roi.findData(s.get("roi"), self._role)
        if i >= 0:
            self.roi.setCurrentIndex(i)


class TimeSeriesPanel(QtWidgets.QWidget):
    """Processing > Time series: for each ROI ticked (a replica of the ROI
    table of Define ROI, with its own ticks: the ROIs active here), a time
    series built from its quantities of Define ROI > Apply (U_ROIxx,
    V_ROIxx, vorticity_ROIxx, kinetic_energy_ROIxx -- not calculated
    again): the mean or standard deviation over the ROI, of the quantity or
    of its absolute value, for every frame. Figure 'Time series'."""

    STEP = "timeseries"
    FIG = "Time series"

    def __init__(self, main):
        super().__init__()
        self.main = main
        self._role = E("Qt.ItemDataRole.UserRole")
        lay = QtWidgets.QVBoxLayout(self)
        self.note = QtWidgets.QLabel("Load a run first (File > Load data).")
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        box = QtWidgets.QGroupBox("ROI")
        v = QtWidgets.QVBoxLayout(box)
        self.rois = RoiTable(main, "tick")
        self.rois.setToolTip(
            "the ROI table of Define ROI: tick the ROIs to make them active "
            "(one curve each, in its colour); grey: not applied yet "
            "(Define ROI > Apply)")
        v.addWidget(self.rois)
        lay.addWidget(box)
        qbox = QtWidgets.QGroupBox("Time series of")
        g = QtWidgets.QGridLayout(qbox)
        self.quantity = QtWidgets.QComboBox()
        for name, _q, _s, _a, desc in lib.TIME_SERIES:
            self.quantity.addItem(name)
            self.quantity.setItemData(self.quantity.count() - 1, desc,
                                      E("Qt.ItemDataRole.ToolTipRole"))
        self.quantity.setToolTip("the quantity of the time series, for "
                                 "every frame, over the points of the ROI "
                                 "(NaN -- mask, rejected vectors -- left "
                                 "out)")
        g.addWidget(QtWidgets.QLabel("Quantity"), 0, 0)
        g.addWidget(self.quantity, 0, 1)
        g.setColumnStretch(1, 1)
        self.desc = QtWidgets.QLabel("")
        self.desc.setWordWrap(True)
        self.desc.setStyleSheet("color: gray;")
        g.addWidget(self.desc, 1, 0, 1, 2)
        lay.addWidget(qbox)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        lay.addStretch(1)
        self.quantity.currentIndexChanged.connect(lambda *_: self.draw())
        self.rois.ticksChanged.connect(self.draw)
        self._desc()

    def _desc(self):
        i = self.quantity.currentIndex()
        self.desc.setText(lib.TIME_SERIES[i][4] if i >= 0 else "")

    def sync(self, redraw=True):
        """The ROI table again (Define ROI changed, Apply, a new run)."""
        self.rois.refill()
        d = self.main.data
        self.note.setVisible(d is None)
        self.info.setText(
            "" if d is None else
            "Calibrated units (m/s, 1/s, m²/s²), time in s; mask "
            "applied: %s (Maps and Quiver). The ROI quantities are those of "
            "Define ROI > Apply (calculated once if missing)."
            % (d.mask_name or "none"))
        if redraw:
            self.draw()

    def data_loaded(self, d):
        self.rois._own_ticks = {}
        self.sync(redraw=True)

    def open(self):
        self.sync(redraw=False)
        self.draw(force=True)

    def options(self):
        return dict(rois=self.rois.ticked_fields(),
                    name=self.quantity.currentText())

    def draw(self, force=False):
        self._desc()
        d = self.main.data
        if d is None:
            return None
        page = self.main.figures.find(self.STEP, self.FIG)
        if page is None and not force and self.main.step != self.STEP:
            return None
        if not d.can_calibrate:
            self.info.setText("The run cannot be calibrated: no ROI field, "
                              "no time series.")
        opts = self.options()
        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(E("Qt.CursorShape.WaitCursor")))
        try:
            if page is None:
                fig = lib.time_series_figure(d, fig=lib.new_figure(), **opts)
                page = self.main.figures.add(
                    self.STEP, self.FIG, fig, data_fn=self._data,
                    update_fn=lambda: self.draw(force=True))
            else:
                lib.time_series_figure(d, fig=page.figure, **opts)
                page.redraw()
        except Exception as exc:
            self.main.status("Figure error: %s" % exc, log=True)
            return None
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        return page

    def _data(self):
        return lib.time_series_data(self.main.data, **self.options())

    def get_state(self):
        return dict(quantity=self.quantity.currentText(),
                    ticks=dict(self.rois._own_ticks))

    def set_state(self, s):
        if s.get("quantity"):
            self.quantity.setCurrentText(s["quantity"])
        self._pending_ticks = s.get("ticks")

    def apply_pending(self):
        t = getattr(self, "_pending_ticks", None)
        if t:
            self.rois._own_ticks.update(t)
            self.sync(redraw=False)
        self._pending_ticks = None


class FFTTimePanel(QtWidgets.QWidget):
    """Spectral analysis > FFT in time: ROI (drop-down, with Full), FFT
    quantity (the list of the time series), order of the calculation
    (average over the ROI then FFT, and / or FFT at each point then
    average: one or both ticked), Apply -> figure 'FFT in time'
    (amplitude, phase, real and imaginary parts; one curve per order).
    Display: frequency or period, log amplitude. Frequency ranges: Add
    range (drag on the figure), Remove; the range ticked (one at a time)
    is the one used. FFT first: Plot FFT maps -> figure 'FFT map' (the
    amplitude integrated over the range ticked, at each point)."""

    STEP = "fft_time"
    FIG = "FFT in time"
    MAP = "FFT map"

    def __init__(self, main):
        super().__init__()
        self.main = main
        self._role = E("Qt.ItemDataRole.UserRole")
        self.results = {}                # order -> result of fft_in_time
        self.windows = []                # [dict(fmin, fmax, on)] Hz
        self._spans = []
        lay = QtWidgets.QVBoxLayout(self)
        self.note = QtWidgets.QLabel("Load a run first (File > Load data).")
        self.note.setWordWrap(True)
        lay.addWidget(self.note)

        box = QtWidgets.QGroupBox("ROI")
        g = QtWidgets.QHBoxLayout(box)
        self.roi = RoiCombo(main)
        self.roi.setToolTip(
            "Full field, or an ROI ticked in Define ROI (its quantities of "
            "Define ROI > Apply are used, not calculated again)")
        g.addWidget(self.roi)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("FFT quantity")
        g = QtWidgets.QVBoxLayout(box)
        self.quantity = QtWidgets.QComboBox()
        for name, _q, _s, _a, desc in lib.TIME_SERIES:
            self.quantity.addItem(name)
            self.quantity.setItemData(self.quantity.count() - 1, desc,
                                      E("Qt.ItemDataRole.ToolTipRole"))
        g.addWidget(self.quantity)
        self.desc = QtWidgets.QLabel("")
        self.desc.setWordWrap(True)
        self.desc.setStyleSheet("color: gray;")
        g.addWidget(self.desc)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("Spectral function")
        g = QtWidgets.QGridLayout(box)
        self.function = QtWidgets.QComboBox()
        for key, label in lib.SPECTRAL_FUNCTIONS:
            self.function.addItem(label, key)
            self.function.setItemData(self.function.count() - 1,
                                      lib.SPECTRAL_HELP[key],
                                      E("Qt.ItemDataRole.ToolTipRole"))
        g.addWidget(self.function, 0, 0, 1, 2)
        self.seg_lab = QtWidgets.QLabel("Segment length")
        self.seg = QtWidgets.QSpinBox()
        self.seg.setRange(0, 100000)
        self.seg.setSpecialValueText("auto")
        self.seg.setSuffix(" frames")
        self.seg.setToolTip("Welch: the length of each segment (0 = auto: "
                            "a power of 2 near a quarter of the record); "
                            "longer -> finer in frequency, noisier")
        g.addWidget(self.seg_lab, 1, 0)
        g.addWidget(self.seg, 1, 1)
        g.setColumnStretch(1, 1)
        self.remove_mean = QtWidgets.QCheckBox("Remove mean")
        self.remove_mean.setChecked(True)
        self.remove_mean.setToolTip(
            "the mean of each series removed before the spectral function "
            "(f = 0 becomes 0); power spectrum: in each segment")
        self.detrend = QtWidgets.QCheckBox("Detrend")
        self.detrend.setToolTip(
            "the linear trend of each series (the best straight line in "
            "time) removed before the spectral function -- the mean too; "
            "power spectrum: in each segment")
        g.addWidget(self.remove_mean, 2, 0)
        g.addWidget(self.detrend, 2, 1)
        self.fdesc = QtWidgets.QLabel("")
        self.fdesc.setWordWrap(True)
        self.fdesc.setStyleSheet("color: gray;")
        g.addWidget(self.fdesc, 3, 0, 1, 2)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("Order of the calculation")
        g = QtWidgets.QVBoxLayout(box)
        self.order_box = {}
        tips = {"average": "first average the quantity over the ROI (the "
                           "time series), then its spectral analysis",
                "fft": "the spectral analysis at each point of the ROI, "
                       "then averaged over the ROI (FFT -- amplitude: mean "
                       "of the amplitudes; real, imaginary parts: mean; "
                       "phase: of the mean spectrum; power spectrum, "
                       "periodogram: mean of the densities) -- allows the "
                       "FFT maps"}
        for key, label in lib.FFT_ORDERS:
            cb = QtWidgets.QCheckBox(label)
            cb.setToolTip(tips[key] + " (both can be ticked: one curve "
                                      "each)")
            self.order_box[key] = cb
            g.addWidget(cb)
        self.order_box["average"].setChecked(True)
        lay.addWidget(box)

        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.setMinimumHeight(34)
        f = self.apply_btn.font(); f.setBold(True)
        self.apply_btn.setFont(f)
        self.apply_btn.setToolTip("calculate the FFT selected: figure 'FFT "
                                  "in time' (amplitude, phase, real part, "
                                  "imaginary part)")
        lay.addWidget(self.apply_btn)
        self.result = QtWidgets.QLabel("")
        self.result.setWordWrap(True)
        lay.addWidget(self.result)

        box = QtWidgets.QGroupBox("Display")
        g = QtWidgets.QGridLayout(box)
        self.display = QtWidgets.QComboBox()
        self.display.addItem("frequency", "frequency")
        self.display.addItem("period", "period")
        self.display.setToolTip("x axis: frequency (Hz), or period (s, log "
                                "scale; f = 0 left out)")
        self.log_amp = QtWidgets.QCheckBox("log scale (amplitude / density)")
        g.addWidget(QtWidgets.QLabel("x axis"), 0, 0)
        g.addWidget(self.display, 0, 1)

        def fspin(tip):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(0.0, 1.0e5)
            sp.setDecimals(4)
            sp.setSingleStep(0.05)
            sp.setSpecialValueText("auto")
            sp.setSuffix(" Hz")
            sp.setToolTip(tip + " (0 = auto); in period display: the "
                          "periods 1/fmax .. 1/fmin")
            sp.setKeyboardTracking(False)
            return sp
        self.fmin = fspin("the lowest frequency shown")
        self.fmax = fspin("the highest frequency shown")
        g.addWidget(QtWidgets.QLabel("fmin"), 1, 0)
        g.addWidget(self.fmin, 1, 1)
        g.addWidget(QtWidgets.QLabel("fmax"), 2, 0)
        g.addWidget(self.fmax, 2, 1)
        self.frange = QtWidgets.QLabel("")
        self.frange.setStyleSheet("color: gray;")
        self.frange.setWordWrap(True)
        g.addWidget(self.frange, 3, 1)
        g.addWidget(self.log_amp, 4, 1)
        g.setColumnStretch(1, 1)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("Frequency / period ranges")
        v = QtWidgets.QVBoxLayout(box)
        self.win_tree = QtWidgets.QTreeWidget()
        self.win_tree.setColumnCount(3)
        self.win_tree.setHeaderLabels(["Range (tick: used)",
                                       "frequency", "period"])
        self.win_tree.setRootIsDecorated(False)
        self.win_tree.setMaximumHeight(140)
        self.win_tree.setToolTip(
            "the ranges selected on the figure; the range ticked -- one at "
            "a time -- is the one used (FFT maps, Save data); none ticked: "
            "all frequencies")
        v.addWidget(self.win_tree)
        row = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("Add range")
        self.add_btn.setToolTip("then drag on one of the plots of the FFT "
                                "figure (left button held); Esc cancels")
        self.rem_btn = QtWidgets.QPushButton("Remove range")
        row.addWidget(self.add_btn)
        row.addWidget(self.rem_btn)
        v.addLayout(row)
        lay.addWidget(box)

        self.map_btn = QtWidgets.QPushButton("Plot FFT maps")
        self.map_btn.setToolTip(
            "(spectral analysis first then average) a map of the FFT "
            "amplitude / power spectral density integrated "
            "over the selected frequency / period range, at each point of "
            "the ROI")
        lay.addWidget(self.map_btn)
        self.map_style = MapStyleBox("FFT map display")
        lay.addWidget(self.map_style)
        self.map_style.changed.connect(self._restyle_map)

        box = QtWidgets.QGroupBox("Find peaks")
        g = QtWidgets.QGridLayout(box)
        self.npeaks = QtWidgets.QSpinBox()
        self.npeaks.setRange(1, 50)
        self.npeaks.setValue(3)
        self.npeaks.setToolTip("the number n of dominant peaks to find")
        self.peaks_btn = QtWidgets.QPushButton("Find peaks")
        self.peaks_btn.setToolTip(
            "the n dominant peaks of the amplitude (FFT) or of the density "
            "(power spectrum, periodogram), for each order calculated: the "
            "n highest local maxima, f = 0 left out, in the frequency range "
            "ticked (all frequencies if none); marked on the figure")
        g.addWidget(QtWidgets.QLabel("Number of peaks n"), 0, 0)
        g.addWidget(self.npeaks, 0, 1)
        g.addWidget(self.peaks_btn, 0, 2)
        self.peaks_tree = QtWidgets.QTreeWidget()
        self.peaks_tree.setHeaderLabels(["#", "order", "frequency", "period",
                                         "amplitude"])
        self.peaks_tree.setRootIsDecorated(False)
        self.peaks_tree.setMaximumHeight(160)
        g.addWidget(self.peaks_tree, 1, 0, 1, 3)
        g.setColumnStretch(0, 1)
        lay.addWidget(box)
        lay.addStretch(1)
        self.peaks = {}                          # order -> peaks

        self.quantity.currentIndexChanged.connect(lambda *_: self._desc())
        self.function.currentIndexChanged.connect(self._function_changed)
        self.detrend.toggled.connect(self._detrend_toggled)
        self.peaks_btn.clicked.connect(lambda *_: self.find_peaks())
        self.apply_btn.clicked.connect(self.apply)
        self.display.currentIndexChanged.connect(lambda *_: self.redraw())
        self.fmin.valueChanged.connect(lambda *_: self.redraw())
        self.fmax.valueChanged.connect(lambda *_: self.redraw())
        self.log_amp.toggled.connect(lambda *_: self.redraw())
        self.add_btn.clicked.connect(self.start_range)
        self.rem_btn.clicked.connect(self.remove_range)
        self.win_tree.itemChanged.connect(self._win_ticked)
        self.map_btn.clicked.connect(self.plot_map)
        for cb in self.order_box.values():
            cb.toggled.connect(lambda *_: self._state())
        self._desc()
        self._state()

    # ---- state ------------------------------------------------------------ #
    def _desc(self):
        i = self.quantity.currentIndex()
        self.desc.setText(lib.TIME_SERIES[i][4] if i >= 0 else "")
        fn = self.function.currentData(self._role) or "fft"
        self.fdesc.setText(lib.SPECTRAL_HELP[fn])
        for w in (self.seg_lab, self.seg):
            w.setVisible(fn == "power")

    def _detrend_toggled(self, on):
        """Detrend removes the mean too: Remove mean ticked and greyed."""
        if on:
            self._mean_before = self.remove_mean.isChecked()
            self.remove_mean.setChecked(True)
        else:
            self.remove_mean.setChecked(getattr(self, "_mean_before", True))
        self.remove_mean.setEnabled(not on)

    def _function_changed(self, *_):
        """A power spectrum is best seen on a log scale: ticked when one
        is chosen, unticked for the FFT."""
        fn = self.function.currentData(self._role)
        self.log_amp.setChecked(fn != "fft")
        self._desc()

    def orders(self):
        """The orders ticked ('average', 'fft')."""
        return [k for k, cb in self.order_box.items() if cb.isChecked()]

    @property
    def res(self):
        """The results shown, in the order of FFT_ORDERS (or None)."""
        out = [self.results[k] for k, _l in lib.FFT_ORDERS
               if k in self.results]
        return out or None

    def _state(self):
        d = self.main.data
        self.note.setVisible(d is None)
        self.apply_btn.setEnabled(d is not None and bool(self.orders()))
        self.map_btn.setVisible("fft" in self.orders())
        ticked = self._ticked_range()
        self.map_btn.setEnabled("fft" in self.results and ticked)
        self.map_btn.setToolTip(
            "(spectral analysis first then average) a map of the FFT "
            "amplitude / power spectral density integrated over the "
            "ticked frequency / period range, at each point of the ROI"
            + ("" if ticked else " -- add a frequency / period range and "
               "tick it first"))
        self.map_style.setVisible("fft" in self.orders())
        self.add_btn.setEnabled(bool(self.results))
        self.peaks_btn.setEnabled(bool(self.results))

    def sync(self, redraw=False):
        self.roi.refill()
        self._state()

    def data_loaded(self, d):
        self.results = {}
        self.peaks = {}
        self._fill_peaks()
        self.result.setText("")
        self.sync()

    def open(self):
        self.sync()

    # ---- calculation ------------------------------------------------------ #
    def apply(self):
        d = self.main.data
        if d is None:
            return
        roi = self.roi.currentData(self._role)
        name = self.quantity.currentText()
        orders = self.orders()
        if not orders:
            QtWidgets.QMessageBox.information(
                self, "FFT in time", "Tick at least one order of the "
                "calculation.")
            return
        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(E("Qt.CursorShape.WaitCursor")))
        try:
            fn = self.function.currentData(self._role)
            results = {o: lib.fft_in_time(
                d, roi, name, o, fn, self.seg.value() or None,
                remove_mean=self.remove_mean.isChecked(),
                detrend=self.detrend.isChecked()) for o in orders}
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, "FFT in time", str(exc))
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self.results = results
        had_peaks = bool(self.peaks)
        self.peaks = {}
        lines = []
        for o in orders:
            r = results[o]
            k = int(np.argmax(r["amp"][1:])) + 1 if r["amp"].size > 1 else 0
            psd = r["function"] != "fft"
            lines.append("%s: largest peak (f > 0) %.4g %s, %s %.3g %s" % (
                dict(lib.FFT_ORDERS)[o], r["f"][k], r["funit"],
                "density" if psd else "amplitude", r["amp"][k],
                lib._psd_unit(r["unit"]) if psd else r["unit"]))
        r = results[orders[0]]
        self.result.setText(
            "%s, %s, %s: %d frequencies (df = %.4g %s), %d point(s).\n%s"
            % (r["roi"], name, self.function.currentText(), r["f"].size,
                r["f"][1] if r["f"].size > 1 else 0, r["funit"],
                r["n_points"], "\n".join(lines)))
        self.main.mark_done("fft_apply")
        self.main.write_log("FFT in time: " + self.result.text().replace(
            "\n", "; "))
        if had_peaks:                            # found again on the new FFT
            self.find_peaks(redraw=False)
        else:
            self._fill_peaks()
        self.redraw(force=True)
        if self.main.figures.find(self.STEP, self.MAP) is not None and \
                "fft" in self.results and self._ticked_range():
            self.plot_map()
        self._drop_map_if_no_range()
        self._state()

    def _ticked_range(self):
        return any(w.get("on") for w in self.windows)

    def _drop_map_if_no_range(self):
        """No frequency / period range ticked: no map of the integrated
        amplitude (the figure is closed)."""
        page = self.main.figures.find(self.STEP, self.MAP)
        if page is not None and (not self._ticked_range()
                                 or "fft" not in self.results):
            self.main.figures.remove(page)

    def _restyle_map(self):
        if self.main.figures.find(self.STEP, self.MAP) is not None:
            self.plot_map()

    def find_peaks(self, redraw=True):
        """The n dominant peaks of each order calculated."""
        if not self.results:
            return
        n = self.npeaks.value()
        self.peaks = {o: lib.find_spectral_peaks(r, n, self.windows)
                      for o, r in self.results.items()}
        self.main.mark_done("fft_peaks")
        self._fill_peaks()
        for o, pk in self.peaks.items():
            r = self.results[o]
            self.main.write_log("Peaks (%s, %s): %s" % (
                dict(lib.FFT_ORDERS)[o], r["name"], "; ".join(
                    "#%d %.4g %s (period %.4g %s) %.3g" % (
                        p["rank"], p["f"], r["funit"], p["period"],
                        r["tunit"], p["value"]) for p in pk) or "none"))
        if redraw:
            self.redraw(force=True)

    def _fill_peaks(self):
        t = self.peaks_tree
        t.clear()
        short = {"average": "average first", "fft": "spectral first"}
        for o, pk in self.peaks.items():
            r = self.results.get(o)
            if r is None:
                continue
            unit = (lib._psd_unit(r["unit"]) if r.get("function", "fft")
                    != "fft" else r["unit"])
            for p in pk:
                it = QtWidgets.QTreeWidgetItem([
                    "%d" % p["rank"], short[o],
                    "%.4g %s" % (p["f"], r["funit"]),
                    "%.4g %s" % (p["period"], r["tunit"]),
                    "%.3g %s" % (p["value"], unit)])
                it.setForeground(1, QtGui.QBrush(QtGui.QColor(
                    lib.FFT_COLOURS[o])))
                t.addTopLevelItem(it)
        if not t.topLevelItemCount() and self.peaks:
            t.addTopLevelItem(QtWidgets.QTreeWidgetItem(
                ["", "no peak in the range"]))
        for c in range(5):
            t.resizeColumnToContents(c)

    # ---- figures ---------------------------------------------------------- #
    def redraw(self, force=False):
        d = self.main.data
        if d is None:
            return None
        page = self.main.figures.find(self.STEP, self.FIG)
        if page is None and not force:
            return None
        disp = self.display.currentData(self._role)
        if page is None:
            fig = lib.fft_figure(d, self.res, disp, self.windows,
                                 self.log_amp.isChecked(),
                                 fig=lib.new_figure((9, 6.5)),
                                 peaks=self.peaks, **self._frange())
            page = self.main.figures.add(
                self.STEP, self.FIG, fig,
                data_fn=lambda: lib.fft_data(self.main.data, self.res,
                                             self.windows, self.peaks)
                if self.results else {},
                update_fn=self.apply)
        else:
            lib.fft_figure(d, self.res, disp, self.windows,
                           self.log_amp.isChecked(), fig=page.figure,
                           peaks=self.peaks, **self._frange())
            page.redraw()
        if self._spans:                          # still selecting: again
            self.start_range()
        return page

    def _frange(self):
        """fmin, fmax (Hz; None: auto) of the display; the line below them
        (the periods, the frequencies of the result)."""
        lo = self.fmin.value() or None
        hi = self.fmax.value() or None
        bad = lo is not None and hi is not None and hi <= lo
        r = self.res[0] if self.res else None
        fu = r["funit"] if r else "Hz"
        for sp in (self.fmin, self.fmax):
            sp.setSuffix(" " + fu)
        parts = []
        if r is not None:
            parts.append("spectrum: %.4g .. %.4g %s" % (
                r["f"][1] if r["f"].size > 1 else 0, r["f"][-1], fu))
        if lo or hi:
            parts.append("periods %s .. %s %s" % (
                "%.4g" % (1.0 / hi) if hi and not bad else "auto",
                "%.4g" % (1.0 / lo) if lo else "auto",
                r["tunit"] if r else "s"))
        if bad:
            parts.append("fmax <= fmin: fmax not used")
        self.frange.setText("; ".join(parts))
        return dict(fmin=lo, fmax=None if bad else hi)

    def plot_map(self):
        d = self.main.data
        if d is not None and "fft" in self.results and \
                not self._ticked_range():
            QtWidgets.QMessageBox.information(
                self, "FFT maps", "Add a frequency / period range and tick "
                "it first: there is no map without a ticked range.")
            return
        if d is None or "fft" not in self.results:
            QtWidgets.QMessageBox.information(
                self, "FFT maps", "Tick 'Spectral analysis first then "
                "average over the ROI' and press Apply first.")
            return
        F = self.main.figures
        page = F.find(self.STEP, self.MAP)
        res = self.results["fft"]
        self.main.mark_done("fft_map")
        fig = lib.fft_map_figure(d, res, self.windows,
                                 fig=page.figure if page else None,
                                 **self.map_style.options())
        if page is None:
            F.add(self.STEP, self.MAP, fig,
                  data_fn=lambda: lib.fft_map_data(
                      self.main.data, self.results["fft"], self.windows),
                  update_fn=self.plot_map)
        else:
            page.redraw()
            F.focus(page)

    # ---- frequency ranges ------------------------------------------------- #
    def start_range(self):
        """Add range: drag on one of the plots (left button held)."""
        from matplotlib.widgets import SpanSelector
        page = self.redraw(force=True) if self.main.figures.find(
            self.STEP, self.FIG) is None else \
            self.main.figures.find(self.STEP, self.FIG)
        if page is None or not self.results:
            return
        self._stop_spans()
        toolbar_tool_off(page.toolbar)
        page.drawing = True
        self.main.figures.focus(page)
        for ax in page.figure.axes:
            try:
                sp = SpanSelector(ax, self._range_selected, "horizontal",
                                  useblit=False, interactive=False,
                                  props=dict(facecolor="#2ca02c",
                                             alpha=0.25))
            except TypeError:                    # older matplotlib
                sp = SpanSelector(ax, self._range_selected, "horizontal",
                                  rectprops=dict(facecolor="#2ca02c",
                                                 alpha=0.25))
            self._spans.append(sp)
        self._esc = page.canvas.mpl_connect(
            "key_press_event", lambda ev: self._stop_spans()
            if ev.key == "escape" else None)
        self._span_page = page
        page.canvas.setFocus()
        self.main.status("Drag on a plot of 'FFT in time' to select the "
                         "range (Esc cancels)")

    def _stop_spans(self):
        page = getattr(self, "_span_page", None)
        for sp in self._spans:
            try:
                sp.set_active(False)
                sp.disconnect_events()
            except Exception:
                pass
        self._spans = []
        if page is not None:
            if getattr(self, "_esc", None) is not None:
                page.canvas.mpl_disconnect(self._esc)
            QtCore.QTimer.singleShot(
                300, lambda p=page: setattr(p, "drawing", False))
        self._esc = None
        self._span_page = None

    def _range_selected(self, a, b):
        if b - a <= 0:
            return
        self._stop_spans()
        if self.display.currentData(self._role) == "period":
            a, b = 1.0 / b, 1.0 / a              # periods -> frequencies
        for w in self.windows:                   # the new one is ticked
            w["on"] = False
        self.windows.append(dict(fmin=float(a), fmax=float(b), on=True))
        self._fill_windows()
        self.main.write_log("FFT range W%d: %.4g .. %.4g %s" % (
            len(self.windows), a, b, self.res[0]["funit"] if self.res
            else ""))
        self._ranges_changed()

    def _fill_windows(self):
        t = self.win_tree
        t.blockSignals(True)
        t.clear()
        fu = self.res[0]["funit"] if self.res else "Hz"
        tu = self.res[0]["tunit"] if self.res else "s"
        for k, w in enumerate(self.windows):
            per = ("%.4g .. %.4g %s" % (1.0 / w["fmax"], 1.0 / w["fmin"], tu)
                   if w["fmin"] > 0 else ">= %.4g %s" % (1.0 / w["fmax"], tu)
                   if w["fmax"] > 0 else "--")
            it = QtWidgets.QTreeWidgetItem([
                "W%d" % (k + 1), "%.4g .. %.4g %s" % (w["fmin"], w["fmax"],
                                                       fu), per])
            it.setFlags(it.flags() | E("Qt.ItemFlag.ItemIsUserCheckable"))
            it.setCheckState(0, E("Qt.CheckState.Checked") if w.get("on")
                             else E("Qt.CheckState.Unchecked"))
            it.setToolTip(0, "ticked: the range used (one at a time)")
            t.addTopLevelItem(it)
        for c in range(3):
            t.resizeColumnToContents(c)
        t.blockSignals(False)

    def _win_ticked(self, it, col):
        k = self.win_tree.indexOfTopLevelItem(it)
        if 0 <= k < len(self.windows):
            on = it.checkState(0) == E("Qt.CheckState.Checked")
            t = self.win_tree
            t.blockSignals(True)                 # one ticked at a time --
            for j, w in enumerate(self.windows):  # set in place (no refill
                w["on"] = on if j == k else False  # inside this signal)
                t.topLevelItem(j).setCheckState(
                    0, E("Qt.CheckState.Checked") if w["on"]
                    else E("Qt.CheckState.Unchecked"))
            t.blockSignals(False)
            QtCore.QTimer.singleShot(0, self._ranges_changed)

    def remove_range(self):
        items = self.win_tree.selectedItems()
        if not items:
            self.main.status("Select the range to remove in the list first")
            return
        k = self.win_tree.indexOfTopLevelItem(items[0])
        del self.windows[k]
        self._fill_windows()
        self._ranges_changed()

    def _ranges_changed(self):
        self.redraw()
        self._drop_map_if_no_range()
        if self.main.figures.find(self.STEP, self.MAP) is not None and \
                "fft" in self.results:
            self.plot_map()
        self._state()

    # ---- session ---------------------------------------------------------- #
    def get_state(self):
        return dict(roi=self.roi.currentData(self._role),
                    quantity=self.quantity.currentText(),
                    orders=self.orders(),
                    function=self.function.currentData(self._role),
                    segment=self.seg.value(),
                    npeaks=self.npeaks.value(),
                    remove_mean=self.remove_mean.isChecked(),
                    detrend=self.detrend.isChecked(),
                    map_style=self.map_style.get_state(),
                    display=self.display.currentData(self._role),
                    fmin=self.fmin.value(), fmax=self.fmax.value(),
                    log_amp=self.log_amp.isChecked(),
                    windows=[dict(w) for w in self.windows])

    def set_state(self, s):
        if s.get("quantity"):
            self.quantity.setCurrentText(s["quantity"])
        i = self.function.findData(s.get("function"), self._role)
        if i >= 0:
            self.function.setCurrentIndex(i)
        self.seg.setValue(int(s.get("segment") or 0))
        self.npeaks.setValue(int(s.get("npeaks") or 3))
        if s.get("detrend") is not None:
            self.detrend.setChecked(bool(s["detrend"]))
        if s.get("remove_mean") is not None and not self.detrend.isChecked():
            self.remove_mean.setChecked(bool(s["remove_mean"]))
        self.map_style.set_state(s.get("map_style"))
        orders = s.get("orders") or ([s["order"]] if s.get("order")
                                     else None)
        if orders:
            for k, cb in self.order_box.items():
                cb.setChecked(k in orders)
        i = self.display.findData(s.get("display"), self._role)
        if i >= 0:
            self.display.setCurrentIndex(i)
        self.fmin.setValue(float(s.get("fmin") or 0))
        self.fmax.setValue(float(s.get("fmax") or 0))
        self.log_amp.setChecked(bool(s.get("log_amp")))
        self.windows = [dict(fmin=w["fmin"], fmax=w["fmax"],
                             on=bool(w.get("on")))
                        for w in s.get("windows") or []]
        self._fill_windows()
        self._pending_roi = s.get("roi")

    def apply_pending(self):
        r = getattr(self, "_pending_roi", None)
        if r:
            i = self.roi.findData(r, self._role)
            if i >= 0:
                self.roi.setCurrentIndex(i)
        self._pending_roi = None


class FFTSpacePanel(QtWidgets.QWidget):
    """Spectral analysis > FFT in space: region (Full or an active ROI),
    direction (along x: each row, along z: each column, or along a ticked
    line: its values stored by Define ROI > Apply), quantity, spectral
    function (FFT, power spectrum, periodogram; remove mean, detrend,
    window), time (one frame, or the mean of the spectra of all frames).
    Apply -> figure 'FFT in space' (amplitude, phase, real and imaginary
    parts, or the density). Display: wavenumber k (rad/m), 1/lambda or
    wavelength, kmin / kmax, log scale. Find peaks. k-t diagram (own
    section): the wavenumber k_x, k_z or k along a line (those available),
    the spectrum at every frame -> figure 'k-t diagram'."""

    STEP = "fft_space"
    FIG = "FFT in space"
    KT = "k-t diagram"

    def __init__(self, main):
        super().__init__()
        self.main = main
        self._role = E("Qt.ItemDataRole.UserRole")
        self.res = None
        self.kt_res = None
        self.peaks = []
        lay = QtWidgets.QVBoxLayout(self)
        self.note = QtWidgets.QLabel("Load a run first (File > Load data).")
        self.note.setWordWrap(True)
        lay.addWidget(self.note)

        box = QtWidgets.QGroupBox("Region and direction")
        g = QtWidgets.QGridLayout(box)
        self.roi = RoiCombo(main)
        self.roi.setToolTip(
            "Full field, or an active ROI (its quantities of Define ROI > "
            "Apply are used, not calculated again)")
        self.direction = QtWidgets.QComboBox()
        for key, label in lib.SPACE_DIRECTIONS:
            self.direction.addItem(label, key)
        self.direction.setToolTip(
            "along x: the spectrum of each row of the region, averaged over "
            "the rows (wavenumber k_x); along z: of each column (k_z); along "
            "a line: the values along a ticked line of Define ROI in the "
            "region (wavenumber k along the line)")
        self.line = QtWidgets.QComboBox()
        self.line.setToolTip("a line ticked in Define ROI (its quantities of "
                             "Define ROI > Apply are used)")
        self.line_lab = QtWidgets.QLabel("Line")
        g.addWidget(QtWidgets.QLabel("Region"), 0, 0)
        g.addWidget(self.roi, 0, 1)
        g.addWidget(QtWidgets.QLabel("Direction"), 1, 0)
        g.addWidget(self.direction, 1, 1)
        g.addWidget(self.line_lab, 2, 0)
        g.addWidget(self.line, 2, 1)
        g.setColumnStretch(1, 1)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("FFT quantity")
        g = QtWidgets.QVBoxLayout(box)
        self.quantity = QtWidgets.QComboBox()
        for q in lib.ROI_QUANTITIES:
            self.quantity.addItem(q)
        g.addWidget(self.quantity)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("Spectral function")
        g = QtWidgets.QGridLayout(box)
        self.function = QtWidgets.QComboBox()
        for key, label in lib.SPECTRAL_FUNCTIONS:
            self.function.addItem(label, key)
        g.addWidget(self.function, 0, 0, 1, 2)
        self.seg_lab = QtWidgets.QLabel("Segment length")
        self.seg = QtWidgets.QSpinBox()
        self.seg.setRange(0, 100000)
        self.seg.setSpecialValueText("auto")
        self.seg.setSuffix(" points")
        self.seg.setToolTip("Welch: the number of points of each segment (0 "
                            "= auto: a power of 2 near half the series)")
        g.addWidget(self.seg_lab, 1, 0)
        g.addWidget(self.seg, 1, 1)
        self.win_lab = QtWidgets.QLabel("Window")
        self.window = QtWidgets.QComboBox()
        for key, label in lib.SPACE_WINDOWS:
            self.window.addItem(label, key)
        self.window.setToolTip(
            "FFT, periodogram: the series multiplied by a Hann window before "
            "the transform (less leakage at the ends of short series; FFT "
            "amplitudes corrected by the window's mean); the power spectrum "
            "always uses a Hann window")
        g.addWidget(self.win_lab, 2, 0)
        g.addWidget(self.window, 2, 1)
        self.remove_mean = QtWidgets.QCheckBox("Remove mean")
        self.remove_mean.setChecked(True)
        self.remove_mean.setToolTip("the mean of each series removed first "
                                    "(k = 0 becomes 0)")
        self.detrend = QtWidgets.QCheckBox("Detrend")
        self.detrend.setToolTip("the linear trend of each series (in space) "
                                "removed first -- the mean too")
        g.addWidget(self.remove_mean, 3, 0)
        g.addWidget(self.detrend, 3, 1)
        self.fdesc = QtWidgets.QLabel("")
        self.fdesc.setWordWrap(True)
        self.fdesc.setStyleSheet("color: gray;")
        g.addWidget(self.fdesc, 4, 0, 1, 2)
        g.setColumnStretch(1, 1)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("Time")
        g = QtWidgets.QGridLayout(box)
        self.tmode = QtWidgets.QComboBox()
        self.tmode.addItem("mean over all frames", "mean")
        self.tmode.addItem("single frame", "frame")
        self.tmode.setToolTip("the spectrum of one frame, or the mean of the "
                              "spectra of every frame (amplitudes / "
                              "densities averaged)")
        self.frame = QtWidgets.QSpinBox()
        self.frame.setRange(0, 0)
        self.frame_info = QtWidgets.QLabel("")
        self.frame_info.setStyleSheet("color: gray;")
        g.addWidget(self.tmode, 0, 0, 1, 2)
        g.addWidget(QtWidgets.QLabel("Frame"), 1, 0)
        g.addWidget(self.frame, 1, 1)
        g.addWidget(self.frame_info, 2, 0, 1, 2)
        g.setColumnStretch(1, 1)
        lay.addWidget(box)

        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.setMinimumHeight(34)
        f = self.apply_btn.font(); f.setBold(True)
        self.apply_btn.setFont(f)
        self.apply_btn.setToolTip("calculate the spatial spectrum: figure "
                                  "'FFT in space'")
        lay.addWidget(self.apply_btn)
        self.result = QtWidgets.QLabel("")
        self.result.setWordWrap(True)
        lay.addWidget(self.result)

        box = QtWidgets.QGroupBox("Display")
        g = QtWidgets.QGridLayout(box)
        self.display = QtWidgets.QComboBox()
        self.display.addItem("wavenumber k (rad/m)", "k")
        self.display.addItem("1/λ (1/m)", "inv")
        self.display.addItem("wavelength λ (m)", "lambda")
        self.display.setToolTip("x axis: the wavenumber k = 2 pi / lambda, "
                                "1 / lambda, or the wavelength lambda (log "
                                "scale; k = 0 left out)")

        def kspin(tip):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(0.0, 1.0e7)
            sp.setDecimals(3)
            sp.setSingleStep(10.0)
            sp.setSpecialValueText("auto")
            sp.setSuffix(" rad/m")
            sp.setToolTip(tip + " (0 = auto), as a wavenumber k = 2 pi / "
                          "lambda; also used by the k-t diagram")
            sp.setKeyboardTracking(False)
            return sp
        self.kmin = kspin("the lowest wavenumber shown")
        self.kmax = kspin("the highest wavenumber shown")
        self.log_amp = QtWidgets.QCheckBox("log scale (amplitude / density)")
        self.krange = QtWidgets.QLabel("")
        self.krange.setStyleSheet("color: gray;")
        self.krange.setWordWrap(True)
        g.addWidget(QtWidgets.QLabel("x axis"), 0, 0)
        g.addWidget(self.display, 0, 1)
        g.addWidget(QtWidgets.QLabel("kmin"), 1, 0)
        g.addWidget(self.kmin, 1, 1)
        g.addWidget(QtWidgets.QLabel("kmax"), 2, 0)
        g.addWidget(self.kmax, 2, 1)
        g.addWidget(self.krange, 3, 1)
        g.addWidget(self.log_amp, 4, 1)
        g.setColumnStretch(1, 1)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("Find peaks")
        g = QtWidgets.QGridLayout(box)
        self.npeaks = QtWidgets.QSpinBox()
        self.npeaks.setRange(1, 50)
        self.npeaks.setValue(3)
        self.peaks_btn = QtWidgets.QPushButton("Find peaks")
        self.peaks_btn.setToolTip(
            "the n dominant peaks of the amplitude (FFT) or density: the n "
            "highest local maxima, k = 0 left out, within kmin .. kmax; "
            "marked on the figure")
        g.addWidget(QtWidgets.QLabel("Number of peaks n"), 0, 0)
        g.addWidget(self.npeaks, 0, 1)
        g.addWidget(self.peaks_btn, 0, 2)
        self.peaks_tree = QtWidgets.QTreeWidget()
        self.peaks_tree.setHeaderLabels(["#", "k", "wavelength", "value"])
        self.peaks_tree.setRootIsDecorated(False)
        self.peaks_tree.setMaximumHeight(140)
        g.addWidget(self.peaks_tree, 1, 0, 1, 3)
        g.setColumnStretch(0, 1)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("k-t diagram")
        g = QtWidgets.QGridLayout(box)
        self.kt_comp = QtWidgets.QComboBox()
        self.kt_comp.setToolTip(
            "the wavenumber of the diagram, among those available for the "
            "region: k_x (along x, each row), k_z (along z, each column), "
            "k along each ticked line crossing the region")
        self.kt_log = QtWidgets.QCheckBox("log colour scale")
        self.kt_log.setChecked(True)
        self.kt_cmap = QtWidgets.QComboBox()
        self.kt_cmap.addItems([c for c in lib.MAP_CMAPS if c != "auto"])
        self.kt_btn = QtWidgets.QPushButton("Plot k-t diagram")
        self.kt_btn.setToolTip(
            "the spatial spectrum (quantity, spectral function and options "
            "above) at every frame: wavenumber in x (the x axis of Display, "
            "kmin / kmax), time in y, amplitude / density in colour -- "
            "figure 'k-t diagram'")
        g.addWidget(QtWidgets.QLabel("Wavenumber"), 0, 0)
        g.addWidget(self.kt_comp, 0, 1)
        g.addWidget(QtWidgets.QLabel("Colour map"), 1, 0)
        g.addWidget(self.kt_cmap, 1, 1)
        g.addWidget(self.kt_log, 2, 1)
        g.addWidget(self.kt_btn, 3, 0, 1, 2)
        self.kt_info = QtWidgets.QLabel("")
        self.kt_info.setWordWrap(True)
        self.kt_info.setStyleSheet("color: gray;")
        g.addWidget(self.kt_info, 4, 0, 1, 2)
        g.setColumnStretch(1, 1)
        lay.addWidget(box)
        lay.addStretch(1)

        self.function.currentIndexChanged.connect(self._function_changed)
        self.detrend.toggled.connect(self._detrend_toggled)
        self.direction.currentIndexChanged.connect(lambda *_: self._state())
        self.roi.currentIndexChanged.connect(lambda *_: self._fill_lines())
        self.tmode.currentIndexChanged.connect(lambda *_: self._state())
        self.frame.valueChanged.connect(lambda *_: self._frame_info())
        self.apply_btn.clicked.connect(lambda *_: self.apply())
        self.peaks_btn.clicked.connect(lambda *_: self.find_peaks())
        self.kt_btn.clicked.connect(lambda *_: self.plot_kt())
        for w in (self.display,):
            w.currentIndexChanged.connect(lambda *_: self._redraw_all())
        for w in (self.kmin, self.kmax):
            w.valueChanged.connect(lambda *_: self._redraw_all())
        self.log_amp.toggled.connect(lambda *_: self.redraw())
        self.kt_log.toggled.connect(lambda *_: self.redraw_kt())
        self.kt_cmap.currentIndexChanged.connect(lambda *_: self.redraw_kt())
        self._function_changed()
        self._state()

    # ---- state ------------------------------------------------------------ #
    def _function_changed(self, *_):
        fn = self.function.currentData(self._role) or "fft"
        self.fdesc.setText(lib.SPECTRAL_HELP[fn].replace("Hz", "(1/m)")
                           .replace("over f", "over 1/λ")
                           .replace("frequency", "wavenumber"))
        for w in (self.seg_lab, self.seg):
            w.setVisible(fn == "power")
        for w in (self.win_lab, self.window):
            w.setVisible(fn != "power")
        self.log_amp.setChecked(fn != "fft")

    def _detrend_toggled(self, on):
        if on:
            self._mean_before = self.remove_mean.isChecked()
            self.remove_mean.setChecked(True)
        else:
            self.remove_mean.setChecked(getattr(self, "_mean_before", True))
        self.remove_mean.setEnabled(not on)

    def _lines(self):
        """[(name, rect)] of the lines ticked in Define ROI."""
        rp = self.main.roi_panel
        shown = rp.shown("line")
        return [(n, p) for n, p in rp.shapes("line")
                if shown is None or n in shown]

    def _fill_lines(self):
        keep = self.line.currentText()
        self.line.blockSignals(True)
        self.line.clear()
        for n, _p in self._lines():
            self.line.addItem(n)
        if self.line.findText(keep) >= 0:
            self.line.setCurrentText(keep)
        self.line.blockSignals(False)
        self._fill_kt()

    def _fill_kt(self):
        """The wavenumbers of the k-t diagram available for the region:
        k_x, k_z and k along each ticked line crossing it."""
        d = self.main.data
        keep = self.kt_comp.currentData(self._role)
        cb = self.kt_comp
        cb.blockSignals(True)
        cb.clear()
        if d is not None:
            roi = self.roi.currentData(self._role)
            try:
                vw = lib._roi_view(d, d.can_calibrate, roi)[0]
                ny, nx = vw.x.shape
            except Exception:
                ny = nx = 0
            if nx >= 4:
                cb.addItem("k_x  (along x)", ("x", None))
            if ny >= 4:
                cb.addItem("k_z  (along z)", ("z", None))
            for n, rect in self._lines():
                try:
                    if lib.line_in_roi(d, rect, roi):
                        cb.addItem("k along %s" % n, ("line", n))
                except Exception:
                    continue
        for i in range(cb.count()):
            if cb.itemData(i, self._role) == keep:
                cb.setCurrentIndex(i)
        cb.blockSignals(False)
        self.kt_btn.setEnabled(d is not None and cb.count() > 0)

    def _frame_info(self):
        d = self.main.data
        self.frame_info.setText(d.frame_text(self.frame.value())
                                if d is not None else "")

    def _state(self):
        d = self.main.data
        self.note.setVisible(d is None)
        line = self.direction.currentData(self._role) == "line"
        for w in (self.line_lab, self.line):
            w.setVisible(line)
        self.frame.setEnabled(self.tmode.currentData(self._role) == "frame")
        self.apply_btn.setEnabled(d is not None and (not line or
                                                     self.line.count() > 0))
        self.peaks_btn.setEnabled(self.res is not None)
        self.kt_btn.setEnabled(d is not None and self.kt_comp.count() > 0)

    def sync(self, redraw=False):
        self.roi.refill()
        self._fill_lines()
        self._state()

    def data_loaded(self, d):
        self.res = self.kt_res = None
        self.peaks = []
        self._fill_peaks()
        self.result.setText("")
        self.kt_info.setText("")
        self.frame.setRange(0, d.nt - 1)
        self._frame_info()
        self.sync()

    def open(self):
        self.sync()

    # ---- calculation ------------------------------------------------------ #
    def _options(self):
        return dict(function=self.function.currentData(self._role),
                    nperseg=self.seg.value() or None,
                    remove_mean=self.remove_mean.isChecked(),
                    detrend=self.detrend.isChecked(),
                    window=self.window.currentData(self._role))

    def _calc(self, direction, line, frame):
        d = self.main.data
        roi = self.roi.currentData(self._role)
        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(E("Qt.CursorShape.WaitCursor")))
        try:
            return lib.fft_in_space(d, roi, self.quantity.currentText(),
                                    direction, line, frame=frame,
                                    **self._options())
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def apply(self):
        d = self.main.data
        if d is None:
            return
        direction = self.direction.currentData(self._role)
        line = self.line.currentText() if direction == "line" else None
        frame = ("mean" if self.tmode.currentData(self._role) == "mean"
                 else self.frame.value())
        try:
            r = self._calc(direction, line, frame)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "FFT in space", str(exc))
            return
        self.res = r
        k = int(np.nanargmax(r["amp"][1:])) + 1 if r["amp"].size > 1 else 0
        psd = r["function"] != "fft"
        f1 = r["f"][1] if r["f"].size > 1 else 0
        self.result.setText(
            "%s, %s, %s: %d wavenumbers (dk = %.4g rad/%s, kmax = %.4g "
            "rad/%s), %d %s%s.\n%s\nLargest peak (k > 0): k = %.4g rad/%s "
            "(λ = %.4g %s), %s %.3g %s" % (
                r["roi"], r["quantity"], self.function.currentText(),
                r["f"].size, 2 * np.pi * f1, r["lunit"],
                2 * np.pi * r["f"][-1], r["lunit"], r["n_points"],
                r["points_word"], ", %d left out" % r["n_left_out"]
                if r["n_left_out"] else "", r["desc"],
                2 * np.pi * r["f"][k], r["lunit"],
                1.0 / r["f"][k] if r["f"][k] > 0 else np.inf, r["lunit"],
                "density" if psd else "amplitude", r["amp"][k],
                r["psd_unit"] if psd else r["unit"]))
        self.main.mark_done("fftspace_apply")
        self.main.write_log("FFT in space: " + self.result.text().replace(
            "\n", "; "))
        if self.peaks:
            self.find_peaks(redraw=False)
        self.redraw(force=True)
        self._state()

    def _kwindow(self):
        """kmin / kmax as a range of f (cycles per length) for peaks."""
        lo, hi = self.kmin.value(), self.kmax.value()
        if not (lo or hi):
            return []
        return [dict(fmin=lo / (2 * np.pi), fmax=(hi / (2 * np.pi)) if hi
                     else 1e30, on=True)]

    def find_peaks(self, redraw=True):
        if self.res is None:
            return
        self.peaks = lib.find_spectral_peaks(self.res, self.npeaks.value(),
                                             self._kwindow())
        self.main.mark_done("fftspace_peaks")
        self._fill_peaks()
        r = self.res
        self.main.write_log("FFT in space, peaks (%s): %s" % (
            r["quantity"], "; ".join(
                "#%d k = %.4g rad/%s (λ = %.4g %s) %.3g" % (
                    p["rank"], 2 * np.pi * p["f"], r["lunit"], p["period"],
                    r["lunit"], p["value"]) for p in self.peaks) or "none"))
        if redraw:
            self.redraw(force=True)

    def _fill_peaks(self):
        t = self.peaks_tree
        t.clear()
        r = self.res
        for p in self.peaks if r is not None else []:
            t.addTopLevelItem(QtWidgets.QTreeWidgetItem([
                "%d" % p["rank"],
                "%.4g rad/%s" % (2 * np.pi * p["f"], r["lunit"]),
                "%.4g %s" % (p["period"], r["lunit"]),
                "%.3g %s" % (p["value"], r["psd_unit"]
                             if r["function"] != "fft" else r["unit"])]))
        for c in range(4):
            t.resizeColumnToContents(c)

    def plot_kt(self):
        d = self.main.data
        comp = self.kt_comp.currentData(self._role)
        if d is None or comp is None:
            return
        direction, line = comp
        try:
            self.kt_res = self._calc(direction, line, "mean")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "k-t diagram", str(exc))
            return
        r = self.kt_res
        self.kt_info.setText("%s of %s in %s: %d wavenumbers x %d frames "
                             "(%s)" % (r["kname"], r["quantity"], r["roi"],
                                       r["f"].size, r["amp_t"].shape[0],
                                       r["desc"].split(";")[0]))
        self.main.mark_done("fftspace_kt")
        self.main.write_log("k-t diagram: " + self.kt_info.text())
        self.redraw_kt(force=True)

    # ---- figures ---------------------------------------------------------- #
    def _krange(self):
        lo, hi = self.kmin.value() or None, self.kmax.value() or None
        bad = lo is not None and hi is not None and hi <= lo
        r = self.res
        parts = []
        if r is not None:
            parts.append("spectrum: k = %.4g .. %.4g rad/%s (λ = %.4g "
                         ".. %.4g %s)" % (
                             2 * np.pi * r["f"][1] if r["f"].size > 1 else 0,
                             2 * np.pi * r["f"][-1], r["lunit"],
                             1.0 / r["f"][-1] if r["f"][-1] else np.inf,
                             1.0 / r["f"][1] if r["f"].size > 1 else np.inf,
                             r["lunit"]))
        if bad:
            parts.append("kmax <= kmin: kmax not used")
        self.krange.setText("; ".join(parts))
        return (lo, None if bad else hi)

    def _redraw_all(self):
        self.redraw()
        self.redraw_kt()

    def redraw(self, force=False):
        d = self.main.data
        if d is None:
            return None
        page = self.main.figures.find(self.STEP, self.FIG)
        if page is None and not force:
            self._krange()
            return None
        lo, hi = self._krange()
        kw = dict(fmin=lo / (2 * np.pi) if lo else None,
                  fmax=hi / (2 * np.pi) if hi else None)
        disp = self.display.currentData(self._role)
        res = [self.res] if self.res is not None else None
        if page is None:
            fig = lib.fft_figure(d, res, disp, (), self.log_amp.isChecked(),
                                 fig=lib.new_figure((9, 6.5)),
                                 peaks={"space": self.peaks}, **kw)
            page = self.main.figures.add(
                self.STEP, self.FIG, fig,
                data_fn=lambda: lib.fft_space_data(self.main.data, self.res,
                                                   self.peaks),
                update_fn=self.apply)
        else:
            lib.fft_figure(d, res, disp, (), self.log_amp.isChecked(),
                           fig=page.figure, peaks={"space": self.peaks}, **kw)
            page.redraw()
        return page

    def redraw_kt(self, force=False):
        d = self.main.data
        if d is None:
            return None
        page = self.main.figures.find(self.STEP, self.KT)
        if page is None and not force:
            return None
        lo, hi = self._krange()
        kw = dict(display=self.display.currentData(self._role), kmin=lo,
                  kmax=hi, log=self.kt_log.isChecked(),
                  cmap=self.kt_cmap.currentText())
        if page is None:
            fig = lib.kt_figure(d, self.kt_res, fig=lib.new_figure((8, 6.5)),
                                **kw)
            page = self.main.figures.add(
                self.STEP, self.KT, fig,
                data_fn=lambda: lib.fft_space_data(self.main.data,
                                                   self.kt_res),
                update_fn=self.plot_kt)
        else:
            lib.kt_figure(d, self.kt_res, fig=page.figure, **kw)
            page.redraw()
            self.main.figures.focus(page)
        return page

    # ---- session ---------------------------------------------------------- #
    def get_state(self):
        return dict(roi=self.roi.currentData(self._role),
                    direction=self.direction.currentData(self._role),
                    line=self.line.currentText(),
                    quantity=self.quantity.currentText(),
                    function=self.function.currentData(self._role),
                    segment=self.seg.value(),
                    window=self.window.currentData(self._role),
                    remove_mean=self.remove_mean.isChecked(),
                    detrend=self.detrend.isChecked(),
                    time=self.tmode.currentData(self._role),
                    frame=self.frame.value(),
                    display=self.display.currentData(self._role),
                    kmin=self.kmin.value(), kmax=self.kmax.value(),
                    log_amp=self.log_amp.isChecked(),
                    npeaks=self.npeaks.value(),
                    kt=list(self.kt_comp.currentData(self._role) or []),
                    kt_log=self.kt_log.isChecked(),
                    kt_cmap=self.kt_cmap.currentText())

    def set_state(self, s):
        def pick(cb, val):
            i = cb.findData(val, self._role)
            if i >= 0:
                cb.setCurrentIndex(i)
        pick(self.direction, s.get("direction"))
        if s.get("quantity"):
            self.quantity.setCurrentText(s["quantity"])
        pick(self.function, s.get("function"))
        self.seg.setValue(int(s.get("segment") or 0))
        pick(self.window, s.get("window"))
        if s.get("detrend") is not None:
            self.detrend.setChecked(bool(s["detrend"]))
        if s.get("remove_mean") is not None and not self.detrend.isChecked():
            self.remove_mean.setChecked(bool(s["remove_mean"]))
        pick(self.tmode, s.get("time"))
        pick(self.display, s.get("display"))
        self.kmin.setValue(float(s.get("kmin") or 0))
        self.kmax.setValue(float(s.get("kmax") or 0))
        if s.get("log_amp") is not None:
            self.log_amp.setChecked(bool(s["log_amp"]))
        self.npeaks.setValue(int(s.get("npeaks") or 3))
        if s.get("kt_log") is not None:
            self.kt_log.setChecked(bool(s["kt_log"]))
        if s.get("kt_cmap"):
            self.kt_cmap.setCurrentText(s["kt_cmap"])
        self._pending = s

    def apply_pending(self):
        s = getattr(self, "_pending", None)
        self._pending = None
        if not s:
            return
        i = self.roi.findData(s.get("roi"), self._role)
        if i >= 0:
            self.roi.setCurrentIndex(i)
        self._fill_lines()
        if s.get("line") and self.line.findText(s["line"]) >= 0:
            self.line.setCurrentText(s["line"])
        if s.get("frame") is not None:
            self.frame.setValue(int(s["frame"]))
        kt = tuple(s.get("kt") or ())
        for i in range(self.kt_comp.count()):
            if tuple(self.kt_comp.itemData(i, self._role)) == kt:
                self.kt_comp.setCurrentIndex(i)
        self._state()


SCRIPT_TEMPLATE = '''"""A post-processing script for the PIV GUI (Advanced > Python script).

Run with Execute on the run loaded. Names available:
  d            the loaded run (PIVData): d.run, d.run_dir, d.nt, d.t,
               d.X_px, d.Z_px, d.U_px, d.V_px (px, px/frame), d.params ...
  np, lib      numpy, piv_gui_lib
  view(roi=None, calibrated=True)
               the run as shown (lib.RunView): .x, .z (m), .t (s),
               .uv() -> (u, v) for all frames (m/s), .uv(i) one frame;
               roi: None / 'Full' or an ROI field ('Validated_ROI01')
  quantity(name, roi=None)
               a quantity for every frame (ny, nx, nt): 'U', 'V',
               'velocity norm', 'divergence', 'vorticity', 'kinetic energy'
               (an ROI: its stored quantity of Define ROI > Apply)
  line(name, quantity, roi=None)
               the values along a line of Define ROI: dict(values
               (npts, nt), s, x, z, unit)
  variables()  every variable of Show all variables, as a dict
  results      a dict: put here what the script produces -- it is kept,
               listed in Show / Save variables, saved with the session
  figure(name) a new matplotlib figure shown in the GUI (under
               Advanced > Python script)
  log(text)    a line in the log; print() goes to the output too
  run_dir, post_dir   the run folder and its PostProcessing folder
  roi_field(name)     'ROI01' -> its field name ('Validated_ROI01')
  gui, G       the main window, the GUI module (its classes: RoiCombo,
               MovieBox, ...); QtWidgets, QtCore, QtGui
  add_advanced_item(id, label, panel)
               a new item of the Advanced menu with its own panel
See Claude_for_Developer.md (with the GUI) and Example_Scripts/.
"""

# example: the time mean of the vorticity of the full field and its
# spatial mean in time
w = quantity("vorticity")
results["vorticity_time_mean"] = np.nanmean(w, axis=2)
results["vorticity_space_mean"] = np.nanmean(w, axis=(0, 1))

fig = figure("Mean vorticity")
ax = fig.add_subplot(111)
vw = view()
ax.plot(vw.t, results["vorticity_space_mean"])
ax.set_xlabel("time (%s)" % vw.tunit)
ax.set_ylabel("mean vorticity (1/s)")
log("vorticity: time mean map and spatial mean series computed")
'''


class ScriptPanel(QtWidgets.QWidget):
    """Advanced > Python script: a Python script of the user (a .py file)
    run on the loaded run. The script gets the run and helpers (see
    SCRIPT_TEMPLATE) and puts what it produces in `results`: kept in
    memory, listed in Show / Save variables, saved with the session (a
    .scripts.npz next to the session file) and executed again when a
    session is loaded or applied to another run. The run information has
    a section 'Advanced: Python scripts' listing the scripts executed."""

    STEP = "python_script"

    def __init__(self, main):
        super().__init__()
        self.main = main
        self._role = E("Qt.ItemDataRole.UserRole")
        self._extra = []
        lay = QtWidgets.QVBoxLayout(self)
        self.note = QtWidgets.QLabel("Load a run first (File > Load data).")
        self.note.setWordWrap(True)
        lay.addWidget(self.note)

        box = QtWidgets.QGroupBox("Script")
        g = QtWidgets.QGridLayout(box)
        self.script = QtWidgets.QComboBox()
        self.script.setSizeAdjustPolicy(E(
            "QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon"))
        self.script.setToolTip("the .py scripts of the root folder, of its "
                               "'scripts' folder and of the GUI's "
                               "Example_Scripts folder, and those opened")
        self.open_btn = QtWidgets.QPushButton("Open...")
        self.open_btn.setToolTip("choose a Python script (.py) anywhere")
        self.new_btn = QtWidgets.QPushButton("New...")
        self.new_btn.setToolTip("write a template script (with the names "
                                "available and an example) and select it")
        g.addWidget(self.script, 0, 0, 1, 3)
        g.addWidget(self.open_btn, 1, 0)
        g.addWidget(self.new_btn, 1, 1)
        g.setColumnStretch(2, 1)
        self.path_lab = QtWidgets.QLabel("")
        self.path_lab.setWordWrap(True)
        self.path_lab.setStyleSheet("color: gray;")
        g.addWidget(self.path_lab, 2, 0, 1, 3)
        self.preview = QtWidgets.QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setFont(QtGui.QFontDatabase.systemFont(
            E("QFontDatabase.SystemFont.FixedFont")))
        self.preview.setMinimumHeight(160)
        self.preview.setToolTip("the script (read only: edit it in your "
                                "editor; it is read again at Execute)")
        g.addWidget(self.preview, 3, 0, 1, 3)
        lay.addWidget(box)

        self.exec_btn = QtWidgets.QPushButton("Execute")
        self.exec_btn.setMinimumHeight(34)
        f = self.exec_btn.font(); f.setBold(True)
        self.exec_btn.setFont(f)
        self.exec_btn.setToolTip(
            "run the script on the loaded run (read again from the file): "
            "what it puts in `results` is kept, listed in Show / Save "
            "variables and saved with the session")
        lay.addWidget(self.exec_btn)

        box = QtWidgets.QGroupBox("Output")
        v = QtWidgets.QVBoxLayout(box)
        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QtGui.QFontDatabase.systemFont(
            E("QFontDatabase.SystemFont.FixedFont")))
        self.output.setMaximumBlockCount(5000)
        self.output.setMinimumHeight(110)
        v.addWidget(self.output)
        self.res_tree = QtWidgets.QTreeWidget()
        self.res_tree.setHeaderLabels(["result", "format", "size"])
        self.res_tree.setRootIsDecorated(True)
        self.res_tree.setMinimumHeight(110)
        self.res_tree.setToolTip("what the scripts produced (results), by "
                                 "script; also in View > Show all variables "
                                 "(script_results)")
        v.addWidget(self.res_tree)
        self.clear_btn = QtWidgets.QPushButton("Remove the results of this "
                                               "script")
        v.addWidget(self.clear_btn)
        lay.addWidget(box)
        lay.addStretch(1)

        self.script.currentIndexChanged.connect(lambda *_: self._show())
        self.open_btn.clicked.connect(lambda *_: self.open_script())
        self.new_btn.clicked.connect(lambda *_: self.new_script())
        self.exec_btn.clicked.connect(lambda *_: self.execute())
        self.clear_btn.clicked.connect(lambda *_: self.clear_results())
        self.fill()

    # ---- scripts ---------------------------------------------------------- #
    def found(self):
        root = self.main.load_panel.root_dir
        out = []
        here = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "Example_Scripts")    # with the GUI
        for d in ([root, os.path.join(root, "scripts")] if root else []) + \
                [here]:
            try:
                out += [os.path.join(d, f) for f in sorted(os.listdir(d))
                        if f.endswith(".py") and not f.startswith(".")]
            except OSError:
                continue
        return out

    def fill(self, select=None):
        keep = select or self.script.currentData(self._role)
        paths = self.found()
        paths += [p for p in self._extra if p not in paths]
        cb = self.script
        cb.blockSignals(True)
        cb.clear()
        for p in paths:
            cb.addItem(os.path.basename(p), p)
            cb.setItemData(cb.count() - 1, p,
                           E("Qt.ItemDataRole.ToolTipRole"))
        if not paths:
            cb.addItem("(no script found -- Open... or New...)", None)
        if keep in paths:
            cb.setCurrentIndex(paths.index(keep))
        cb.blockSignals(False)
        self._show()

    def select(self, path):
        if path and path not in self._extra:
            self._extra.append(path)
        self.fill(select=path)

    def _show(self):
        p = self.script.currentData(self._role)
        self.path_lab.setText(p or "")
        try:
            with open(p) as fh:
                self.preview.setPlainText(fh.read())
        except Exception:
            self.preview.setPlainText("")
        self._state()
        self._fill_results()

    def open_script(self):
        start = os.path.dirname(self.script.currentData(self._role) or "") \
            or self.main.setup_folder()
        path, _f = QtWidgets.QFileDialog.getOpenFileName(
            self, "Python script", start,
            "Python script (*.py);;All files (*)")
        if path:
            self.select(path)

    def new_script(self):
        path, _f = QtWidgets.QFileDialog.getSaveFileName(
            self, "New Python script", os.path.join(
                self.main.setup_folder(), "my_processing.py"),
            "Python script (*.py)")
        if not path:
            return
        if not path.endswith(".py"):
            path += ".py"
        try:
            with open(path, "w") as fh:
                fh.write(SCRIPT_TEMPLATE)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "New script", str(exc))
            return
        self.main.write_log("Template script written: %s" % path)
        self.select(path)

    def _state(self):
        d = self.main.data
        self.note.setVisible(d is None)
        self.exec_btn.setEnabled(d is not None and bool(
            self.script.currentData(self._role)))
        self.clear_btn.setEnabled(self._key() in self.main.script_results)

    def _key(self, path=None):
        p = path or self.script.currentData(self._role) or ""
        return os.path.splitext(os.path.basename(p))[0]

    def open(self):
        self.fill()

    def data_loaded(self, d):
        self._state()
        self._fill_results()

    # ---- execution -------------------------------------------------------- #
    def execute(self, path=None):
        """Run a script on the loaded run. Returns True when it ran
        without error."""
        import io
        import contextlib
        import datetime
        import traceback
        main = self.main
        d = main.data
        path = path or self.script.currentData(self._role)
        if d is None or not path:
            return False
        name = self._key(path)
        results = {}
        lines = []

        def log(text):
            lines.append(str(text))
            main.write_log("Script %s: %s" % (name, text))

        def view(roi=None, calibrated=True):
            return lib._roi_view(d, calibrated and d.can_calibrate,
                                 roi_field(roi))[0]

        def quantity(qname, roi=None):
            return lib._quantity_cube(d, roi_field(roi), qname)[0]

        def line(lname, qname, roi=None):
            roi = roi_field(roi)
            lines_ = dict(lib.parse_shapes(d.params.get("LINES"), "LINE"))
            p = lib.line_profile(d, lines_[lname], qname, 0,
                                 d.can_calibrate, None, roi, line=lname)
            e = lib._stored_line(d, lines_[lname], qname, roi,
                                 d.can_calibrate)
            return dict(values=None if e is None else e["data"],
                        s=p["s"], x=p["x"], z=p["z"], unit=p["unit"])

        def figure(fname):
            fig = lib.new_figure()
            page = main.figures.find(self.STEP, fname)
            if page is not None:
                main.figures.remove(page)
            self._figs.append((fname, fig))
            return fig
        self._figs = []
        def roi_field(name):
            """'ROI01' -> its field name ('Validated_ROI01'); 'Full' and
            field names are returned as they are."""
            if name in (None, "", "Full"):
                return "Full"
            for fname, info in d.roi_fields().items():
                if name in (fname, info["roi"]):
                    return fname
            raise ValueError("%s is not an ROI of Define ROI (applied: %s)"
                             % (name, ", ".join(i["roi"] for i in
                                                d.roi_fields().values())))
        import sys as _sys
        ns = dict(d=d, np=np, lib=lib, view=view, quantity=quantity,
                  roi_field=roi_field, gui=main,
                  G=_sys.modules[type(main).__module__],
                  QtWidgets=QtWidgets, QtCore=QtCore, QtGui=QtGui,
                  add_advanced_item=main.add_advanced_item,
                  line=line, variables=lambda: main._variables()[0],
                  results=results, figure=figure, log=log,
                  run_dir=d.run_dir, post_dir=getattr(d, "post_dir", None),
                  __name__="__piv_script__", __file__=path)
        buf = io.StringIO()
        t0 = time.time()
        main.write_log("Script %s: executing %s" % (name, path))
        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(E("Qt.CursorShape.WaitCursor")))
        ok, err = True, ""
        try:
            with open(path) as fh:
                code = fh.read()
            with contextlib.redirect_stdout(buf), \
                    contextlib.redirect_stderr(buf):
                exec(compile(code, path, "exec"), ns)
        except Exception:
            ok = False
            err = traceback.format_exc(limit=-6)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        dt = time.time() - t0
        out = buf.getvalue()
        self.output.setPlainText(
            out + ("\n".join(lines) + "\n" if lines and not out else "")
            + ("\nERROR:\n" + err if err else "")
            + "\n-- %s in %.2f s, %d result(s): %s" % (
                "done" if ok else "FAILED", dt, len(results),
                ", ".join(results) or "none"))
        if out.strip():
            main.write_log("Script %s output:\n%s" % (name, out.rstrip()))
        if not ok:
            main.write_log("Script %s FAILED:\n%s" % (name, err.rstrip()),
                           error=True)
        for fname, fig in self._figs:            # the figures of the script
            main.figures.add(self.STEP, fname, fig)
        if results or ok:
            main.script_results[name] = results
        main.script_runs = [r for r in main.script_runs
                            if r["script"] != name]
        main.script_runs.append(dict(
            script=name, path=path, ok=ok,
            time=datetime.datetime.now().isoformat(timespec="seconds"),
            seconds=round(dt, 3), results=sorted(results),
            figures=[f for f, _g in self._figs],
            error=err.strip().splitlines()[-1] if err else ""))
        if ok:
            main.mark_done("script:" + path)
        main.show_info(d)                        # the run information
        main._refresh_save_variables()
        main.status("Script %s %s (%.2f s): %d result(s)" % (
            name, "done" if ok else "FAILED", dt, len(results)), log=True)
        self._fill_results()
        self._state()
        return ok

    def clear_results(self):
        name = self._key()
        self.main.script_results.pop(name, None)
        self.main.script_runs = [r for r in self.main.script_runs
                                 if r["script"] != name]
        self.main.journal = [a for a in self.main.journal
                             if not (a.startswith("script:") and
                                     self._key(a[7:]) == name)]
        self.main.write_log("Script %s: results removed" % name)
        if self.main.data is not None:
            self.main.show_info(self.main.data)
        self._fill_results()
        self._state()

    def _fill_results(self):
        import gui_variables as gv
        t = self.res_tree
        t.clear()
        for name, res in self.main.script_results.items():
            top = QtWidgets.QTreeWidgetItem([name, "script", ""])
            for k, v in res.items():
                fmt, size = gv.describe(v)
                QtWidgets.QTreeWidgetItem(top, [k, fmt, size])
            t.addTopLevelItem(top)
            top.setExpanded(True)
        for c in range(3):
            t.resizeColumnToContents(c)

    # ---- session ---------------------------------------------------------- #
    def get_state(self):
        return dict(script=self.script.currentData(self._role),
                    opened=list(self._extra))

    def set_state(self, s):
        self._extra = [p for p in s.get("opened") or [] if os.path.isfile(p)]
        self.fill(select=s.get("script"))


class PlaceholderPanel(QtWidgets.QWidget):
    def __init__(self, label):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        t = QtWidgets.QLabel("<h3>%s</h3><p>Not available in version 1.</p>"
                             "<p>This panel will hold the parameters of "
                             "the step; its figures will appear in the "
                             "list on the right under <i>%s</i>.</p>"
                             % (label, label))
        t.setWordWrap(True)
        lay.addWidget(t)
        lay.addStretch(1)


# --------------------------------------------------------------------------- #
#  Calibration
# --------------------------------------------------------------------------- #
def toolbar_tool_off(toolbar):
    """Switch off the zoom / pan tool of a matplotlib toolbar (the view is
    kept). matplotlib names the modes 'zoom rect' and 'pan/zoom'."""
    mode = str(getattr(toolbar, "mode", "") or "")
    if mode.startswith("pan"):
        toolbar.pan()                    # toggles pan off
    elif mode.startswith("zoom"):
        toolbar.zoom()                   # toggles zoom off


class SegmentEditor(object):
    """The calibration segment on the calibration image, editable while
    the selection is on (button 'Select') and no
    toolbar tool (zoom / pan) is on:
      - hold the left button and drag: draws a new segment (start where the
        button is pressed, end where it is released);
      - press on an end of the segment (within PICK screen pixels) and
        drag: moves that end.
    on_change(points) is called at every mouse move (live length) and
    on_release(points) when the button is released. The view (zoom) is
    never changed."""

    PICK = 10                                    # screen px

    def __init__(self, main, on_change, on_release):
        self.main, self.on_change, self.on_release = main, on_change, \
            on_release
        self.points = None                       # [(x, y), (x, y)] in px
        self.page = self.ax = None
        self.line = self.ends = None
        self.cids = []
        self.drag = None                         # 0 / 1: end being moved
        self.active = False                      # selection button on
        self._new = False
        self._before = None

    # ---- figure ----------------------------------------------------------- #
    def attach(self, page):
        """Connect to the page's (newly drawn) calibration image."""
        self.detach()
        self.page, self.ax = page, page.figure.axes[0]
        self.line, = self.ax.plot([], [], "-", color="red", lw=1.6,
                                  zorder=5, scalex=False, scaley=False)
        self.ends, = self.ax.plot([], [], "o", ms=8, mfc="none", mec="red",
                                  mew=1.6, zorder=6, scalex=False,
                                  scaley=False)
        c = page.canvas
        self.cids = [c.mpl_connect("button_press_event", self._press),
                     c.mpl_connect("motion_notify_event", self._move),
                     c.mpl_connect("button_release_event", self._release)]
        self._show()

    def detach(self):
        if self.page is not None:
            for cid in self.cids:
                try:
                    self.page.canvas.mpl_disconnect(cid)
                except Exception:
                    pass
        self.cids = []
        for art in (self.line, self.ends):
            try:
                art.remove()
            except Exception:
                pass
        self.line = self.ends = None
        self.drag = None

    def set_points(self, points):
        self.points = None if not points else \
            [tuple(map(float, p)) for p in points[:2]]
        self._show()

    def _show(self):
        if self.line is None:
            return
        if self.points:
            p = np.asarray(self.points, float)
            self.line.set_data(p[:, 0], p[:, 1])
            self.ends.set_data(p[:, 0], p[:, 1])
        else:
            self.line.set_data([], [])
            self.ends.set_data([], [])
        self.page.canvas.draw_idle()

    # ---- mouse ------------------------------------------------------------ #
    def _tool_on(self):
        tb = self.page.toolbar
        return bool(getattr(tb, "mode", "")) if tb is not None else False

    def _near(self, ev):
        """The end of the segment under the mouse (None if none)."""
        if not self.points:
            return None
        xy = self.ax.transData.transform(np.asarray(self.points, float))
        d = np.hypot(xy[:, 0] - ev.x, xy[:, 1] - ev.y)
        i = int(np.argmin(d))
        return i if d[i] <= self.PICK else None

    def _press(self, ev):
        if not self.active or ev.button != 1 or ev.inaxes is not self.ax \
                or ev.xdata is None:
            return
        if self._tool_on():
            self.main.status("The toolbar zoom / pan tool is on: click its "
                             "button again to switch it off, then draw", 0)
            return
        self._before = self.points
        i = self._near(ev)
        if i is not None:                        # move an end
            self.drag, self._new = i, False
        else:                                    # new segment
            p = (float(ev.xdata), float(ev.ydata))
            self.points = [p, p]
            self.drag, self._new = 1, True
        self._show()
        self.on_change(self.points)

    def _move(self, ev):
        if self.drag is None or ev.inaxes is not self.ax or ev.xdata is None:
            return
        pts = list(self.points)
        pts[self.drag] = (float(ev.xdata), float(ev.ydata))
        self.points = pts
        self._show()
        self.on_change(self.points)

    def _release(self, ev):
        if self.drag is None or ev.button != 1:
            return
        self._move(ev)
        self.drag = None
        if self._new:
            # a click without dragging keeps the previous segment
            xy = self.ax.transData.transform(np.asarray(self.points, float))
            if np.hypot(*(xy[1] - xy[0])) < 3:
                self.points = self._before
        self._new = False
        self._show()
        self.on_change(self.points)
        self.on_release(self.points)


class CalibrationPanel(QtWidgets.QWidget):
    """Pre-processing > Calibration: calibration image; segment drawn on it
    (left button held) whose ends can be moved; its length in px, the real
    length in mm -> XSCALE = YSCALE [m/px]; the time base (from the log, or
    typed when the run has no log). Apply: used for the loaded run;
    Apply: used for the loaded run (in memory); Save: the calibration file
    (.pivcalib.json) that Load calibration and File > Load data run the
    step from."""

    STEP = "calibration"
    FIG = "Calibration image"

    def __init__(self, main):
        super().__init__()
        self.main = main
        self._role = E("Qt.ItemDataRole.UserRole")
        self.scale = None                        # applied m/px
        self.seg_m = None                        # applied length [m]
        self._extra = []
        self._manual_typed = False
        self.editor = SegmentEditor(main, self._segment_changed,
                                    self._segment_released)
        lay = QtWidgets.QVBoxLayout(self)

        self.cal_load_btn = QtWidgets.QPushButton("Load calibration...")
        self.cal_load_btn.setMinimumHeight(32)
        self.cal_load_btn.setToolTip(
            "read a calibration file (.pivcalib.json, saved with Save below) "
            "and run this step again with it: image, segment, length in mm "
            "and time base set, then Apply (File > Load data can also load "
            "it with the run)")
        self.cal_load_btn.clicked.connect(lambda: self.load_calibration())
        lay.addWidget(self.cal_load_btn)
        self.cal_file_label = QtWidgets.QLabel("")
        self.cal_file_label.setWordWrap(True)
        self.cal_file_label.setStyleSheet("color: gray;")
        self.cal_file_label.setVisible(False)
        lay.addWidget(self.cal_file_label)

        box = QtWidgets.QGroupBox("Calibration image")
        vb = QtWidgets.QVBoxLayout(box)
        g = QtWidgets.QHBoxLayout()
        vb.addLayout(g)
        rrow = QtWidgets.QHBoxLayout()
        rrow.addWidget(QtWidgets.QLabel("Rotation"))
        self.rotate = QtWidgets.QSpinBox()
        self.rotate.setRange(-90, 180)
        self.rotate.setSingleStep(90)
        self.rotate.setWrapping(True)
        self.rotate.setSuffix(" deg")
        self.rotate.setKeyboardTracking(False)
        self.rotate.setToolTip(
            "rotation of the PIV fields and of the images, in deg: 0, 90, "
            "180 or -90 (another value goes to the nearest one; 270 = -90). "
            "Changing it loads the run again rotated; it is shown in the run "
            "information and kept in the calibration file (Save)")
        rrow.addWidget(self.rotate)
        rrow.addStretch(1)
        vb.addLayout(rrow)
        self.rotate.valueChanged.connect(lambda *_: self._rotate_changed())
        self.image = QtWidgets.QComboBox()
        self.image.setSizeAdjustPolicy(E(
            "QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon"))
        self.image.setToolTip("image files and background*.mat of the run "
                              "folder and of the root folder (files named "
                              "*calib* first); Browse... for any other file")
        browse = QtWidgets.QPushButton("Browse...")
        browse.setMaximumWidth(90)
        g.addWidget(self.image, 1)
        g.addWidget(browse)
        lay.addWidget(box)
        self.image.activated.connect(lambda *_: self.image_chosen())
        browse.clicked.connect(self.browse)

        self.select_btn = QtWidgets.QPushButton("Draw segment")
        self.select_btn.setCheckable(True)
        self.select_btn.setMinimumHeight(34)
        self.select_btn.setToolTip(
            "starts the selection on the calibration image (zoom kept, "
            "toolbar zoom / pan switched off): hold the left button and "
            "drag to draw the segment; drag one of its ends to move it. "
            "Click again to stop.")
        self.reset_btn = QtWidgets.QPushButton("Reset")
        self.reset_btn.setMinimumHeight(34)
        self.reset_btn.setToolTip(
            "delete the segment from the figure and clear its length in px "
            "and in mm (nothing else changes)")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.select_btn, 3)
        row.addWidget(self.reset_btn, 1)
        lay.addLayout(row)
        self.select_btn.toggled.connect(self.select)
        self.reset_btn.clicked.connect(self.reset)

        box = QtWidgets.QGroupBox("Spatial calibration (px -> m)")
        f = QtWidgets.QFormLayout(box)
        f.setFieldGrowthPolicy(E(
            "QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow"))
        self.len_label = QtWidgets.QLabel("-- (draw the segment)")
        self.len_label.setTextInteractionFlags(
            E("Qt.TextInteractionFlag.TextSelectableByMouse"))
        self.ends_label = QtWidgets.QLabel("")
        self.ends_label.setWordWrap(True)
        self.mm = QtWidgets.QDoubleSpinBox()
        self.mm.setDecimals(3)
        self.mm.setRange(0.0, 1e6)
        self.mm.setSingleStep(1.0)
        self.mm.setSpecialValueText("--")        # 0 = not given
        self.mm.setSuffix(" mm")
        self.mm.setKeyboardTracking(True)
        self.mm.setToolTip("real length of the segment, in millimetres")
        self.preview = QtWidgets.QLabel("--")
        self.preview.setWordWrap(True)
        f.addRow("Segment length", self.len_label)
        f.addRow("", self.ends_label)
        f.addRow("Length in mm", self.mm)
        f.addRow("Scale", self.preview)
        row = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.setToolTip(
            "compute XSCALE = YSCALE from the segment and use it for the "
            "loaded run (in memory: nothing is written; Save keeps the "
            "calibration in a calibration file)")
        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.setToolTip(
            "save the calibration in a calibration file (.pivcalib.json): "
            "everything needed to run this step again with Load "
            "calibration (image, segment, lengths in px and mm, scale, "
            "time base)")
        for b in (self.apply_btn, self.save_btn):
            b.setMinimumHeight(32)
            row.addWidget(b)
        f.addRow(row)
        self.cal_label = QtWidgets.QLabel("--")
        self.cal_label.setWordWrap(True)
        f.addRow("Applied", self.cal_label)
        lay.addWidget(box)
        self.mm.valueChanged.connect(lambda *_: self._update_preview())
        self.apply_btn.clicked.connect(lambda: self.apply())
        self.save_btn.clicked.connect(lambda: self.save_calibration())

        box = QtWidgets.QGroupBox("Time base")
        f = QtWidgets.QFormLayout(box)
        f.setFieldGrowthPolicy(E(
            "QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow"))
        self.manual = QtWidgets.QCheckBox("Manual")
        self.manual.setToolTip(
            "set the time base by hand instead of using the log (as for a "
            "run without a log); always on when the run has no log. "
            "Unticking it goes back to the log at once. The time base is "
            "kept in memory, not in the parameter file")
        self.fpf = QtWidgets.QSpinBox()
        self.fpf.setRange(1, 100)
        self.fpf.setValue(2)
        self.fpf.setToolTip("camera frames per PIV field (PIVlab pairs: 2); "
                            "PIV field rate = camera fps / frames per field")
        self.dt = QtWidgets.QLineEdit()
        self.cam = QtWidgets.QLineEdit()
        self.dt.setToolTip("pulse separation [s]: time between frames A and "
                           "B of a PIV pair; converts px/frame to px/s")
        self.cam.setToolTip("camera frame rate [Hz]; the time between PIV "
                            "fields is TSCALE = frames per field / camera "
                            "fps (timestamps of the fields)")
        self.tb_label = QtWidgets.QLabel("Load a run first.")
        self.tb_label.setWordWrap(True)
        self.tb_save = QtWidgets.QPushButton("Apply")
        self.tb_save.setMinimumHeight(32)
        self.tb_save.setToolTip(
            "Manual only: use these values (kept in memory, not saved in "
            "the parameter file) and load the run again with them")
        f.addRow(self.manual)
        f.addRow("Frames per PIV field", self.fpf)
        f.addRow("Pulse separation in s", self.dt)
        f.addRow("Camera FPS in Hz", self.cam)
        f.addRow(self.tb_label)
        f.addRow(self.tb_save)
        self.manual.toggled.connect(self._manual_toggled)
        self.cam.textChanged.connect(lambda *_: self._tb_hint())
        self.fpf.valueChanged.connect(lambda *_: self._tb_hint())
        self.tb_save.clicked.connect(self.apply_time)
        lay.addWidget(box)

        self._last_apply = None                  # "apply" / "apply_and_save"
        lay.addStretch(1)
        self.refresh()

    # ---- state shown --------------------------------------------------- #
    def refresh(self):
        """Called when the panel is opened and after a Load (the step
        starts empty: a calibration comes from Apply, Load calibration or
        the calibration file of Load data)."""
        self.fill_images()
        d = self.main.data
        if d is not None:
            self.set_rotation(d.params.get("ROTATE") or 0)
            self.manual.blockSignals(True)
            self.manual.setChecked(d.acq.source == "manual"
                                   or not d.acq.log_ok)
            self.fpf.blockSignals(True)
            self.fpf.setValue(int(d.acq.frames_per_field))
            self.fpf.blockSignals(False)
            self.manual.blockSignals(False)
            self._manual_typed = False
        if d is not None and d.can_calibrate:
            self.cal_label.setText("<b>XSCALE = %.6g, YSCALE = %.6g m/px</b>"
                                   " (loaded run%s)" % (
                                       d.params["XSCALE"], d.params["YSCALE"],
                                       "" if d.calibrated
                                       else ", shown uncalibrated"))
        elif d is not None:
            self.cal_label.setText("loaded run not calibrated (px)")
        else:
            self.cal_label.setText("--")
        self._time_fields()
        self._segment_changed(self.editor.points)

    # ---- time base ------------------------------------------------------- #
    def _time_fields(self):
        """Fill the Time base box from the loaded run; the two fields are
        editable when Manual is ticked (always ticked without a log)."""
        d = self.main.data
        m = self.manual
        if d is None:
            m.blockSignals(True)
            m.setChecked(False)
            m.blockSignals(False)
            for w in (m, self.fpf, self.dt, self.cam):
                w.setEnabled(False)
            self.tb_label.setText("Load a run first (File > Load data).")
            self._update_buttons()
            return
        has_log = d.acq.log_ok
        if not has_log and not m.isChecked():
            m.blockSignals(True)
            m.setChecked(True)
            m.blockSignals(False)
        m.setEnabled(has_log)
        manual = m.isChecked()
        if not manual:                           # the log's values
            log = lib.read_acquisition_log(
                d.log_path or os.path.join(d.run_dir,
                                           d.params["LOG_FILENAME"]),
                self.main.time_base["frames_per_field"])
            self.dt.setText("%.6g" % log.dt)
            self.cam.setText("%.6g" % log.cam_fps)
            self.fpf.blockSignals(True)
            self.fpf.setValue(int(self.main.time_base["frames_per_field"]))
            self.fpf.blockSignals(False)
        elif not self._manual_typed:
            self.dt.setText("%.6g" % d.acq.dt)
            self.cam.setText("%.6g" % d.acq.cam_fps)
        for w in (self.fpf, self.dt, self.cam):
            w.setEnabled(manual)
        self._manual_typed = manual
        self._tb_hint()
        self._update_buttons()

    def _tb_hint(self):
        d = self.main.data
        if d is None:
            return
        fpf = int(self.fpf.value())
        try:
            fps = float(self.cam.text().replace(",", "."))
            hz = ("TSCALE = %d / camera fps = %.6g s (PIV fields at %.6g "
                  "Hz). " % (fpf, fpf / fps, fps / fpf))
        except (ValueError, ZeroDivisionError):
            hz = ""
        if not d.acq.log_ok:
            txt = ("<b>No log file for this run:</b> enter the values, "
                   "then Apply (kept in memory, not in the parameter "
                   "file).")
        elif self.manual.isChecked():
            txt = ("Manual: the values typed are used instead of the log "
                   "once applied (kept in memory, not in the parameter "
                   "file); untick Manual to go back to the log.")
        else:
            txt = "From the log file %s." % os.path.basename(d.log_path or "")
        self.tb_label.setText(hz + txt)

    def open(self):
        """The panel is opened: show the calibration image if it is not
        shown yet."""
        self.refresh()
        if self.main.figures.find(self.STEP, self.FIG) is None and \
                self.image.currentData(self._role):
            self.show_image()

    def _new_scale(self):
        """m/px from the segment and the length in mm (None if either is
        missing)."""
        px = lib.segment_length(self.editor.points or [])
        mm = self.mm.value()
        if px <= 0 or mm <= 0:
            return None
        return float("%.10g" % (mm * 1e-3 / px))

    def _update_preview(self):
        s = self._new_scale()
        if s is None:
            self.preview.setText("-- (draw the segment and give its length)")
        else:
            self.preview.setText("%.6g m/px   (%.4g px/mm)"
                                 % (s, 1e-3 / s))
        self._update_buttons()

    def _update_buttons(self):
        ok = self._new_scale() is not None
        self.apply_btn.setEnabled(ok and self.main.data is not None)
        self.save_btn.setEnabled(ok or self.scale is not None)
        self.tb_save.setEnabled(self.main.data is not None
                                and self.manual.isChecked())

    def _segment_changed(self, points):
        px = lib.segment_length(points or [])
        if px > 0:
            self.len_label.setText("<b>%.2f px</b>" % px)
            (x0, y0), (x1, y1) = points
            self.ends_label.setText("(%.1f, %.1f) -> (%.1f, %.1f) px"
                                    % (x0, y0, x1, y1))
        else:
            self.len_label.setText("-- (draw the segment)")
            self.ends_label.setText("")
        self._update_preview()

    def select(self, on):
        """Selection button: start / stop the segment selection on the
        calibration image (the image shown as it is: zoom kept)."""
        if on:
            path = self.image.currentData(self._role)
            page = self.main.figures.find(self.STEP, self.FIG)
            if page is None or getattr(page, "image_path", None) != \
                    (path, self._rotate()):
                page = self.show_image()
            if page is None:
                self.select_btn.setChecked(False)
                return
            if page.float_window is None:
                self.main.figures.focus(page)
            toolbar_tool_off(page.toolbar)       # view kept
            self.editor.active = True
            self.select_btn.setText("Drawing... (click to stop)")
            self.main.status("Hold the left button and drag to draw the "
                             "segment; drag one of its ends to move it", 0)
        else:
            self.editor.active = False
            self.editor.drag = None
            self.select_btn.setText("Draw segment")
            self.main.status("Selection stopped")

    def reset(self):
        """Delete the segment (figure and lengths in px and mm); the
        selection state, the applied calibration and the parameter file
        are left as they are."""
        self.editor.drag = None
        self.editor.set_points(None)
        self.mm.setValue(0.0)                    # shown as "--"
        self._segment_changed(None)
        self.main.status("Segment reset", log=True)

    def _segment_released(self, points):
        px = lib.segment_length(points or [])
        if px > 0:
            self.main.status("Segment %.2f px -- give its length in mm, then "
                             "Apply" % px, log=True)

    # ---- image ------------------------------------------------------------ #
    def fill_images(self, select=None):
        lp = self.main.load_panel
        keep = select or self.image.currentData(self._role)
        found = lib.find_calibration_images(lp.run_dir, lp.root_dir)
        paths = found + [p for p in self._extra if p not in found]
        cb = self.image
        cb.blockSignals(True)
        cb.clear()
        for p in paths:
            where = ("run" if lp.run_dir and os.path.dirname(p) == lp.run_dir
                     else "root" if os.path.dirname(p) == lp.root_dir
                     else os.path.dirname(p))
            cb.addItem("%s   (%s)" % (os.path.basename(p), where), p)
            cb.setItemData(cb.count() - 1, p, E("Qt.ItemDataRole.ToolTipRole"))
        if not paths:
            cb.addItem("(no image found -- Browse...)", None)
        if keep in paths:
            cb.setCurrentIndex(paths.index(keep))
        cb.blockSignals(False)

    def browse(self):
        lp = self.main.load_panel
        start = lp.run_dir or lp.root_dir or os.path.expanduser("~")
        path, _f = QtWidgets.QFileDialog.getOpenFileName(
            self, "Calibration image", start,
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.mat);;"
            "All files (*)")
        if path:
            if path not in self._extra:
                self._extra.append(path)
            self.fill_images(select=path)
            self.image_chosen()

    @staticmethod
    def norm_rotation(v):
        """0, 90, 180 or -90 deg: the nearest one (270 = -90)."""
        v = int(round(float(v) / 90.0)) * 90 % 360
        return -90 if v == 270 else v

    def _rotate(self):
        return self.norm_rotation(self.rotate.value())

    def image_chosen(self):
        """A calibration image chosen (list or Browse...): shown, and the
        run information updated (CALIB_IMAGE) and saved."""
        self.show_image()
        d = self.main.data
        path = self.image.currentData(self._role)
        if d is None or not path or d.calib_info.get("CALIB_IMAGE") == path:
            return
        d.calib_info["CALIB_IMAGE"] = path
        ci = dict(d.params.get("CALIB_INFO") or {})
        ci["CALIB_IMAGE"] = path
        d.params["CALIB_INFO"] = ci              # kept at a reload
        self.main.write_log("Calibration image: %s" % path)
        self.main.show_info(d)                   # table + run_information

    def set_rotation(self, v):
        """The Rotation box (no reload; the segment is taken as drawn in
        this rotation)."""
        self.rotate.blockSignals(True)
        self.rotate.setValue(self.norm_rotation(v or 0))
        self.rotate.blockSignals(False)
        self._rot_shown = self._rotate()

    def _image_shape(self, path):
        """(rows, columns) of an image, not rotated (cached)."""
        cache = self.__dict__.setdefault("_shapes", {})
        if path not in cache:
            cache[path] = tuple(np.asarray(lib.read_image(path)).shape[:2])
        return cache[path]

    def _rotate_changed(self):
        """Rotation changed: the value set to 0 / 90 / 180 / -90; the
        calibration image turned at once (with the segment drawn on it);
        the run loaded again rotated."""
        rot = self._rotate()
        if rot != self.rotate.value():
            self.set_rotation(rot)
        old = getattr(self, "_rot_shown", 0)
        self._rot_shown = rot
        path = self.image.currentData(self._role)
        if old != rot and self.editor.points and path:
            try:                                 # the segment turns too
                self.editor.set_points(lib.rotate_points_px(
                    self.editor.points, old, rot, self._image_shape(path)))
                self._segment_changed(self.editor.points)
            except Exception as exc:
                self.main.write_log("WARNING: segment not rotated: %s" % exc)
        if path and (self.main.figures.find(self.STEP, self.FIG) is not None
                     or self.main.step == self.STEP):
            self.show_image()                    # the figure rotated
        d = self.main.data
        if d is not None and int(d.params.get("ROTATE") or 0) != rot:
            if self.main._thread is not None:
                return
            P = dict(d.params)
            P["ROTATE"] = rot
            self.main.status("Rotation %d deg: the run is loaded again" % rot,
                             log=True)
            self.main._after_load_step = self.STEP
            self.main.load_run(d.run_dir, P, None,
                               calibrate=d.calibration_asked,
                               velocity=d.default)

    def _result_text(self):
        if self.scale is None:
            return None
        return "applied: %.6g m/px" % self.scale

    def show_image(self):
        """The selected image in the figure 'Calibration image', with the
        segment (editable). Redrawing the same image keeps the view (zoom)
        the user left."""
        path = self.image.currentData(self._role)
        if not path:
            return None
        try:
            page = self.main.figures.find(self.STEP, self.FIG)
            view = None
            if page is not None and getattr(page, "image_path", None) == \
                    (path, self._rotate()) and page.figure.axes:
                ax = page.figure.axes[0]
                view = (ax.get_xlim(), ax.get_ylim())
            self.editor.detach()
            fig = lib.calibration_figure(
                path, self._rotate(), result=self._result_text(),
                fig=page.figure if page is not None else lib.new_figure())
            if view is not None:
                fig.axes[0].set_xlim(view[0])
                fig.axes[0].set_ylim(view[1])
            if page is None:
                page = self.main.figures.add(
                    self.STEP, self.FIG, fig, data_fn=self._data,
                    update_fn=self.show_image)
            page.image_path = (path, self._rotate())
            self.editor.attach(page)
            page.redraw()
            self.main.figures.focus(page)
            return page
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Calibration image",
                                          "%s: %s" % (os.path.basename(path),
                                                      exc))
            return None

    def _data(self):
        out = {"image": self.image.currentData(self._role) or "",
               "rotate_deg": self._rotate()}
        if self.editor.points:
            out.update(segment_px=np.asarray(self.editor.points, float),
                       length_px=lib.segment_length(self.editor.points))
        if self.mm.value() > 0:
            out.update(length_mm=self.mm.value())
        if self.scale is not None:
            out.update(XSCALE_m_per_px=self.scale)
        return out

    # ---- apply / save ----------------------------------------------------- #
    def _time_base(self):
        """(pulse separation [s], camera fps [Hz]) typed in the Time base
        box."""
        try:
            dt = float(self.dt.text().replace(",", "."))
            fps = float(self.cam.text().replace(",", "."))
        except ValueError:
            raise ValueError("the pulse separation and the camera fps must "
                             "be numbers")
        if dt <= 0 or fps <= 0:
            raise ValueError("the pulse separation and the camera fps must "
                             "be positive")
        return dt, fps

    def apply_time(self):
        """Time base > Apply (Manual only): the frames per PIV field, the
        pulse separation and the camera fps typed are kept in memory (the
        GUI's time base, not the parameter file) and the run is loaded
        again with them."""
        d = self.main.data
        if d is None or not self.manual.isChecked():
            return
        try:
            dt, fps = self._time_base()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Time base", str(exc))
            return
        fpf = int(self.fpf.value())
        self.main.time_base = lib.full_time_base(dict(
            frames_per_field=fpf, pulse_separation=dt, camera_fps=fps,
            manual=bool(d.acq.log_ok)))
        self.main.status("Time base (in memory): %d frames per PIV field, "
                         "pulse separation %.6g s, camera %.6g Hz, TSCALE "
                         "%.6g s%s" % (fpf, dt, fps, fpf / fps,
                                       ", manual (log not used)"
                                       if d.acq.log_ok else ""), log=True)
        self._reload()

    def _manual_toggled(self, on):
        """Manual unticked while the run uses a manual time base: back to
        the log at once (nothing to save: the time base is in memory)."""
        if not on and self.main.time_base.get("manual") and \
                self.main.data is not None and self.main.data.acq.log_ok:
            fpf = self.main.time_base["frames_per_field"]
            self.main.time_base = lib.full_time_base(dict(
                self.main.time_base, manual=False))
            self.main.status("Time base: back to the log (%d frames per PIV "
                             "field)" % fpf, log=True)
            self._time_fields()
            self._reload()
            return
        self._time_fields()

    def _reload(self):
        """Load the loaded run again (same parameters, the time base in
        memory), staying in this step."""
        d, lp = self.main.data, self.main.load_panel
        self.main._after_load_step = self.STEP
        self.main.load_run(d.run_dir, dict(d.params),
                           getattr(self.main, "_param_file", None),
                           calibrate=d.calibration_asked, velocity=d.default)

    def apply(self, save=False):
        """Apply: XSCALE = YSCALE from the segment, used for the loaded run
        (in memory: no parameter file any more -- Save keeps it in a
        calibration file); the calibration record (CALIB_IMAGE,
        CALIB_SEGMENT_PX, CALIB_LENGTH_PX, CALIB_LENGTH_MM) goes with the
        run's parameters in memory and to the run information."""
        lp = self.main.load_panel
        scale = self._new_scale()
        if scale is None:
            self.main.status("Draw the segment and give its length in mm "
                             "first")
            return
        d = self.main.data
        pts = [[round(x, 3), round(y, 3)] for x, y in self.editor.points]
        changes = dict(XSCALE=float(scale), YSCALE=float(scale))
        self.scale, self.seg_m = scale, self.mm.value() * 1e-3
        record = dict(CALIB_IMAGE=self.image.currentData(self._role),
                      CALIB_SEGMENT_PX=pts,
                      CALIB_LENGTH_PX=round(lib.segment_length(pts), 3),
                      CALIB_LENGTH_MM=float(self.mm.value()))
        text = "XSCALE = YSCALE = %.6g m/px" % scale
        self._last_apply = "apply"
        self.cal_label.setText("<b>%s</b>" % text)
        self.main.status("Calibration applied: %s" % text, log=True)
        self.main.mark_done("calibration")
        if self.main.figures.find(self.STEP, self.FIG) is not None:
            self.show_image()
        if d is None:
            self.refresh()
            return
        d.calib_info.update(record)
        self.select_btn.setChecked(False)
        self.main._after_load_step = self.STEP
        P = dict(d.params)
        P.update(changes)
        P["ROTATE"] = self._rotate()
        P["CALIB_INFO"] = dict(record)
        self.main.load_run(d.run_dir, P, None, calibrate=True,
                           velocity=d.default)

    # ---- calibration file ------------------------------------------------- #
    CAL_EXT = CAL_EXT

    @staticmethod
    def calib_info_of(rec):
        """The CALIB_ record of a calibration file."""
        return dict(CALIB_IMAGE=rec.get("image"),
                    CALIB_SEGMENT_PX=rec.get("segment_px"),
                    CALIB_LENGTH_PX=rec.get("length_px"),
                    CALIB_LENGTH_MM=rec.get("length_mm"))

    def time_base_from(self, rec, log_ok=None):
        """The time base of a calibration file set in memory (Manual and
        its values, or the log with its frames per PIV field); log_ok: the
        loaded run has a log (None: not known yet). Returns notes."""
        tb = rec.get("time_base") or {}
        if not tb:
            return []
        fpf = int(tb.get("frames_per_field") or 2)
        if tb.get("manual") or log_ok is False:
            try:
                dt = float(str(tb.get("pulse_separation_s")).replace(",", "."))
                fps = float(str(tb.get("camera_fps_Hz")).replace(",", "."))
            except (TypeError, ValueError):
                return ["time base of the file not usable"]
            self.main.time_base = lib.full_time_base(dict(
                frames_per_field=fpf, pulse_separation=dt, camera_fps=fps,
                manual=bool(tb.get("manual"))))
            self.manual.blockSignals(True)
            self.manual.setChecked(True)
            self.manual.blockSignals(False)
            self.fpf.blockSignals(True)
            self.fpf.setValue(fpf)
            self.fpf.blockSignals(False)
            self.dt.setText("%.6g" % dt)
            self.cam.setText("%.6g" % fps)
            self._manual_typed = True
        else:
            self.main.time_base = lib.full_time_base(dict(
                self.main.time_base, frames_per_field=fpf, manual=False))
            self.manual.blockSignals(True)
            self.manual.setChecked(False)
            self.manual.blockSignals(False)
            self._manual_typed = False
        return []

    def fill_from(self, rec):
        """The widgets of the step from a calibration file (image,
        segment, length in mm); returns notes."""
        notes = []
        if rec.get("rotate_deg") is not None:
            self.set_rotation(rec["rotate_deg"])
        img = rec.get("image")
        if img and os.path.isfile(img):
            if img not in self._extra:
                self._extra.append(img)
            self.fill_images(select=img)
        elif img:
            notes.append("image not found: %s (the segment is used "
                         "without it)" % img)
        self.editor.set_points(rec.get("segment_px") or None)
        self.mm.setValue(float(rec.get("length_mm") or 0))
        self._segment_changed(self.editor.points)
        return notes

    def after_load(self, path, rec):
        """Load data loaded the run with this calibration file (XSCALE,
        YSCALE, time base already used): the step shown as run from it."""
        notes = self.fill_from(rec)
        d = self.main.data
        self.scale = (float(d.params["XSCALE"]) if d is not None
                      and d.can_calibrate else None)
        self._last_apply = "apply"
        self.cal_file_label.setText("Loaded: %s%s" % (
            path, "".join("<br>" + n for n in notes)))
        self.cal_file_label.setVisible(True)
        self.refresh()
        if self.scale is not None:
            self.main.mark_done("calibration")
        self.main.write_log("Calibration step run from %s: %s%s" % (
            path, "XSCALE = YSCALE = %.6g m/px" % self.scale
            if self.scale else "no scale",
            "".join("; " + n for n in notes)))

    def calibration_record(self):
        """Everything needed to run this step again (the calibration file)."""
        import datetime
        d = self.main.data
        pts = self.editor.points
        tb = dict(self.main.time_base)
        return {
            "kind": "PIV_GUI calibration", "version": 1,
            "saved": datetime.datetime.now().isoformat(timespec="seconds"),
            "run": d.run if d is not None else None,
            "run_folder": d.run_dir if d is not None else None,
            "image": self.image.currentData(self._role),
            "rotate_deg": self._rotate(),
            "segment_px": ([[float(x), float(y)] for x, y in pts]
                           if pts else None),
            "length_px": (float(lib.segment_length(pts)) if pts else None),
            "length_mm": self.mm.value() or None,
            "scale_m_per_px": self._new_scale(),
            "applied_m_per_px": self.scale,
            "XSCALE": (float(d.params["XSCALE"]) if d is not None
                       and d.can_calibrate else self.scale),
            "YSCALE": (float(d.params["YSCALE"]) if d is not None
                       and d.can_calibrate else self.scale),
            "applied_with": self._last_apply,
            "time_base": {
                "manual": bool(self.manual.isChecked()),
                "frames_per_field": int(self.fpf.value()),
                "pulse_separation_s": self.dt.text(),
                "camera_fps_Hz": self.cam.text(),
                "in_use": {k: (v if isinstance(v, (int, float, str, bool,
                                                   type(None))) else str(v))
                           for k, v in tb.items()}},
        }

    def save_calibration(self, path=None):
        """Save calibration...: the calibration file (asks where; proposed
        <run>/PostProcessing/<run>.pivcalib.json). Returns the path."""
        rec = self.calibration_record()
        if rec["segment_px"] is None and rec["applied_m_per_px"] is None:
            QtWidgets.QMessageBox.information(
                self, "Save calibration", "Nothing to save yet: draw the "
                "segment and give its length in mm first.")
            return None
        if path is None:
            d = self.main.data
            folder = self.main.setup_folder()
            stem = (d.run if d is not None else "calibration")
            path, _f = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save calibration", os.path.join(
                    folder, stem + self.CAL_EXT),
                json_filter("Calibration", self.CAL_EXT))
            if not path:
                return None
        path = with_double_ext(path, self.CAL_EXT)
        try:
            with open(path, "w") as fh:
                json.dump(rec, fh, indent=1, default=str)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save calibration",
                                          "%s: %s" % (path, exc))
            return None
        self.cal_file_label.setText("Saved: %s" % path)
        self.cal_file_label.setVisible(True)
        self.main.setup_file_used("calibration", path)
        self.main.status("Calibration saved in %s (segment %s, %s mm, "
                         "scale %s m/px, time base %s)" % (
                             path, "yes" if rec["segment_px"] else "no",
                             rec["length_mm"], rec["scale_m_per_px"],
                             "manual" if rec["time_base"]["manual"]
                             else "from the log"), log=True)
        return path

    def load_calibration(self, path=None, run=True):
        """Load calibration...: read a calibration file, set the image, the
        segment, the length in mm and the time base, then run this step
        again (Apply: the loaded run, and the parameter file). Returns the
        record read."""
        if path is None:
            start = self.main.setup_folder()
            path, _f = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load calibration", start,
                json_filter("Calibration", self.CAL_EXT))
            if not path:
                return None
        try:
            with open(path) as fh:
                rec = json.load(fh)
            if not isinstance(rec, dict) or not (
                    rec.get("segment_px") or rec.get("applied_m_per_px")
                    or rec.get("XSCALE")):
                raise ValueError("not a calibration file")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load calibration",
                                          "%s: %s" % (path, exc))
            return None
        notes = self.fill_from(rec)
        d = self.main.data
        notes += self.time_base_from(
            rec, log_ok=d.acq.log_ok if d is not None else None)
        self.cal_file_label.setText("Loaded: %s%s" % (
            path, "".join("<br>" + n for n in notes)))
        self.cal_file_label.setVisible(True)
        self.main.setup_file_used("calibration", path)
        self.main.status("Calibration loaded from %s%s" % (
            path, "".join("; " + n for n in notes)), log=True)
        if not run:
            return rec
        if d is None:
            self.refresh()
            self.main.status("Calibration loaded; load a run to apply it",
                             log=True)
            return rec
        if self._new_scale() is not None:
            self.apply()
        elif rec.get("XSCALE"):                  # no segment: the scales
            P = dict(d.params)
            P.update(XSCALE=float(rec["XSCALE"]),
                     YSCALE=float(rec.get("YSCALE") or rec["XSCALE"]),
                     ROTATE=self._rotate())
            self.scale = float(rec["XSCALE"])
            self.main.mark_done("calibration")
            self.main._after_load_step = self.STEP
            self.main.load_run(d.run_dir, P, None, calibrate=True,
                               velocity=d.default)
        else:                                    # only the time base
            self._reload()
        return rec

    # ---- session ---------------------------------------------------------- #
    def get_state(self):
        return dict(image=self.image.currentData(self._role),
                    segment=self.editor.points,
                    length_mm=self.mm.value() or None,
                    rotate_deg=self._rotate())

    def set_state(self, s):
        if s.get("rotate_deg") is not None:
            self.set_rotation(s["rotate_deg"])
        if s.get("image") and os.path.isfile(s["image"]):
            if s["image"] not in self._extra:
                self._extra.append(s["image"])
            self.fill_images(select=s["image"])
        if s.get("segment"):
            self.editor.set_points(s["segment"])
        mm = s.get("length_mm")
        if mm is None and s.get("length_m"):     # older sessions
            mm = float(s["length_m"]) * 1e3
        if mm:
            self.mm.setValue(float(mm))
        self._segment_changed(self.editor.points)


# --------------------------------------------------------------------------- #
#  Pre-processing > Define ROI
# --------------------------------------------------------------------------- #
class ShapeDrawer(object):
    """One shape drawn on a figure, with the mouse:
      - 'rect' (two opposite corners), 'line' (its two ends), 'circle'
        (centre, then a point of the circle): drag with the left button
        held. With edit=True (Mask step), a rectangle or a circle stays
        editable once drawn: drag inside it to move it, drag a corner (the
        rectangle) or the edge (the circle) to resize it; a right click or
        Enter validates it. Without edit, the release validates it.
      - 'poly': every left click is a new vertex; Enter or a right click
        closes the polygon (Backspace removes the last vertex).
    done(points) is called when the shape is validated -- points in the
    axes' unit -- or with None when cancelled (Esc, or cancel()). A click
    without dragging does not draw a rectangle / circle / line. The view
    (zoom) is not changed; the figure's right-click menu is off meanwhile."""

    STYLE = {"rect": dict(color="red", ls="--", lw=1.6),
             "line": dict(color="magenta", ls="-", lw=2.0, marker="o", ms=4),
             "circle": dict(color="orange", ls="-", lw=1.8),
             "poly": dict(color="orange", ls="-", lw=1.8, marker="o", ms=4)}
    PICK = 10                                    # screen px

    def __init__(self, main, page, kind, done, edit=False):
        self.main, self.page, self.kind, self.done = main, page, kind, done
        self.edit = bool(edit) and kind in ("rect", "circle")
        self.ax = page.figure.axes[0]
        self.p0 = self.p1 = None
        self.verts = []                          # poly
        self.cursor = None
        self.editing = False                     # drawn, being adjusted
        self.drag = None                         # what the button moves
        self.art = self.handles = None
        page.drawing = True                      # right click: not the menu
        toolbar_tool_off(page.toolbar)
        c = page.canvas
        self.cids = [c.mpl_connect("button_press_event", self._press),
                     c.mpl_connect("motion_notify_event", self._move),
                     c.mpl_connect("button_release_event", self._release),
                     c.mpl_connect("key_press_event", self._key)]

    def rearm(self, page):
        """The same drawing, on the redrawn axes of `page` (state kept)."""
        done, self.done = self.done, None
        state = (list(self.verts), self.p0, self.p1, self.editing)
        self.finish(None)
        new = ShapeDrawer(self.main, page, self.kind, done, self.edit)
        new.verts, new.p0, new.p1, new.editing = state
        if new.verts or new.editing:
            new._show()
        return new

    # ---- drawing ---------------------------------------------------------- #
    def _xy(self):
        if self.kind == "poly":
            pts = self.verts + ([self.cursor] if self.cursor else [])
            return [p[0] for p in pts], [p[1] for p in pts]
        if self.kind == "line":
            (x0, z0), (x1, z1) = self.p0, self.p1
            return [x0, x1], [z0, z1]
        return lib.area_outline(self.kind, [self.p0, self.p1])

    def _corners(self):
        (x0, z0), (x1, z1) = self.p0, self.p1
        return [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]

    def _show(self):
        if self.art is None:
            self.art, = self.ax.plot([], [], zorder=6, scalex=False,
                                     scaley=False, **self.STYLE[self.kind])
        self.art.set_data(*self._xy())
        if self.editing:                         # the handles
            if self.handles is None:
                self.handles, = self.ax.plot(
                    [], [], "s", ms=7, mfc="white", mec="black", zorder=7,
                    scalex=False, scaley=False)
            pts = self._corners() if self.kind == "rect" else \
                [self.p0, self.p1]
            self.handles.set_data([p[0] for p in pts], [p[1] for p in pts])
        self.page.canvas.draw_idle()

    def _tool_on(self):
        if getattr(self.page.toolbar, "mode", ""):
            self.main.status("The toolbar zoom / pan tool is on: click its "
                             "button again to switch it off, then draw", 0)
            return True
        return False

    def _scr(self, p):
        return self.ax.transData.transform([p])[0]

    def _dist(self, p, ev):
        return float(np.hypot(*(self._scr(p) - (ev.x, ev.y))))

    # ---- mouse ------------------------------------------------------------ #
    def _press(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None or self._tool_on():
            return
        p = (float(ev.xdata), float(ev.ydata))
        if self.kind == "poly":
            if ev.button == 3:
                self._close_poly()
            elif ev.button == 1:
                self.verts.append(p)
                self._show()
                self.main.status("%d vertices -- left click: a new vertex; "
                                 "Enter or right click closes the polygon "
                                 "(Backspace: last vertex off, Esc cancels)"
                                 % len(self.verts), 0)
            return
        if self.editing:
            if ev.button == 3:
                self._validate()
            elif ev.button == 1:
                self.drag = self._hit(ev, p)
            return
        if ev.button == 1:
            self.p0 = self.p1 = p
            self._show()

    def _hit(self, ev, p):
        """What a press on the drawn shape grabs: a corner / the edge
        (resize) or the inside (move); None elsewhere."""
        if self.kind == "rect":
            cs = self._corners()
            d = [self._dist(c, ev) for c in cs]
            k = int(np.argmin(d))
            if d[k] <= self.PICK:
                return ("corner", cs[(k + 2) % 4])    # the opposite stays
            (x0, z0), (x1, z1) = self.p0, self.p1
            if min(x0, x1) <= p[0] <= max(x0, x1) and \
                    min(z0, z1) <= p[1] <= max(z0, z1):
                return ("move", p, self.p0, self.p1)
            return None
        # circle: centre (move), edge (resize), inside (move)
        r = float(np.hypot(*(self._scr(self.p1) - self._scr(self.p0))))
        dc = self._dist(self.p0, ev)
        if abs(dc - r) <= self.PICK:
            return ("edge",)
        if dc < r:
            return ("move", p, self.p0, self.p1)
        return None

    def _move(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None:
            return
        p = (float(ev.xdata), float(ev.ydata))
        if self.kind == "poly":
            if self.verts:
                self.cursor = p
                self._show()
            return
        if self.editing:
            if self.drag is None:
                return
            if self.drag[0] == "corner":
                self.p0, self.p1 = self.drag[1], p
            elif self.drag[0] == "edge":
                self.p1 = p
            else:                                # move
                _m, a, q0, q1 = self.drag
                dx, dz = p[0] - a[0], p[1] - a[1]
                self.p0 = (q0[0] + dx, q0[1] + dz)
                self.p1 = (q1[0] + dx, q1[1] + dz)
            self._show()
            return
        if self.p0 is None:
            return
        self.p1 = self._snap(p, ev)
        self._show()

    @staticmethod
    def _shift(ev):
        """Shift held during the mouse event."""
        if "shift" in str(getattr(ev, "key", "") or ""):
            return True
        if "shift" in (getattr(ev, "modifiers", None) or ()):
            return True
        try:
            return bool(QtWidgets.QApplication.keyboardModifiers()
                        & E("Qt.KeyboardModifier.ShiftModifier"))
        except Exception:
            return False

    def _snap(self, p, ev):
        """A line drawn with Shift held: horizontal or vertical (0, 90,
        180 or 270 deg on the screen), whichever is nearer."""
        if self.kind != "line" or self.p0 is None or not self._shift(ev):
            return p
        a, b = self._scr(self.p0), self._scr(p)
        if abs(b[0] - a[0]) >= abs(b[1] - a[1]):
            return (p[0], self.p0[1])            # horizontal
        return (self.p0[0], p[1])                # vertical

    def _release(self, ev):
        if self.kind == "poly" or ev.button != 1:
            return
        if self.editing:
            self._move(ev)
            self.drag = None
            return
        if self.p0 is None:
            return
        self._move(ev)
        if np.hypot(*(self._scr(self.p1) - self._scr(self.p0))) < 4:
            self._clear()                        # a click: keep armed
            self.p0 = self.p1 = None
            return
        if not self.edit:
            self.finish([self.p0, self.p1])
            return
        self.editing = True                      # adjust, then validate
        self._show()
        self.main.status(
            "Drag inside the %s to move it, %s to resize it; right click "
            "or Enter validates it (Esc cancels)" % (
                "rectangle" if self.kind == "rect" else "circle",
                "a corner" if self.kind == "rect" else "its edge"), 0)

    def _validate(self):
        pts = [self.p0, self.p1]
        # deferred: out of the mouse handler (the panel redraws the figure)
        QtCore.QTimer.singleShot(0, lambda: self.finish(pts))

    def _close_poly(self):
        if len(self.verts) < 3:
            self.main.status("A polygon needs at least 3 vertices", 0)
            return
        pts = list(self.verts)
        QtCore.QTimer.singleShot(0, lambda: self.finish(pts))

    def _key(self, ev):
        if ev.key == "escape":
            self.finish(None)
        elif ev.key in ("enter", "return"):
            if self.kind == "poly":
                self._close_poly()
            elif self.editing:
                self._validate()
        elif self.kind == "poly" and ev.key == "backspace" and self.verts:
            self.verts.pop()
            self._show()

    def _clear(self):
        for a in (self.art, self.handles):
            if a is not None:
                try:
                    a.remove()
                except Exception:
                    pass
        self.art = self.handles = None
        self.page.canvas.draw_idle()

    def cancel(self):
        self.finish(None)

    def finish(self, points):
        if self.done is None and not self.cids:
            return                               # already finished
        for cid in self.cids:
            try:
                self.page.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self.cids = []
        self._clear()
        page = self.page
        # the right click that validated must not open the menu
        QtCore.QTimer.singleShot(300, lambda: setattr(page, "drawing",
                                                      False))
        done, self.done = self.done, None
        if done is not None:
            done(points)


class RoiPanel(QtWidgets.QWidget):
    """Pre-processing > Define ROI.

    Figure: the current frame of Load data or an image, in the calibrated
    view (m). ROI area: the rectangles of PTS_ROI; ROI Line: the segments of
    LINES -- each with a tick box (drawn on the figure when ticked) and an
    editable name; Add draws one on the figure (drag with the left button
    held), Remove deletes the selected one (and, for an ROI, its quantities
    '<quantity>_ROIxx' in memory). Apply uses the lists for the loaded run
    (in memory; no parameter file) and calculates for each active ROI the
    velocity norm, divergence, vorticity and kinetic energy from the
    velocity extracted in the ROI ('<quantity>_ROIxx', all frames) -- only
    those that do not exist yet. Save writes the ROI file (.pivroi.json)
    that Load ROI selection and File > Load data run the step from."""

    STEP = "roi"
    FIG = "ROI selection"
    SOURCES = [("Current Frame", "frame"), ("Select Image", "image")]
    KINDS = {"rect": ("ROI", "PTS_ROI"), "line": ("LINE", "LINES")}

    def __init__(self, main):
        super().__init__()
        self.main = main
        self._role = E("Qt.ItemDataRole.UserRole")
        self._extra = []
        self.drawer = None
        self.dirty = False
        self._populated = None                   # (run, parameter file)
        lay = QtWidgets.QVBoxLayout(self)
        self.roi_load_btn = QtWidgets.QPushButton("Load ROI selection...")
        self.roi_load_btn.setMinimumHeight(32)
        self.roi_load_btn.setToolTip(
            "read an ROI file (.pivroi.json, saved with Save below) and run "
            "this step again with it: the ROIs and lines, their ticks and "
            "the figure, then Apply (File > Load data can also load it with "
            "the run)")
        self.roi_load_btn.clicked.connect(lambda: self.load_rois())
        lay.addWidget(self.roi_load_btn)
        self.roi_file_label = QtWidgets.QLabel("")
        self.roi_file_label.setWordWrap(True)
        self.roi_file_label.setStyleSheet("color: gray;")
        self.roi_file_label.setVisible(False)
        lay.addWidget(self.roi_file_label)

        box = QtWidgets.QGroupBox("Figure")
        g = QtWidgets.QGridLayout(box)
        self.source = QtWidgets.QComboBox()
        for lab, key in self.SOURCES:
            self.source.addItem(lab, key)
        self.source.setToolTip(
            "the figure to select from: the current frame of Load data "
            "(Frame of its Display), or an image placed on the PIV grid")
        self.image = QtWidgets.QComboBox()
        self.image.setSizeAdjustPolicy(E(
            "QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon"))
        self.image.setToolTip("image files and background*.mat of the run "
                              "and root folders; Browse... for another one")
        self.browse_btn = QtWidgets.QPushButton("Browse...")
        self.browse_btn.setMaximumWidth(90)
        g.addWidget(QtWidgets.QLabel("Select from"), 0, 0)
        g.addWidget(self.source, 0, 1, 1, 2)
        self.image_lab = QtWidgets.QLabel("image")
        g.addWidget(self.image_lab, 1, 0)
        g.addWidget(self.image, 1, 1)
        g.addWidget(self.browse_btn, 1, 2)
        g.setColumnStretch(1, 1)
        lay.addWidget(box)
        self.style = MapStyleBox()
        lay.addWidget(self.style)
        self.style.changed.connect(lambda: self.draw(keep_view=True))
        self.source.currentIndexChanged.connect(lambda *_: self._source())
        self.image.activated.connect(lambda *_: self.draw())
        self.browse_btn.clicked.connect(self.browse)

        self.units_note = QtWidgets.QLabel("")
        self.units_note.setWordWrap(True)
        lay.addWidget(self.units_note)

        self.trees = {}
        self.add_btn, self.rem_btn = {}, {}
        for kind, title, add_txt, rem_txt, cols in (
                ("rect", "ROI area", "Add ROI", "Remove ROI",
                 ["ROI", "x (m)", "z (m)"]),
                ("line", "ROI Line", "Add line", "Remove line",
                 ["Line", "start (m)", "end (m)"])):
            box = QtWidgets.QGroupBox(title)
            v = QtWidgets.QVBoxLayout(box)
            t = QtWidgets.QTreeWidget()
            t.setHeaderLabels(cols)
            t.setRootIsDecorated(False)
            t.setMinimumHeight(90)
            t.setMaximumHeight(170)
            t.setToolTip("tick: drawn on the figure; double click a name to "
                         "rename it")
            t.itemChanged.connect(lambda it, col, k=kind:
                                  self._item_changed(k, it, col))
            t.itemSelectionChanged.connect(lambda: self.draw(keep_view=True))
            v.addWidget(t)
            row = QtWidgets.QHBoxLayout()
            a = QtWidgets.QPushButton(add_txt)
            r = QtWidgets.QPushButton(rem_txt)
            a.setToolTip("drag on the figure with the left button held to "
                         "draw the %s (Esc cancels)"
                         % ("rectangle" if kind == "rect" else "segment"))
            r.setToolTip("remove the selected %s" % title.split()[-1].lower())
            a.clicked.connect(lambda _c=False, k=kind: self.add(k))
            r.clicked.connect(lambda _c=False, k=kind: self.remove(k))
            row.addWidget(a)
            row.addWidget(r)
            v.addLayout(row)
            lay.addWidget(box)
            self.trees[kind] = t
            self.add_btn[kind], self.rem_btn[kind] = a, r

        self.save_note = QtWidgets.QLabel("")
        self.save_note.setWordWrap(True)
        lay.addWidget(self.save_note)
        self.update_btn = QtWidgets.QPushButton("Apply")
        self.update_btn.setMinimumHeight(34)
        f = self.update_btn.font(); f.setBold(True)
        self.update_btn.setFont(f)
        self.update_btn.setToolTip(
            "use the ROIs and the lines of the lists for the loaded run (in "
            "memory: nothing is written); then, for each active ROI, extract "
            "the velocity in it (<velocity>_ROIxx) and calculate from it the "
            "velocity norm, divergence, vorticity and kinetic energy for "
            "every frame (<quantity>_ROIxx); and along each ticked line the "
            "same quantities, in the full field and in each active ROI it "
            "crosses (<quantity>_LINEx_region, used by Profiles) -- only the "
            "ones that do not exist yet (or whose ROI, line or mask "
            "changed)")
        self.update_btn.clicked.connect(self.update_param_file)
        self.roi_save_btn = QtWidgets.QPushButton("Save")
        self.roi_save_btn.setMinimumHeight(34)
        self.roi_save_btn.setToolTip(
            "Apply first, then save the ROI selection in an ROI file "
            "(.pivroi.json): the ROIs "
            "and lines with their names, coordinates (m) and ticks, the "
            "figure and its display -- everything needed to run this step "
            "again (Load ROI selection, File > Load data)")
        self.roi_save_btn.clicked.connect(lambda: self.save_rois())
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.update_btn, 1)
        row.addWidget(self.roi_save_btn, 1)
        lay.addLayout(row)
        lay.addStretch(1)
        self._source()
        self._state()

    # ---- lists ------------------------------------------------------------ #
    def _fmt(self, kind, pts):
        (x1, z1), (x2, z2) = pts
        if kind == "rect":
            return ["%.4g .. %.4g" % tuple(sorted((x1, x2))),
                    "%.4g .. %.4g" % tuple(sorted((z1, z2)))]
        return ["(%.4g, %.4g)" % (x1, z1), "(%.4g, %.4g)" % (x2, z2)]

    def _add_item(self, kind, name, pts, checked=True):
        t = self.trees[kind]
        t.blockSignals(True)
        it = QtWidgets.QTreeWidgetItem([name] + self._fmt(kind, pts))
        it.setFlags(it.flags() | E("Qt.ItemFlag.ItemIsUserCheckable")
                    | (E("Qt.ItemFlag.ItemIsEditable") if kind == "line"
                       else E("Qt.ItemFlag.NoItemFlags")))
        it.setCheckState(0, E("Qt.CheckState.Checked") if checked
                         else E("Qt.CheckState.Unchecked"))
        it.setData(0, self._role, [list(p) for p in pts])
        it.setData(1, self._role, name)          # name before an edit
        t.addTopLevelItem(it)
        t.blockSignals(False)
        for c in range(3):
            t.resizeColumnToContents(c)
        return it

    def shapes(self, kind, checked_only=False, colours=False):
        """[(name, ((x1, z1), (x2, z2)))] of a list (+ the colour of each,
        given by its place in the list, when colours)."""
        t, out = self.trees[kind], []
        for i in range(t.topLevelItemCount()):
            it = t.topLevelItem(i)
            if checked_only and it.checkState(0) != E("Qt.CheckState.Checked"):
                continue
            (x1, z1), (x2, z2) = it.data(0, self._role)
            out.append((it.text(0), ((x1, z1), (x2, z2)))
                       + ((lib.roi_colour(kind, i),) if colours else ()))
        return out

    def shown(self, kind):
        """{name: colour} of the ROIs ('rect') or lines ('line') ticked in
        the list -- drawn on Maps and Quiver, and (ROIs) offered there; None
        when the lists are not those of the loaded run yet (all shown)."""
        d = self.main.data
        if d is None or getattr(self, "_populated", None) != (
                d.run_dir, self.main.load_panel.param_file):
            return None
        t, out = self.trees[kind], {}
        for i in range(t.topLevelItemCount()):
            it = t.topLevelItem(i)
            if it.checkState(0) == E("Qt.CheckState.Checked"):
                out[it.text(0)] = lib.roi_colour(kind, i)
        return out

    def _maps_follow(self):
        """Maps and Quiver follows the ticks (its ROI list and figure)."""
        d = self.main.data
        if d is None:
            return
        self.main.load_panel._fill_rois(d)
        self.main.draw_field()
        self.main.profiles_panel.sync()
        self.main.timeseries_panel.sync()
        self.main.fft_time_panel.sync()
        self.main.fft_space_panel.sync()

    def _recolour(self):
        """The colour of each ROI / line as an icon in front of its name
        (its colour on the figure)."""
        for kind, t in self.trees.items():
            t.blockSignals(True)
            for i in range(t.topLevelItemCount()):
                pm = QtGui.QPixmap(12, 12)
                pm.fill(QtGui.QColor(lib.roi_colour(kind, i)))
                p = QtGui.QPainter(pm)           # a border (white shows)
                p.setPen(QtGui.QColor("#555555"))
                p.drawRect(0, 0, 11, 11)
                p.end()
                t.topLevelItem(i).setIcon(0, QtGui.QIcon(pm))
            t.blockSignals(False)

    def _names(self, kind, but=None):
        t = self.trees[kind]
        return [t.topLevelItem(i).text(0) for i in range(t.topLevelItemCount())
                if t.topLevelItem(i) is not but]

    def _unique(self, kind, base):
        """base, or -- if the name is taken -- the same prefix with the
        number incremented ('ROI3' taken -> 'ROI4', 'wake' -> 'wake2'):
        no '_2' suffix."""
        import re
        names = set(self._names(kind))
        if base not in names:
            return base
        m = re.match(r"^(.*?)(\d+)$", base)
        prefix, i = (m.group(1), int(m.group(2)) + 1) if m else (base, 2)
        while "%s%d" % (prefix, i) in names:
            i += 1
        return "%s%d" % (prefix, i)

    def _renumber(self):
        """The ROIs named by their place in the list: ROI01, ROI02, ...
        (the xx of '<velocity>_ROIxx' and '<quantity>_ROIxx')."""
        t = self.trees["rect"]
        t.blockSignals(True)
        for i in range(t.topLevelItemCount()):
            it = t.topLevelItem(i)
            it.setText(0, lib.roi_name(i + 1))
            it.setData(1, self._role, lib.roi_name(i + 1))
        t.blockSignals(False)

    def _rois_to_run(self):
        """After an ROI is added or removed: the ROIs renumbered, the
        loaded run's PTS_ROI (in memory; the parameter file at Apply) and
        its ROI fields follow, and the stored ROI quantities follow their
        rectangle to its new number -- those of a removed ROI deleted."""
        self._renumber()
        d = self.main.data
        if d is None:
            return
        shapes = self.shapes("rect")
        renamed, deleted = lib.renumber_roi_quantities(
            d, [r for _n, r in shapes])
        d.params["PTS_ROI"] = lib.shapes_to_param(shapes)
        if deleted:
            self.main.write_log("    deleted: " + ", ".join(deleted))
        if renamed:
            self.main.write_log("    renamed: " + ", ".join(
                "%s -> %s" % rn for rn in renamed))
        self.main.show_info(d)

    def _next_name(self, kind):
        """ROI<n> / LINE<n>: n one more than the highest number used."""
        import re
        base = self.KINDS[kind][0]
        nums = [int(m.group(1)) for m in
                (re.match(r"^%s(\d+)$" % base, n) for n in self._names(kind))
                if m]
        return self._unique(kind, "%s%d" % (base, max(nums + [0]) + 1))

    def _item_changed(self, kind, it, col):
        if col == 0:
            new, old = it.text(0).strip(), it.data(1, self._role)
            if new != old:                       # renamed
                if not new or new in self._names(kind, but=it):
                    new = self._unique(kind, new or old)
                t = self.trees[kind]
                t.blockSignals(True)
                it.setText(0, new)
                it.setData(1, self._role, new)
                t.blockSignals(False)
            self._set_dirty()
        self.draw(keep_view=True)
        self._maps_follow()

    def _selected_name(self):
        for t in self.trees.values():
            items = t.selectedItems()
            if items:
                return items[0].text(0)
        return None

    def _set_dirty(self, on=True):
        self.dirty = on
        self._state()

    def _state(self):
        d = self.main.data
        ok = d is not None and d.can_calibrate
        busy = self.drawer is not None
        for k in self.trees:
            self.add_btn[k].setEnabled(ok and not busy)
            self.rem_btn[k].setEnabled(not busy)
        self.update_btn.setEnabled(d is not None and not busy)
        self.roi_save_btn.setEnabled(not busy and any(
            t.topLevelItemCount() for t in self.trees.values()))
        if d is None:
            self.units_note.setText("Load a run first (File > Load data).")
        elif not d.can_calibrate:
            self.units_note.setText(
                "<b>The ROIs and the lines are stored in m:</b> calibrate the "
                "run first (XSCALE / YSCALE, Pre-processing > Calibration).")
        else:
            self.units_note.setText("Coordinates in m (calibrated view).")
        nq = len(getattr(d, "roi_quantities", {})) if d is not None else 0
        nl = len(getattr(d, "line_quantities", {})) if d is not None else 0
        self.save_note.setText(
            ("<b>Not applied yet</b>" if self.dirty else "Applied")
            + ("<br>ROI quantities in memory: %d (&lt;quantity&gt;_ROIxx); "
               "along the lines: %d (&lt;quantity&gt;_LINEx_region)"
               % (nq, nl) if d is not None else ""))

    # ---- filling ---------------------------------------------------------- #
    def populate(self, d):
        """The lists from the run's parameters (PTS_ROI, LINES)."""
        if self.drawer is not None:
            self.drawer.cancel()
        for kind, (base, key) in self.KINDS.items():
            t = self.trees[kind]
            ticks = {t.topLevelItem(i).text(0): t.topLevelItem(i).checkState(
                0) == E("Qt.CheckState.Checked")
                for i in range(t.topLevelItemCount())}   # kept (a reload)
            t.clear()
            for name, pts in lib.parse_shapes(d.params.get(key), base):
                self._add_item(kind, name, pts,
                               checked=ticks.get(name, True))
        self._renumber()
        self._populated = (d.run_dir, self.main.load_panel.param_file)
        # ROIs of the file named otherwise: renamed by their place (in
        # memory now; in the file at Apply)
        old = [n for n, _r in lib.parse_shapes(d.params.get("PTS_ROI"),
                                               "ROI")]
        named = [n for n, _r in self.shapes("rect")]
        if old != named:
            d.params["PTS_ROI"] = lib.shapes_to_param(self.shapes("rect"))
            self.main.write_log("ROIs named by their place in the list: "
                                + ", ".join("%s -> %s" % p for p in
                                            zip(old, named) if p[0] != p[1])
                                + " (in memory)")
        self._set_dirty(old != named)

    def data_loaded(self, d):
        """After a Load: the lists from the parameter file (unless edits not
        saved yet for the same run and file), the figure redrawn if shown."""
        key = (d.run_dir, self.main.load_panel.param_file)
        if not (self.dirty and key == self._populated):
            self.populate(d)
        self.fill_images()
        self._state()
        if self.main.figures.find(self.STEP, self.FIG) is not None:
            self.draw(keep_view=True)
        self.main.load_panel._fill_rois(d)       # the ticked ROIs only

    def open(self):
        d = self.main.data
        if d is not None and self._populated is None:
            self.populate(d)
        self.fill_images()
        self._state()
        if d is not None and self.main.figures.find(self.STEP,
                                                    self.FIG) is None:
            self.draw()

    def fill_images(self, select=None):
        lp = self.main.load_panel
        keep = select or self.image.currentData(self._role)
        found = lib.find_background_images(lp.run_dir, lp.root_dir)
        paths = found + [p for p in self._extra if p not in found]
        cb = self.image
        cb.blockSignals(True)
        cb.clear()
        for p in paths:
            cb.addItem(os.path.basename(p), p)
            cb.setItemData(cb.count() - 1, p, E("Qt.ItemDataRole.ToolTipRole"))
        if not paths:
            cb.addItem("(no image found -- Browse...)", None)
        if keep in paths:
            cb.setCurrentIndex(paths.index(keep))
        cb.blockSignals(False)

    def browse(self):
        lp = self.main.load_panel
        path, _f = QtWidgets.QFileDialog.getOpenFileName(
            self, "Image", lp.run_dir or lp.root_dir or os.path.expanduser("~"),
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.mat);;"
            "All files (*)")
        if path:
            if path not in self._extra:
                self._extra.append(path)
            self.fill_images(select=path)
            self.draw()

    def _source(self):
        img = self.source.currentData(self._role) == "image"
        for w in (self.image_lab, self.image, self.browse_btn):
            w.setVisible(img)
        if self.main.data is not None:
            self.draw()

    # ---- figure ----------------------------------------------------------- #
    def draw(self, keep_view=False):
        self._recolour()
        d = self.main.data
        if d is None:
            return None
        if self.drawer is not None:              # keep the drawing armed
            keep_view = True
        source = self.source.currentData(self._role)
        frame = self.main.load_panel.display_options()["frame"]
        image = self.image.currentData(self._role)
        page = self.main.figures.find(self.STEP, self.FIG)
        view = None
        if page is not None and page.figure.axes and keep_view:
            ax = page.figure.axes[0]
            view = (ax.get_xlim(), ax.get_ylim())
        try:
            fig = lib.roi_figure(
                d, source, frame, image,
                rois=self.shapes("rect", checked_only=True, colours=True),
                lines=self.shapes("line", checked_only=True, colours=True),
                selected=self._selected_name(),
                fig=page.figure if page is not None else lib.new_figure(),
                **self.style.options())
        except Exception as exc:
            self.main.status("Figure error: %s" % exc, log=True)
            return None
        if view is not None:
            fig.axes[0].set_xlim(view[0])
            fig.axes[0].set_ylim(view[1])
        if page is None:
            page = self.main.figures.add(
                self.STEP, self.FIG, fig, data_fn=self._data,
                update_fn=lambda: self.draw(keep_view=True))
        page.redraw()
        if self.drawer is not None:              # the axes were redrawn
            self.drawer = self.drawer.rearm(page)
        return page

    def _data(self):
        out = {}
        for kind, (base, key) in self.KINDS.items():
            for name, pts in self.shapes(kind):
                out["%s_%s" % (key, name)] = np.asarray(pts, float)
        out["units"] = "m"
        return out

    # ---- add / remove ----------------------------------------------------- #
    def add(self, kind):
        d = self.main.data
        if d is None or not d.can_calibrate:
            self._state()
            return
        page = self.main.figures.find(self.STEP, self.FIG) or self.draw()
        if page is None:
            return
        if page.float_window is None:
            self.main.figures.focus(page)
        if self.drawer is not None:
            self.drawer.cancel()
        self.drawer = ShapeDrawer(self.main, page, kind,
                                  lambda pts, k=kind: self._added(k, pts))
        self._state()
        self.main.status("Drag on the figure with the left button held to "
                         "draw the %s (Esc cancels)"
                         % ("rectangle" if kind == "rect" else "segment"), 0)

    def _added(self, kind, pts):
        self.drawer = None
        if pts is None:
            self.main.status("Cancelled")
        else:
            name = (lib.roi_name(self.trees[kind].topLevelItemCount() + 1)
                    if kind == "rect" else self._next_name(kind))
            it = self._add_item(kind, name, pts)
            self.trees[kind].setCurrentItem(it)
            self._set_dirty()
            if kind == "rect":
                self._rois_to_run()
            self._recolour()
            self._maps_follow()
            self.main.status("%s added: %s" % (name, " / ".join(
                self._fmt(kind, pts))), log=True)
        self._state()
        self.draw(keep_view=True)

    def remove(self, kind):
        t = self.trees[kind]
        items = t.selectedItems()
        if not items:
            self.main.status("Select the one to remove in the list first")
            return
        for it in items:
            t.takeTopLevelItem(t.indexOfTopLevelItem(it))
            self.main.status("%s removed" % it.text(0), log=True)
        if kind == "rect":
            self._rois_to_run()                  # renumbered, quantities too
        self._set_dirty()
        self.draw(keep_view=True)
        self._maps_follow()

    # ---- parameter file --------------------------------------------------- #
    def update_param_file(self):
        """Apply: PTS_ROI and LINES of the lists used for the loaded run (in
        memory: no parameter file); Maps and Quiver and the run information
        follow (no reading again). Then the ROI quantities of the active
        ROIs (lib.compute_roi_quantities): those that exist are kept, the
        others calculated."""
        d, lp = self.main.data, self.main.load_panel
        if d is None:
            QtWidgets.QMessageBox.information(
                self, "Define ROI", "Load a run first (File > Load data).")
            return
        rois = lib.shapes_to_param(self.shapes("rect"))
        lines = lib.shapes_to_param(self.shapes("line"))
        d.params.update(PTS_ROI=rois, LINES=lines)
        self._set_dirty(False)
        self.main.status("ROI selection applied: %d ROI(s), %d line(s)"
                         % (len(rois or {}), len(lines or {})), log=True)
        self.calculate_quantities()
        self.main.mark_done("roi_apply")
        lp._fill_rois(d)
        self.main.draw_field()
        self.main.profiles_panel.sync()
        self.main.timeseries_panel.sync()
        self.main.fft_time_panel.sync()
        self.main.fft_space_panel.sync()
        self.main.show_info(d)

    apply = update_param_file

    # ---- ROI file --------------------------------------------------------- #
    def roi_record(self):
        import datetime
        d = self.main.data
        st = self.get_state()
        return {
            "kind": "PIV_GUI roi", "version": 1,
            "saved": datetime.datetime.now().isoformat(timespec="seconds"),
            "run": d.run if d is not None else None,
            "run_folder": d.run_dir if d is not None else None,
            "units": "m",
            "PTS_ROI": lib.shapes_to_param(self.shapes("rect")),
            "LINES": lib.shapes_to_param(self.shapes("line")),
            "lists": st["lists"], "applied": not self.dirty,
            "figure": {"source": st["source"], "image": st["image"],
                       "style": st["style"]},
        }

    def save_rois(self, path=None):
        """Save: Apply first (the loaded run, the ROI and line quantities),
        then the ROI file (asks where; proposed in the root folder). Returns
        the path."""
        if self.main.data is not None:
            self.update_param_file()
        rec = self.roi_record()
        if not rec["PTS_ROI"] and not rec["LINES"]:
            QtWidgets.QMessageBox.information(
                self, "Save ROI selection", "No ROI or line to save.")
            return None
        if path is None:
            d = self.main.data
            folder = self.main.setup_folder()
            path, _f = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save ROI selection", os.path.join(
                    folder, (d.run if d is not None else "roi") + ROI_EXT),
                json_filter("ROI selection", ROI_EXT))
            if not path:
                return None
        path = with_double_ext(path, ROI_EXT)
        try:
            with open(path, "w") as fh:
                json.dump(lib._jsonable(rec), fh, indent=1)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save ROI selection",
                                          "%s: %s" % (path, exc))
            return None
        self.roi_file_label.setText("Saved: %s" % path)
        self.roi_file_label.setVisible(True)
        self.main.setup_file_used("roi", path)
        self.main.status("ROI selection saved in %s: %d ROI(s), %d line(s)"
                         % (path, len(rec["PTS_ROI"] or {}),
                            len(rec["LINES"] or {})), log=True)
        return path

    def fill_from(self, rec):
        """The lists and the figure settings of an ROI file."""
        fig = rec.get("figure") or {}
        self.set_state(dict(source=fig.get("source"), image=fig.get("image"),
                            style=fig.get("style")))
        lists = rec.get("lists")
        if not lists:                            # only the coordinates
            lists = {kind: [[n, p, True] for n, p in lib.parse_shapes(
                rec.get(key), base)]
                for kind, (base, key) in self.KINDS.items()}
        self.restore_lists(lists, False)

    def load_rois(self, path=None, run=True):
        """Load ROI selection: read an ROI file, set the lists (names,
        coordinates, ticks) and the figure, then Apply. Returns the record."""
        if path is None:
            start = self.main.setup_folder()
            path, _f = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load ROI selection", start,
                json_filter("ROI selection", ROI_EXT))
            if not path:
                return None
        try:
            rec = read_setup_file(path, "roi")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load ROI selection",
                                          "%s: %s" % (path, exc))
            return None
        self.fill_from(rec)
        self.roi_file_label.setText("Loaded: %s" % path)
        self.roi_file_label.setVisible(True)
        self.main.setup_file_used("roi", path)
        self.main.write_log("ROI selection loaded from %s" % path)
        d = self.main.data
        if run and d is not None:
            if not d.can_calibrate:
                self.main.write_log("WARNING: the run is not calibrated: the "
                                    "ROIs (in m) are not applied")
            else:
                self.update_param_file()
        if self.main.figures.find(self.STEP, self.FIG) is not None:
            self.draw(keep_view=True)
        return rec

    def calculate_quantities(self):
        """The ROI quantities '<quantity>_ROIxx' that do not exist yet, for
        the active ROIs (ticked) only."""
        d = self.main.data
        if d is None:
            return
        if not d.can_calibrate:
            self.main.write_log("    ROI quantities: none (the run cannot "
                                "be calibrated: no ROI field)")
            return
        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(E("Qt.CursorShape.WaitCursor")))
        try:
            active = self.shown("rect")
            done, kept, gone = lib.compute_roi_quantities(
                d, calibrated=True, rois=None if active is None
                else set(active))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if gone:
            self.main.write_log("    ROI quantities deleted (ROI removed "
                                "or renumbered): " + ", ".join(gone))
        if kept:
            self.main.write_log("    ROI quantities kept (already "
                                "calculated): " + ", ".join(kept))
        if done:
            self.main.write_log("    ROI quantities calculated (all %d "
                                "frames): %s" % (d.nt, ", ".join(done)))
        off = [i["roi"] for i in d.roi_fields().values()
               if active is not None and i["roi"] not in active]
        if off:
            self.main.write_log("    not calculated (ROI not active: not "
                                "ticked): " + ", ".join(off))
        # the quantities along the ticked lines: full field and each active
        # ROI they cross, every frame (Processing > Profiles uses them)
        shown = self.shown("line")
        lines = {n: pts for n, pts in self.shapes("line")
                 if shown is None or n in shown}
        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(E("Qt.CursorShape.WaitCursor")))
        try:
            ldone, lkept, lgone = lib.compute_line_quantities(
                d, lines, rois=None if active is None else set(active))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if lgone:
            self.main.write_log("    line quantities deleted (line or ROI "
                                "removed / unticked): %d" % len(lgone))
        if ldone or lkept:
            regions = sorted({e["line"] + " in " + e["roi"] for e in
                              d.line_quantities.values()})
            self.main.write_log(
                "    quantities along the lines: %d calculated, %d kept "
                "(all %d frames): %s" % (len(ldone), len(lkept), d.nt,
                                         ", ".join(regions)))
        self.main.status("ROI quantities: %d calculated, %d kept; along the "
                         "lines: %d calculated, %d kept" % (
                             len(done), len(kept), len(ldone), len(lkept)))
        self._state()

    # ---- session ---------------------------------------------------------- #
    def get_state(self):
        lists = {}
        for kind, t in self.trees.items():
            lists[kind] = [[t.topLevelItem(i).text(0),
                            t.topLevelItem(i).data(0, self._role),
                            t.topLevelItem(i).checkState(0)
                            == E("Qt.CheckState.Checked")]
                           for i in range(t.topLevelItemCount())]
        return dict(source=self.source.currentData(self._role),
                    image=self.image.currentData(self._role),
                    style=self.style.get_state(), lists=lists,
                    dirty=bool(self.dirty))

    def restore_lists(self, lists, dirty=False):
        """Sessions: the ROI and line lists (with their ticks) as saved --
        also edits not applied yet."""
        if not lists:
            return
        d = self.main.data
        for kind, rows in lists.items():
            if kind not in self.trees:
                continue
            t = self.trees[kind]
            t.clear()
            for name, pts, on in rows:
                self._add_item(kind, name, pts, checked=bool(on))
        if d is not None:
            shapes = self.shapes("rect")
            if lib.shapes_to_param(shapes) != d.params.get("PTS_ROI"):
                self._rois_to_run()
        self._recolour()
        self._set_dirty(bool(dirty))
        self._maps_follow()

    def set_state(self, s):
        self.style.set_state(s.get("style"))
        if s.get("image") and os.path.isfile(s["image"]):
            if s["image"] not in self._extra:
                self._extra.append(s["image"])
            self.fill_images(select=s["image"])
        i = self.source.findData(s.get("source"), self._role)
        if i >= 0:
            self.source.setCurrentIndex(i)


class MaskPanel(QtWidgets.QWidget):
    """Pre-processing > Mask.

    Masks: the masks already built, each with a tick box (ticked: shaded on
    the figure, each in its own colour), an editable name; Remove mask.
    Figure: the current frame of Load data, or an image file (chosen with a
    file dialog). Select masking area: the tool (polygon, rectangle,
    circle), the list of the areas (each with a tick box), Add (draws a new
    area with the tool) and Remove; New mask from selection (a new mask
    from the ticked areas, added to the list of masks) and Add selection to
    mask (the ticked areas added to every ticked mask).
    The mask applied -- velocity NaN in it for every step -- is chosen in
    Processing > Maps and Quiver > Mask. Masks and areas are kept in the run
    information (run_information.json)."""

    STEP = "mask"
    FIG = "Mask"
    SOURCES = [("Current Frame", "frame"), ("Select Image", "image")]

    def __init__(self, main):
        super().__init__()
        self.main = main
        self._role = E("Qt.ItemDataRole.UserRole")
        self.image_path = None
        self.drawer = None
        lay = QtWidgets.QVBoxLayout(self)
        self.mask_load_btn = QtWidgets.QPushButton("Load mask...")
        self.mask_load_btn.setMinimumHeight(32)
        self.mask_load_btn.setToolTip(
            "read a mask file (.pivmask.json, saved with Save below) and run "
            "this step again with it: the masks, the selection areas, the "
            "masks shown and the mask applied (File > Load data can also "
            "load it with the run)")
        self.mask_load_btn.clicked.connect(lambda: self.load_masks())
        lay.addWidget(self.mask_load_btn)
        self.mask_file_label = QtWidgets.QLabel("")
        self.mask_file_label.setWordWrap(True)
        self.mask_file_label.setStyleSheet("color: gray;")
        self.mask_file_label.setVisible(False)
        lay.addWidget(self.mask_file_label)

        # ---- the masks already built
        box = QtWidgets.QGroupBox("Masks")
        v = QtWidgets.QVBoxLayout(box)
        self.mtree = QtWidgets.QTreeWidget()
        self.mtree.setHeaderLabels(["Mask", "areas", "vectors"])
        self.mtree.setRootIsDecorated(False)
        self.mtree.setMinimumHeight(80)
        self.mtree.setMaximumHeight(160)
        self.mtree.setToolTip(
            "the masks built: ticked ones are shaded on the figure, each in "
            "its colour; double click a name to rename it. The mask applied "
            "(velocity NaN for every step) is chosen in Processing > Maps "
            "and Quiver > Mask")
        self.mtree.itemChanged.connect(self._mask_changed)
        v.addWidget(self.mtree)
        row = QtWidgets.QHBoxLayout()
        self.mrem_btn = QtWidgets.QPushButton("Remove mask")
        self.mrem_btn.setToolTip("remove the selected mask")
        self.mrem_btn.clicked.connect(self.remove_mask)
        self.mupd_btn = QtWidgets.QPushButton("Save")
        self.mupd_btn.setToolTip(
            "save the masks in a mask file (.pivmask.json): every mask (its "
            "areas, points in px, colour), the selection areas, the masks "
            "shown, the mask applied and the figure -- everything needed "
            "to run this step again (Load mask, File > Load data)")
        self.mupd_btn.clicked.connect(lambda: self.save_masks())
        row.addWidget(self.mupd_btn)
        row.addStretch(1)
        row.addWidget(self.mrem_btn)
        v.addLayout(row)
        self.mask_lab = QtWidgets.QLabel("")
        self.mask_lab.setWordWrap(True)
        v.addWidget(self.mask_lab)
        lay.addWidget(box)

        # ---- the figure
        box = QtWidgets.QGroupBox("Figure")
        g = QtWidgets.QGridLayout(box)
        self.source = QtWidgets.QComboBox()
        for lab, key in self.SOURCES:
            self.source.addItem(lab, key)
        self.source.setToolTip(
            "the figure to draw on: the current frame of Load data, or an "
            "image file (Select Image opens a file dialog)")
        self.image_lab = QtWidgets.QLabel("")
        self.image_lab.setWordWrap(True)
        self.change_btn = QtWidgets.QPushButton("Change...")
        self.change_btn.setMaximumWidth(90)
        g.addWidget(QtWidgets.QLabel("Select from"), 0, 0)
        g.addWidget(self.source, 0, 1, 1, 2)
        g.addWidget(self.image_lab, 1, 0, 1, 2)
        g.addWidget(self.change_btn, 1, 2)
        g.setColumnStretch(1, 1)
        lay.addWidget(box)
        self.style = MapStyleBox()
        lay.addWidget(self.style)
        self.style.changed.connect(lambda: self.draw(keep_view=True))
        self.source.activated.connect(lambda *_: self._source(ask=True))
        self.change_btn.clicked.connect(lambda: self.choose_image())

        # ---- the selection areas
        box = QtWidgets.QGroupBox("Select masking area")
        v = QtWidgets.QVBoxLayout(box)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Select tool"))
        self.tool = QtWidgets.QComboBox()
        for lab, key in lib.MASK_TOOLS:
            self.tool.addItem(lab, key)
        self.tool.setToolTip(
            "Polygon: every left click is a new vertex; Enter or the right "
            "button closes it (Backspace: last vertex off). Rectangle: drag "
            "from one corner to the other; Circle: drag from the centre to "
            "the edge -- then they stay editable (drag inside: move; drag a "
            "corner / the edge: resize) until a right click or Enter. Esc "
            "cancels.")
        row.addWidget(self.tool, 1)
        v.addLayout(row)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Area", "tool", "size"])
        self.tree.setRootIsDecorated(False)
        self.tree.setMinimumHeight(100)
        self.tree.setMaximumHeight(200)
        self.tree.setToolTip("ticked areas make the selection used by the "
                             "two buttons below; double click a name to "
                             "rename it")
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.itemSelectionChanged.connect(
            lambda: self.draw(keep_view=True))
        v.addWidget(self.tree)
        row = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("Add")
        self.rem_btn = QtWidgets.QPushButton("Remove")
        self.add_btn.setToolTip("draw a new area on the figure with the "
                                "selected tool")
        self.rem_btn.setToolTip("remove the selected area")
        self.add_btn.clicked.connect(self.add)
        self.rem_btn.clicked.connect(self.remove)
        row.addWidget(self.add_btn)
        row.addWidget(self.rem_btn)
        v.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.new_btn = QtWidgets.QPushButton("New mask from selection")
        self.addto_btn = QtWidgets.QPushButton("Add selection to mask")
        self.new_btn.setToolTip("a new mask made of the ticked areas, added "
                                "to the list of masks")
        self.addto_btn.setToolTip("add the ticked areas to the masks ticked "
                                  "in the list of masks")
        for b in (self.new_btn, self.addto_btn):
            b.setMinimumHeight(32)
            row.addWidget(b)
        self.new_btn.clicked.connect(self.new_mask)
        self.addto_btn.clicked.connect(self.add_to_masks)
        v.addLayout(row)
        lay.addWidget(box)
        lay.addStretch(1)
        self._source()
        self._state()

    # ---- the selection areas ---------------------------------------------- #
    def _size(self, kind, pts_px):
        d = self.main.data
        xs, ys = d.view().pix_scale if d is not None else (1.0, 1.0)
        u = d.view().lunit if d is not None else "px"
        p = np.asarray(pts_px, float) * (xs, ys)
        if kind == "circle":
            return "r = %.4g %s" % (np.hypot(*(p[1] - p[0])), u)
        if kind == "rect":
            w, h = np.abs(p[1] - p[0])
            return "%.4g x %.4g %s" % (w, h, u)
        return "%d vertices" % len(p)

    def _add_item(self, name, kind, pts_px, use=True):
        self.tree.blockSignals(True)
        lab = dict((k, l) for l, k in lib.MASK_TOOLS).get(kind, kind)
        it = QtWidgets.QTreeWidgetItem([name, lab, self._size(kind, pts_px)])
        it.setFlags(it.flags() | E("Qt.ItemFlag.ItemIsUserCheckable")
                    | E("Qt.ItemFlag.ItemIsEditable"))
        it.setCheckState(0, E("Qt.CheckState.Checked") if use
                         else E("Qt.CheckState.Unchecked"))
        it.setData(0, self._role, {"kind": kind, "points_px":
                                   [list(map(float, p)) for p in pts_px]})
        it.setData(1, self._role, name)
        self.tree.addTopLevelItem(it)
        self.tree.blockSignals(False)
        for c in range(3):
            self.tree.resizeColumnToContents(c)
        return it

    def areas(self, used_only=False):
        """[{name, kind, points_px, use}] of the list of areas."""
        out = []
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            use = it.checkState(0) == E("Qt.CheckState.Checked")
            if used_only and not use:
                continue
            a = dict(it.data(0, self._role))
            a.update(name=it.text(0), use=use)
            out.append(a)
        return out

    @staticmethod
    def _unique_in(names, base):
        names = set(names)
        if base not in names:
            return base
        i = 2
        while "%s_%d" % (base, i) in names:
            i += 1
        return "%s_%d" % (base, i)

    def _area_names(self, but=None):
        return [self.tree.topLevelItem(i).text(0)
                for i in range(self.tree.topLevelItemCount())
                if self.tree.topLevelItem(i) is not but]

    def _item_changed(self, it, col):
        if col == 0:
            new, old = it.text(0).strip(), it.data(1, self._role)
            if new != old:
                if not new or new in self._area_names(but=it):
                    new = self._unique_in(self._area_names(but=it),
                                          new or old)
                self.tree.blockSignals(True)
                it.setText(0, new)
                it.setData(1, self._role, new)
                self.tree.blockSignals(False)
        self._sync_record()
        self._state()
        self.draw(keep_view=True)

    def _sync_record(self):
        d = self.main.data
        if d is not None:
            d.mask_record = {"areas": self.areas()}
            d.params["MASK_SELECTION"] = d.mask_record["areas"]

    def _selected_name(self):
        items = self.tree.selectedItems()
        return items[0].text(0) if items else None

    # ---- the masks -------------------------------------------------------- #
    def _swatch(self, colour):
        pm = QtGui.QPixmap(12, 12)
        pm.fill(QtGui.QColor(colour))
        return QtGui.QIcon(pm)

    def fill_masks(self, keep_ticks=True):
        d = self.main.data
        ticks = {}
        if keep_ticks:
            for i in range(self.mtree.topLevelItemCount()):
                it = self.mtree.topLevelItem(i)
                ticks[it.text(0)] = it.checkState(0)
        self.mtree.blockSignals(True)
        self.mtree.clear()
        for name, m in (d.masks.items() if d is not None else []):
            n = int(np.sum(m["mask"])) if m["mask"] is not None else 0
            it = QtWidgets.QTreeWidgetItem([name, "%d" % len(m["areas"]),
                                            "%d" % n])
            it.setIcon(0, self._swatch(m["colour"]))
            it.setFlags(it.flags() | E("Qt.ItemFlag.ItemIsUserCheckable")
                        | E("Qt.ItemFlag.ItemIsEditable"))
            it.setCheckState(0, ticks.get(name, E("Qt.CheckState.Checked")))
            it.setData(0, self._role, name)
            it.setToolTip(0, ", ".join(str(a.get("name"))
                                       for a in m["areas"]))
            self.mtree.addTopLevelItem(it)
        self.mtree.blockSignals(False)
        for c in range(3):
            self.mtree.resizeColumnToContents(c)

    def masks_ticked(self):
        return [self.mtree.topLevelItem(i).data(0, self._role)
                for i in range(self.mtree.topLevelItemCount())
                if self.mtree.topLevelItem(i).checkState(0) ==
                E("Qt.CheckState.Checked")]

    def _mask_changed(self, it, col):
        d = self.main.data
        if d is not None and col == 0:
            old, new = it.data(0, self._role), it.text(0).strip()
            if new != old:                       # renamed
                others = [k for k in d.masks if k != old]
                if not new or new in others:
                    new = self._unique_in(others, new or old)
                d.rename_mask(old, new)
                self.mtree.blockSignals(True)
                it.setText(0, new)
                it.setData(0, self._role, new)
                self.mtree.blockSignals(False)
                self.main.status("Mask %s renamed %s" % (old, new), log=True)
                self._masks_changed()
                return
        self.draw(keep_view=True)

    def _masks_changed(self, applied_changed=False):
        """The list of masks changed: the parameter file (MASKS), the Mask
        list of Maps and Quiver, the run information (saved), the
        figures."""
        d = self.main.data
        self.save_param_file(d)
        self.main.load_panel.fill_masks(d)
        self._state()
        self.main.show_info(d)
        if applied_changed:
            self.main.draw_field()
        self.draw(keep_view=True)

    def save_param_file(self, d):
        """The masks with the run's parameters in memory (MASKS; no
        parameter file any more), kept when the run is loaded again."""
        if d is None:
            return
        d.params["MASKS"] = lib.masks_to_param(d)
        d.params["MASK_APPLIED"] = d.mask_name

    # ---- mask file -------------------------------------------------------- #
    def mask_record_file(self):
        import datetime
        d = self.main.data
        st = self.get_state()
        return {
            "kind": "PIV_GUI mask", "version": 1,
            "saved": datetime.datetime.now().isoformat(timespec="seconds"),
            "run": d.run if d is not None else None,
            "run_folder": d.run_dir if d is not None else None,
            "units": "px of the run's PIV grid",
            "MASKS": lib.masks_to_param(d) if d is not None else None,
            "MASK_APPLIED": d.mask_name if d is not None else None,
            "MASK_SELECTION": self.areas(),
            "masks_shown": st["masks_shown"],
            "figure": {"source": st["source"], "image": st["image"],
                       "tool": st["tool"], "style": st["style"]},
        }

    def save_masks(self, path=None):
        """Save: the mask file (asks where; proposed
        <run>/PostProcessing/<run>.pivmask.json). Returns the path."""
        rec = self.mask_record_file()
        if not rec["MASKS"] and not rec["MASK_SELECTION"]:
            QtWidgets.QMessageBox.information(
                self, "Save mask", "No mask or area to save.")
            return None
        if path is None:
            d = self.main.data
            folder = self.main.setup_folder()
            path, _f = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save mask", os.path.join(
                    folder, (d.run if d is not None else "mask") + MASK_EXT),
                json_filter("Mask", MASK_EXT))
            if not path:
                return None
        path = with_double_ext(path, MASK_EXT)
        try:
            with open(path, "w") as fh:
                json.dump(lib._jsonable(rec), fh, indent=1)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save mask",
                                          "%s: %s" % (path, exc))
            return None
        self.mask_file_label.setText("Saved: %s" % path)
        self.mask_file_label.setVisible(True)
        self.main.setup_file_used("mask", path)
        self.main.status("Masks saved in %s: %s (applied: %s)" % (
            path, ", ".join(rec["MASKS"] or {}) or "none",
            rec["MASK_APPLIED"] or "none"), log=True)
        return path

    def after_load(self, path, rec):
        """Load data loaded the run with this mask file (masks, mask
        applied, selection areas already in): the rest of the step (masks
        shown, figure)."""
        fig = rec.get("figure") or {}
        self.set_state(dict(source=fig.get("source"), image=fig.get("image"),
                            tool=fig.get("tool"), style=fig.get("style")))
        self.restore_shown(rec.get("masks_shown"))
        self.mask_file_label.setText("Loaded: %s" % path)
        self.mask_file_label.setVisible(True)
        self.main.setup_file_used("mask", path)
        d = self.main.data
        self.main.write_log("Mask step run from %s: %s (applied: %s)" % (
            path, ", ".join(d.masks) if d is not None else "", (
                d.mask_name if d is not None else None) or "none"))

    def load_masks(self, path=None):
        """Load mask: read a mask file and run this step again with it --
        the masks (replacing those in memory), the selection areas, the
        masks shown, the mask applied, the figure. Returns the record."""
        d = self.main.data
        if path is None:
            start = self.main.setup_folder()
            path, _f = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load mask", start,
                json_filter("Mask", MASK_EXT))
            if not path:
                return None
        try:
            rec = read_setup_file(path, "mask")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load mask",
                                          "%s: %s" % (path, exc))
            return None
        if d is None:
            QtWidgets.QMessageBox.information(
                self, "Load mask", "Load a run first (File > Load data): "
                "the masks are on its PIV grid.")
            return None
        for name in list(d.masks):
            d.remove_mask(name)
        for name, m in (rec.get("MASKS") or {}).items():
            try:
                d.store_mask(str(name), m.get("areas") or [], m.get("colour"))
            except Exception as exc:
                self.main.write_log("WARNING: mask %s not built: %s"
                                    % (name, exc))
        d.mask_record = {"areas": list(rec.get("MASK_SELECTION") or [])}
        d.apply_mask(rec.get("MASK_APPLIED"))
        self.populate(d)
        self.save_param_file(d)
        self._sync_record()
        self.main.load_panel.fill_masks(d)
        self.after_load(path, rec)
        self._state()
        self.main.show_info(d)
        self.main.draw_field()
        self.draw(keep_view=True)
        self.main.profiles_panel.sync()
        self.main.timeseries_panel.sync()
        self.main.fft_time_panel.sync()
        self.main.fft_space_panel.sync()
        return rec

    def new_mask(self):
        """A new mask made of the ticked areas, added to the list."""
        d = self.main.data
        use = self.areas(used_only=True)
        if d is None or not use:
            self.main.status("Tick the areas of the new mask first")
            return
        name = self._unique_in(d.masks, "Mask%d" % (len(d.masks) + 1))
        d.store_mask(name, use)
        self._sync_record()
        n = d.masks[name]["mask"]
        self.fill_masks()
        self.main.status("%s: %d area(s), %d vectors -- apply it in "
                         "Processing > Maps and Quiver > Mask" % (
                             name, len(use), int(np.sum(n))
                             if n is not None else 0), log=True)
        self._masks_changed()

    def add_to_masks(self):
        """The ticked areas added to every ticked mask."""
        d = self.main.data
        use = self.areas(used_only=True)
        targets = self.masks_ticked()
        if d is None or not use:
            self.main.status("Tick the areas to add first")
            return
        if not targets:
            self.main.status("Tick the masks to add the areas to first")
            return
        for name in targets:
            old = d.masks[name]["areas"]
            have = [a.get("name") for a in old]
            new = [a for a in use if a.get("name") not in have]
            d.store_mask(name, old + new)
            self.main.status("%s: %d area(s) added (%d now)" % (
                name, len(new), len(old) + len(new)), log=True)
        self._sync_record()
        self.fill_masks()
        self._masks_changed(applied_changed=d.mask_name in targets)

    def remove_mask(self):
        d = self.main.data
        items = self.mtree.selectedItems()
        if d is None or not items:
            self.main.status("Select the mask to remove in the list first")
            return
        applied = False
        for it in items:
            name = it.data(0, self._role)
            applied = applied or name == d.mask_name
            d.remove_mask(name)
            self.main.status("Mask %s removed" % name, log=True)
        self.fill_masks()
        self._masks_changed(applied_changed=applied)

    def _state(self):
        d = self.main.data
        busy = self.drawer is not None
        self.add_btn.setEnabled(d is not None and not busy)
        self.rem_btn.setEnabled(not busy)
        for b in (self.new_btn, self.addto_btn, self.mrem_btn,
                  self.mupd_btn):
            b.setEnabled(d is not None and not busy)
        if d is None:
            self.mask_lab.setText("Load a run first (File > Load data).")
        elif d.mask is None:
            self.mask_lab.setText("Applied (Processing > Maps and Quiver > Mask): none")
        else:
            self.mask_lab.setText(
                "Applied (Processing > Maps and Quiver > Mask): <b>%s</b>, %d vectors (%.1f "
                "%%) -> velocity NaN" % (d.mask_name, int(d.mask.sum()),
                                         100 * d.mask.mean()))

    # ---- filling ---------------------------------------------------------- #
    def populate(self, d):
        if self.drawer is not None:
            self.drawer.cancel()
        self.tree.clear()
        for a in d.mask_record.get("areas") or []:
            try:
                self._add_item(str(a.get("name")), a["kind"],
                               a["points_px"], a.get("use", True))
            except Exception:
                continue
        self.fill_masks(keep_ticks=False)
        self._state()

    def data_loaded(self, d):
        shown = (self.get_state()["masks_shown"]
                 if self.mtree.topLevelItemCount() else None)
        self.populate(d)
        self.restore_shown(shown)                # kept at a reload
        if self.main.figures.find(self.STEP, self.FIG) is not None:
            self.draw(keep_view=True)

    def open(self):
        d = self.main.data
        self._state()
        if d is not None and self.main.figures.find(self.STEP,
                                                    self.FIG) is None:
            self.draw()

    def choose_image(self):
        lp = self.main.load_panel
        path, _f = QtWidgets.QFileDialog.getOpenFileName(
            self, "Image for the mask",
            os.path.dirname(self.image_path or "") or lp.run_dir
            or lp.root_dir or os.path.expanduser("~"),
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.mat);;"
            "All files (*)")
        if path:
            self.image_path = path
            self._source()
            return True
        return False

    def _source(self, ask=False):
        img = self.source.currentData(self._role) == "image"
        if img and ask and (not self.image_path or
                            not os.path.isfile(self.image_path)):
            if not self.choose_image():          # cancelled: back to frame
                self.source.setCurrentIndex(0)
                img = False
        elif img and ask:
            self.choose_image()
        self.image_lab.setText(os.path.basename(self.image_path or "")
                               or "(no image)")
        for w in (self.image_lab, self.change_btn):
            w.setVisible(img)
        if self.main.data is not None:
            self.draw()

    # ---- figure ----------------------------------------------------------- #
    def draw(self, keep_view=False):
        d = self.main.data
        if d is None:
            return None
        if self.drawer is not None:
            keep_view = True
        view = d.view()
        xs, ys = view.pix_scale
        shown = [(a["name"], a["kind"],
                  (np.asarray(a["points_px"], float) * (xs, ys)).tolist(),
                  a["use"]) for a in self.areas()]
        masks = [(n, d.masks[n]["mask"], d.masks[n]["colour"])
                 for n in self.masks_ticked() if n in d.masks]
        page = self.main.figures.find(self.STEP, self.FIG)
        lims = None
        if page is not None and page.figure.axes and keep_view:
            ax = page.figure.axes[0]
            lims = (ax.get_xlim(), ax.get_ylim())
        try:
            fig = lib.mask_figure(
                d, self.source.currentData(self._role),
                self.main.load_panel.display_options()["frame"],
                self.image_path, shown, masks, self._selected_name(),
                fig=page.figure if page is not None else lib.new_figure(),
                **self.style.options())
        except Exception as exc:
            self.main.status("Figure error: %s" % exc, log=True)
            return None
        if lims is not None:
            fig.axes[0].set_xlim(lims[0])
            fig.axes[0].set_ylim(lims[1])
        if page is None:
            page = self.main.figures.add(
                self.STEP, self.FIG, fig, data_fn=self._data,
                update_fn=lambda: self.draw(keep_view=True))
        page.redraw()
        if self.drawer is not None:              # the axes were redrawn
            self.drawer = self.drawer.rearm(page)
        return page

    def _data(self):
        d = self.main.data
        out = {"units": "px (points of the areas)"}
        for a in self.areas():
            out["area_%s_%s" % (a["name"], a["kind"])] = np.asarray(
                a["points_px"], float)
        if d is not None:
            for n, m in d.masks.items():
                if m["mask"] is not None:
                    out["mask_%s" % n] = m["mask"].astype(np.uint8)
            out["mask_applied"] = d.mask_name or ""
            out["x"], out["z"] = d.view().x, d.view().z
        return out

    # ---- add / remove areas ----------------------------------------------- #
    def add(self):
        d = self.main.data
        if d is None:
            return
        page = self.main.figures.find(self.STEP, self.FIG) or self.draw()
        if page is None:
            return
        if page.float_window is None:
            self.main.figures.focus(page)
        if self.drawer is not None:
            self.drawer.cancel()
        kind = self.tool.currentData(self._role)
        self.drawer = ShapeDrawer(self.main, page, kind,
                                  lambda pts, k=kind: self._added(k, pts),
                                  edit=True)
        self._state()
        self.main.status({
            "poly": "Left click: a new vertex; Enter or right click closes "
                    "the polygon (Backspace: last vertex off, Esc cancels)",
            "rect": "Drag a rectangle with the left button held; then move "
                    "/ resize it, and validate with a right click or Enter "
                    "(Esc cancels)",
            "circle": "Drag from the centre to the edge with the left "
                      "button held; then move / resize it, and validate "
                      "with a right click or Enter (Esc cancels)"}[kind], 0)

    def _added(self, kind, pts):
        self.drawer = None
        if pts is None:
            self.main.status("Cancelled")
        else:
            xs, ys = self.main.data.view().pix_scale
            px = [(p[0] / xs, p[1] / ys) for p in pts]      # -> px
            name = self._unique_in(self._area_names(), "Area%d" % (
                self.tree.topLevelItemCount() + 1))
            it = self._add_item(name, kind, px)
            self.tree.setCurrentItem(it)
            self._sync_record()
            self.main.status("%s added (%s, %s)" % (
                name, kind, self._size(kind, px)), log=True)
        self._state()
        self.draw(keep_view=True)

    def remove(self):
        items = self.tree.selectedItems()
        if not items:
            self.main.status("Select the area to remove in the list first")
            return
        for it in items:
            self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(it))
            self.main.status("%s removed" % it.text(0), log=True)
        self._sync_record()
        self._state()
        self.draw(keep_view=True)

    # ---- session ---------------------------------------------------------- #
    def get_state(self):
        shown = [self.mtree.topLevelItem(i).text(0)
                 for i in range(self.mtree.topLevelItemCount())
                 if self.mtree.topLevelItem(i).checkState(0)
                 == E("Qt.CheckState.Checked")]
        return dict(source=self.source.currentData(self._role),
                    image=self.image_path, tool=self.tool.currentData(
                        self._role), style=self.style.get_state(),
                    masks_shown=shown)

    def restore_shown(self, names):
        """Sessions: the masks ticked (shown) in the list."""
        if names is None:
            return
        for i in range(self.mtree.topLevelItemCount()):
            it = self.mtree.topLevelItem(i)
            on = it.text(0) in names
            if (it.checkState(0) == E("Qt.CheckState.Checked")) != on:
                it.setCheckState(0, E("Qt.CheckState.Checked") if on
                                 else E("Qt.CheckState.Unchecked"))

    def set_state(self, s):
        self.style.set_state(s.get("style"))
        if s.get("image") and os.path.isfile(s["image"]):
            self.image_path = s["image"]
        i = self.tool.findData(s.get("tool"), self._role)
        if i >= 0:
            self.tool.setCurrentIndex(i)
        i = self.source.findData(s.get("source"), self._role)
        if i >= 0:
            self.source.setCurrentIndex(i)
            self._source()


# --------------------------------------------------------------------------- #
#  Log (as in the DTS GUI): printed text and errors go to the log panel
# --------------------------------------------------------------------------- #
class LogStream(QtCore.QObject):
    """File-like object: what is printed (any thread) is sent to the log
    panel -- through a Qt signal, so from a thread too -- and still to the
    terminal."""

    text = Signal(str)

    def __init__(self, terminal):
        super().__init__()
        self.terminal = terminal

    def write(self, text):
        if text:
            self.text.emit(text)
            if self.terminal is not None:
                try:
                    self.terminal.write(text)
                except Exception:
                    pass
        return len(text)

    def flush(self):
        if self.terminal is not None:
            try:
                self.terminal.flush()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
#  Main window
# --------------------------------------------------------------------------- #
def _session_stem(path):
    """'a/b/my_proc.pivsession.json' -> 'my_proc'."""
    b = os.path.basename(path or "session")
    for ext in (".pivsession.json", ".json"):
        if b.endswith(ext):
            return b[:-len(ext)]
    return os.path.splitext(b)[0]


def _path_row(label, tip, browse):
    """(layout, line edit) of a path with a Browse... button."""
    row = QtWidgets.QHBoxLayout()
    row.addWidget(QtWidgets.QLabel(label))
    ed = QtWidgets.QLineEdit()
    ed.setToolTip(tip)
    row.addWidget(ed, 1)
    b = QtWidgets.QPushButton("Browse...")
    b.clicked.connect(browse)
    row.addWidget(b)
    return row, ed


class ApplySessionDialog(QtWidgets.QDialog):
    """File > Apply session steps to another dataset: a session file and a
    data set (run folder); Run session loads the session, changes its
    paths to the new data set, loads it with the session's parameters and
    performs the steps of the session again; the run information (table
    and run_information.json) is that of the new data set."""

    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Apply session steps to another dataset")
        self.resize(760, 220)
        lay = QtWidgets.QVBoxLayout(self)
        row, self.session = _path_row(
            "Session file", "the session (*.pivsession.json) whose steps are "
            "applied", self.browse_session)
        lay.addLayout(row)
        row, self.dataset = _path_row(
            "Data set", "the run folder (with its PIVlab .mat) to process",
            self.browse_dataset)
        lay.addLayout(row)
        note = QtWidgets.QLabel(
            "The run is loaded with the parameters of the session "
            "(calibration, ROIs, lines, masks); the "
            "steps performed in the session (calibration, ROI Apply, FFT, "
            "...) are performed again on it, and its run information "
            "(table and run_information.json) is made.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        lay.addWidget(note)
        self.state = QtWidgets.QLabel("")
        self.state.setWordWrap(True)
        lay.addWidget(self.state)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        self.run_btn = QtWidgets.QPushButton("Run session")
        f = self.run_btn.font(); f.setBold(True); self.run_btn.setFont(f)
        self.run_btn.clicked.connect(self.run)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.close)
        row.addWidget(self.run_btn)
        row.addWidget(close)
        lay.addLayout(row)

    def browse_session(self):
        p, _f = QtWidgets.QFileDialog.getOpenFileName(
            self, "Session file", self.main.last_dir(), SESSION_EXT)
        if p:
            self.session.setText(p)

    def browse_dataset(self):
        p = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Data set (run folder)", self.main.last_dir())
        if p:
            self.dataset.setText(p)

    def run(self):
        sp, rd = self.session.text().strip(), self.dataset.text().strip()
        if not os.path.isfile(sp) or not os.path.isdir(rd):
            self.state.setText("<b>Choose a session file and a run folder "
                               "first.</b>")
            return
        self.run_btn.setEnabled(False)
        self.state.setText("Running the session on %s ..."
                           % os.path.basename(rd))
        if not self.main.apply_session(sp, rd, done=self._done):
            self.run_btn.setEnabled(True)

    def _done(self, ok, msg):
        self.run_btn.setEnabled(True)
        self.state.setText(("<b>Done.</b> " if ok else "<b>Failed:</b> ")
                           + msg)


class BatchDialog(QtWidgets.QDialog):
    """File > Batch processing: a root folder; a session (the processing
    to apply: loaded and run on its own run, so that the variables it
    makes are known); the runs of the root folder (tick boxes, a search
    field: key words in the folder name); the variables to save (as File
    > Save variables) and their format; Batch process: for each run
    ticked, the session applied (as Apply session steps to another
    dataset), then the variables selected saved in
    <run>/PostProcessing/Variables_<session name>.<ext> and the session
    saved in the same folder (<session name>.pivsession.json)."""

    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.paths = None                        # the variables to save
        self.queue, self.results, self.running = [], [], False
        self.setWindowTitle("Batch processing")
        self.resize(860, 920)
        lay = QtWidgets.QVBoxLayout(self)
        row, self.root = _path_row("Root folder", "the folder of the runs",
                                   self.browse_root)
        self.root.editingFinished.connect(self.fill_runs)
        lay.addLayout(row)

        box = QtWidgets.QGroupBox("Session processing to apply")
        v = QtWidgets.QVBoxLayout(box)
        row, self.session = _path_row(
            "Session file", "the session whose steps are applied to every "
            "run", self.browse_session)
        v.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.run_session_btn = QtWidgets.QPushButton("Load and run session")
        self.run_session_btn.setToolTip(
            "load the session and run it on its own run (the one in the "
            "session file), to obtain all the variables it makes")
        self.run_session_btn.clicked.connect(self.run_session)
        row.addWidget(self.run_session_btn)
        self.session_state = QtWidgets.QLabel("")
        self.session_state.setWordWrap(True)
        row.addWidget(self.session_state, 1)
        v.addLayout(row)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("Runs")
        v = QtWidgets.QVBoxLayout(box)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Search"))
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText(
            "key words in the folder name (all of them, space separated)")
        self.search.textChanged.connect(self.filter_runs)
        row.addWidget(self.search, 1)
        for text, on in (("All", True), ("None", False)):
            b = QtWidgets.QPushButton(text)
            b.setToolTip("tick / untick the runs shown")
            b.clicked.connect(lambda _c=False, o=on: self.tick_shown(o))
            row.addWidget(b)
        v.addLayout(row)
        self.runs = QtWidgets.QListWidget()
        self.runs.itemChanged.connect(lambda *_: self._count())
        v.addWidget(self.runs, 1)
        self.count = QtWidgets.QLabel("")
        self.count.setStyleSheet("color: gray;")
        v.addWidget(self.count)
        lay.addWidget(box, 1)

        box = QtWidgets.QGroupBox("Variables to save")
        g = QtWidgets.QGridLayout(box)
        self.vars_btn = QtWidgets.QPushButton("Select variables...")
        self.vars_btn.setToolTip(
            "the variables available after the processing (as File > Save "
            "variables): tick the ones to save for each run")
        self.vars_btn.clicked.connect(self.select_variables)
        self.vars_state = QtWidgets.QLabel("none selected yet")
        self.fmt = QtWidgets.QComboBox()
        import gui_variables
        for k, (flt, _ext) in gui_variables.variable_formats().items():
            self.fmt.addItem(flt, k)
        g.addWidget(self.vars_btn, 0, 0)
        g.addWidget(self.vars_state, 0, 1)
        g.addWidget(QtWidgets.QLabel("Format"), 1, 0)
        g.addWidget(self.fmt, 1, 1)
        g.setColumnStretch(1, 1)
        lay.addWidget(box)

        row = QtWidgets.QHBoxLayout()
        self.batch_btn = QtWidgets.QPushButton("Batch process")
        f = self.batch_btn.font(); f.setBold(True); self.batch_btn.setFont(f)
        self.batch_btn.setMinimumHeight(34)
        self.batch_btn.clicked.connect(self.start)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setToolTip("stop after the run being processed")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        row.addWidget(self.batch_btn, 1)
        row.addWidget(self.stop_btn)
        lay.addLayout(row)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.bar.setFormat("%p %")
        self.bar.setToolTip("the progress of the batch: the runs done and "
                            "the steps of the run being processed")
        lay.addWidget(self.bar)
        self.progress = QtWidgets.QLabel("Not started.")
        self.progress.setWordWrap(True)
        self.progress.setStyleSheet("color: gray;")
        lay.addWidget(self.progress)
        self.blog = QtWidgets.QPlainTextEdit()
        self.blog.setReadOnly(True)
        self.blog.setMaximumBlockCount(5000)
        self.blog.setFont(QtGui.QFontDatabase.systemFont(
            E("QFontDatabase.SystemFont.FixedFont")))
        self.blog.setMaximumHeight(120)
        self.blog.setToolTip("the log of the batch (the main window's log "
                             "is not used while the batch runs)")
        lay.addWidget(self.blog)
        self.worker = None
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._update_progress)
        self.report = QtWidgets.QTreeWidget()
        self.report.setHeaderLabels(["run", "result", "files"])
        self.report.setRootIsDecorated(False)
        self.report.setMaximumHeight(170)
        lay.addWidget(self.report)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.close)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close)
        lay.addLayout(row)
        self.root.setText(main.load_panel.root.text())
        self.fill_runs()
        self._state()

    # ---- inputs ----------------------------------------------------------- #
    def browse_root(self):
        p = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Root folder", self.root.text() or self.main.last_dir())
        if p:
            self.root.setText(p)
            self.fill_runs()

    def browse_session(self):
        p, _f = QtWidgets.QFileDialog.getOpenFileName(
            self, "Session file", self.main.last_dir(), SESSION_EXT)
        if p:
            self.session.setText(p)
            self.paths = None
            self.vars_state.setText("none selected yet")
            self.session_state.setText("")
            self._state()

    def fill_runs(self):
        root = self.root.text().strip()
        keep = set(self.ticked_runs(names=True))
        self.runs.blockSignals(True)
        self.runs.clear()
        if os.path.isdir(root):
            for name, _has in lib.list_runs(root):
                it = QtWidgets.QListWidgetItem(name)
                it.setFlags(it.flags() | E("Qt.ItemFlag.ItemIsUserCheckable"))
                it.setCheckState(E("Qt.CheckState.Checked") if name in keep
                                 else E("Qt.CheckState.Unchecked"))
                self.runs.addItem(it)
        self.runs.blockSignals(False)
        self.filter_runs()

    def filter_runs(self, *_):
        words = [w.lower() for w in self.search.text().split()]
        for i in range(self.runs.count()):
            it = self.runs.item(i)
            it.setHidden(not all(w in it.text().lower() for w in words))
        self._count()

    def tick_shown(self, on):
        self.runs.blockSignals(True)
        for i in range(self.runs.count()):
            it = self.runs.item(i)
            if not it.isHidden():
                it.setCheckState(E("Qt.CheckState.Checked") if on
                                 else E("Qt.CheckState.Unchecked"))
        self.runs.blockSignals(False)
        self._count()

    def ticked_runs(self, names=False):
        root = self.root.text().strip()
        out = []
        for i in range(self.runs.count()):
            it = self.runs.item(i)
            if it.checkState() == E("Qt.CheckState.Checked"):
                out.append(it.text() if names else os.path.join(root,
                                                                it.text()))
        return out

    def _count(self):
        shown = sum(1 for i in range(self.runs.count())
                    if not self.runs.item(i).isHidden())
        self.count.setText("%d run(s), %d shown, %d ticked" % (
            self.runs.count(), shown, len(self.ticked_runs(names=True))))
        self._state()

    def _state(self):
        busy = self.running
        sp = os.path.isfile(self.session.text().strip())
        self.run_session_btn.setEnabled(sp and not busy)
        self.vars_btn.setEnabled(self.main.data is not None and not busy)
        self.batch_btn.setEnabled(sp and bool(self.paths) and not busy and
                                  bool(self.ticked_runs(names=True)))
        self.stop_btn.setEnabled(busy)

    # ---- the session on its own run; the variables ----------------------- #
    def run_session(self):
        sp = self.session.text().strip()
        self.session_state.setText("Running the session on its own run ...")
        self.running = True
        self._state()
        self.main.load_session(sp, done=self._session_ran)

    def _session_ran(self, ok, msg):
        self.running = False
        d = self.main.data
        self.session_state.setText(
            ("<b>Ready</b>: %s -- now select the variables." % d.run
             if ok and d is not None else "<b>Failed</b>: %s" % msg))
        self._state()
        self.raise_()

    def select_variables(self):
        import gui_variables
        dlg = gui_variables.SaveVariablesDialog(
            self.main._variables, self, self.main.write_log,
            extra=self.main._variables_to_compute, select_only=True,
            ticked=self.paths)
        if _exec(dlg) and dlg.selected_paths:
            self.paths = dlg.selected_paths
            names = sorted({p[0] for p in self.paths})
            self.vars_state.setText("%d variable(s) (%d entries): %s" % (
                len(names), len(self.paths), ", ".join(names)))
        self._state()

    # ---- the batch (background: a hidden worker window) ------------------ #
    def start(self):
        """Batch process: the runs are processed by a hidden worker window
        -- the main window (its run, panels, figures and log) is not
        changed --, one after the other; this window stays open and shows
        the progress."""
        runs = self.ticked_runs()
        if not runs or not self.paths:
            return
        self.queue = list(runs)
        self.n_runs = len(runs)
        self.results = []
        self.report.clear()
        self.blog.clear()
        self.running, self._stop = True, False
        self.sp = self.session.text().strip()
        self.stem = _session_stem(self.sp)
        self.fmt_key = self.fmt.currentData(E("Qt.ItemDataRole.UserRole"))
        self.worker = MainWindow(worker=True)    # never shown
        orig = self.worker.write_log

        def wlog(text, error=False, stamp=True):
            orig(text, error, stamp)
            line = str(text).rstrip("\n")
            self.blog.appendPlainText(
                (time.strftime("%H:%M:%S  ") if stamp else "") + line)
        self.worker.write_log = wlog
        self.worker.status = lambda msg, ms=8000, log=False: (
            wlog(msg) if log else None)
        self._wlog = wlog
        self.bar.setValue(0)
        self.main.write_log("Batch processing (in the background): session "
                            "%s on %d run(s), variables %s (%s)" % (
                                self.sp, len(runs), ", ".join(sorted(
                                    {p[0] for p in self.paths})),
                                self.fmt_key))
        wlog("Batch: session %s on %d run(s)" % (self.sp, len(runs)))
        self._state()
        self._timer.start()
        QtCore.QTimer.singleShot(0, self._next)

    def stop(self):
        self._stop = True
        self.stop_btn.setEnabled(False)
        self.progress.setText(self.progress.text() + " -- stops after this "
                              "run")

    def _run_fraction(self):
        """Progress within the run being processed (0 .. 1)."""
        w = self.worker
        if w is None or not getattr(self, "current", None):
            return 0.0
        if getattr(self, "_saving", False):
            return 0.95
        r = w._replay
        if r is None:
            return 0.1 if w._thread is not None else 0.0
        if r["stage"] == "restore":
            return 0.1
        n = len(r["done"]) + len(r["queue"])
        return 0.2 + 0.7 * (len(r["done"]) / float(n) if n else 1.0)

    def _update_progress(self):
        if not self.running:
            return
        k = len(self.results)
        frac = (k + min(self._run_fraction(), 0.99)) / max(self.n_runs, 1)
        self.bar.setValue(int(round(1000 * frac)))
        w = self.worker
        what = ""
        if getattr(self, "_saving", False):
            what = "saving the variables"
        elif w is not None and w._replay is not None:
            r = w._replay
            what = ("loading the run" if r["stage"] == "restore" else
                    "step %d / %d" % (min(len(r["done"]) + 1, len(r["done"])
                                          + len(r["queue"])),
                                      len(r["done"]) + len(r["queue"])))
        elif w is not None and w._thread is not None:
            what = "loading the run"
        self.progress.setText("Run %d / %d: %s%s%s" % (
            min(k + 1, self.n_runs), self.n_runs,
            os.path.basename(getattr(self, "current", "") or ""),
            " -- " + what if what else "",
            " (stops after this run)" if self._stop else ""))

    def _next(self):
        if self._stop or not self.queue:
            return self._finish()
        rd = self.queue.pop(0)
        self.current = rd
        self._saving = False
        self._update_progress()
        self.worker.apply_session(self.sp, rd, done=self._run_done)

    def _run_done(self, ok, msg):
        rd, files = self.current, []
        w = self.worker
        if ok and w.data is not None:
            self._saving = True
            self._update_progress()
            try:
                d = w.data
                folder = getattr(d, "post_dir", None) or \
                    lib.postproc_dir(d.run_dir)[0]
                files, missing = w.save_selected_variables(
                    self.paths, os.path.join(folder, "Variables_%s"
                                             % self.stem), self.fmt_key)
                sess = w.save_session(os.path.join(   # in the run folder
                    d.run_dir, "%s.session.json" % d.run))
                files.append(sess)
                msg = "ok" + (" (%d entr(ies) not in this run: %s)" % (
                    len(missing), ", ".join(".".join(map(str, p))
                                            for p in missing[:4]))
                    if missing else "")
            except Exception as exc:
                ok, msg = False, "saving: %s" % exc
            self._saving = False
        self.results.append((rd, ok, msg))
        it = QtWidgets.QTreeWidgetItem([os.path.basename(rd),
                                        msg if ok else "FAILED: " + msg,
                                        ", ".join(os.path.basename(f)
                                                  for f in files)])
        if not ok:
            it.setForeground(1, QtGui.QBrush(QtGui.QColor("#c0392b")))
        self.report.addTopLevelItem(it)
        self.report.scrollToItem(it)
        self._wlog("Batch: %s -- %s%s" % (
            os.path.basename(rd), msg if ok else "FAILED: " + msg,
            (" -> " + ", ".join(files)) if files else ""))
        self._update_progress()
        QtCore.QTimer.singleShot(0, self._next)

    def _finish(self):
        self.running = False
        self._timer.stop()
        n_ok = sum(1 for _r, ok, _m in self.results if ok)
        stopped = self._stop and self.queue
        msg = "Batch processing: %d run(s) done, %d failed%s" % (
            n_ok, len(self.results) - n_ok, " (stopped)" if stopped else "")
        self.bar.setValue(1000 if not stopped else self.bar.value())
        self.progress.setText(msg)
        self._wlog(msg)
        self.main.write_log(msg)
        self.report.addTopLevelItem(QtWidgets.QTreeWidgetItem(["", msg]))
        w, self.worker = self.worker, None
        if w is not None:                        # its memory freed
            w.data = None
            w.close()
            w.deleteLater()
        self._state()

    def closeEvent(self, ev):
        """Closing the window during a batch: it goes on in the
        background (the window can be opened again with File > Batch
        processing)."""
        ev.accept()


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, worker=False):
        super().__init__()
        # worker=True: a hidden window that runs a batch in the background
        # (never shown; the user's window and its log are not touched)
        self.worker = bool(worker)
        self.setWindowTitle(APP_NAME)
        self.data = None
        self.step = None
        self._thread = None
        self._session_dir = None
        self._after_load_step = None
        self.journal = []                        # the steps performed
        self._replay = None                      # a session being replayed
        self._batch = False                      # batch: no questions
        # the time base, kept in memory (not in the parameter file):
        # frames per PIV field, and the pulse separation / camera fps used
        # without a log or when manual (Pre-processing > Calibration)
        self.time_base = lib.full_time_base()
        self.setup_files = {}                    # calibration / ROI / mask files loaded
        self.script_results = {}                 # Python script -> results
        self.extra_panels = {}                   # Advanced items of scripts
        self.script_runs = []                    # the scripts executed
        self._after_load_focus = None

        # actions -- one set, shared by the menu bars of every window
        self.actions = {}
        for _menu, items in MENUS:
            for sid, lab in items:
                a = QtWidgets.QAction(lab, self) if hasattr(
                    QtWidgets, "QAction") else QtGui.QAction(lab, self)
                a.triggered.connect(lambda _c=False, s=sid: self.set_step(s))
                self.actions[sid] = a
        mk = (QtWidgets.QAction if hasattr(QtWidgets, "QAction")
              else QtGui.QAction)
        self.actions["new_session"] = mk("New session", self)
        self.actions["new_session"].triggered.connect(
            lambda *_: self.new_session())
        self.actions["load_session"] = mk("Load session...", self)
        self.actions["save_session"] = mk("Save session...", self)
        self.actions["apply_session"] = mk(
            "Apply session steps to another dataset...", self)
        self.actions["apply_session"].triggered.connect(
            lambda *_: self.show_apply_session())
        self.actions["batch"] = mk("Batch processing...", self)
        self.actions["batch"].triggered.connect(lambda *_: self.show_batch())
        self.actions["save_variables"] = mk("Save variables...", self)
        self.actions["save_variables"].triggered.connect(
            self.show_save_variables)
        self.actions["quit"] = mk("Quit", self)
        self.actions["load_session"].triggered.connect(self.load_session)
        self.actions["save_session"].triggered.connect(self.save_session)
        self.actions["quit"].triggered.connect(self.close)
        self.actions["toggle_log"] = mk("Show / hide the log", self)
        self.actions["toggle_log"].setShortcut("Ctrl+L")
        self.actions["clear_log"] = mk("Clear the log", self)
        self.actions["variables"] = mk("Show all variables", self)
        self.actions["toggle_log"].triggered.connect(self.toggle_log)
        self.actions["clear_log"].triggered.connect(
            lambda *_: self.log.clear())
        self.actions["variables"].triggered.connect(self.show_variables)
        self.actions["load"].setShortcut("Ctrl+O")
        self.actions["save_session"].setShortcut("Ctrl+S")
        self.menus = self._build_menus()
        bar = self.menuBar()
        bar.setStyleSheet(MENU_STYLE)
        for m in self.menus:
            bar.addMenu(m)
        self.add_menu_toolbar(self)

        # left: one panel per step
        self.panels = QtWidgets.QStackedWidget()
        self.panel_of = {}
        self.load_panel = LoadPanel(self)
        self.calib_panel = CalibrationPanel(self)
        self.roi_panel = RoiPanel(self)
        self.mask_panel = MaskPanel(self)
        self.maps_panel = MapsPanel(self)
        self.profiles_panel = ProfilesPanel(self)
        self.timeseries_panel = TimeSeriesPanel(self)
        self.fft_time_panel = FFTTimePanel(self)
        self.fft_space_panel = FFTSpacePanel(self)
        self.script_panel = ScriptPanel(self)
        for _menu, items in MENUS:
            for sid, lab in items:
                w = (self.load_panel if sid == "load" else
                     self.calib_panel if sid == "calibration" else
                     self.roi_panel if sid == "roi" else
                     self.mask_panel if sid == "mask" else
                     self.maps_panel if sid == "maps" else
                     self.profiles_panel if sid == "profiles" else
                     self.timeseries_panel if sid == "timeseries" else
                     self.fft_time_panel if sid == "fft_time" else
                     self.fft_space_panel if sid == "fft_space" else
                     self.script_panel if sid == "python_script" else
                     PlaceholderPanel(lab))
                self.panel_of[sid] = w
                self.panels.addWidget(w)
        self.panel_title = QtWidgets.QLabel()
        f = self.panel_title.font(); f.setBold(True); f.setPointSize(
            f.pointSize() + 2); self.panel_title.setFont(f)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panels)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.addWidget(self.panel_title)
        ll.addWidget(scroll, 1)

        # centre + right
        self.figures = FigureManager(self)
        split = QtWidgets.QSplitter(E("Qt.Orientation.Horizontal"))
        split.addWidget(left)
        split.addWidget(self.figures.center)
        split.addWidget(self.figures.tree)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([430, 880, 220])
        # the log below (as in the DTS GUI): View > Show / hide the log
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QtGui.QFontDatabase.systemFont(
            E("QFontDatabase.SystemFont.FixedFont")))
        self.log.setMaximumBlockCount(20000)
        self.log.setMinimumHeight(40)
        self.log.setLineWrapMode(E("QPlainTextEdit.LineWrapMode.WidgetWidth"))
        vert = QtWidgets.QSplitter(E("Qt.Orientation.Vertical"))
        vert.addWidget(split)
        vert.addWidget(self.log)
        vert.setStretchFactor(0, 1)
        vert.setStretchFactor(1, 0)
        vert.setSizes([780, 140])
        self.vsplit = vert
        self.setCentralWidget(vert)
        self._start_log()
        self.statusBar()
        self.resize(1550, 950)
        self.set_step("load")
        if self.worker:
            self._batch = True                   # no questions
            self._replaced = True                # streams left alone
        else:
            self._restore_settings()

    # ----------------------------------------------------------------- #
    def _build_menus(self):
        """The drop-down menus, built once and shared by the menu bar and
        by the menu toolbars of every window."""
        menus = []
        for menu, items in MENUS:
            m = QtWidgets.QMenu(menu, self)
            m.setStyleSheet(MENU_STYLE)
            for sid, _lab in items:
                m.addAction(self.actions[sid])
            if menu == "File":
                m.addSeparator()
                m.addAction(self.actions["new_session"])
                m.addAction(self.actions["load_session"])
                m.addAction(self.actions["save_session"])
                m.addAction(self.actions["apply_session"])
                m.addAction(self.actions["batch"])
                m.addSeparator()
                m.addAction(self.actions["save_variables"])
                m.addSeparator()
                m.addAction(self.actions["quit"])
            menus.append(m)
        m = QtWidgets.QMenu("View", self)
        m.setStyleSheet(MENU_STYLE)
        for key in ("toggle_log", "clear_log", "variables"):
            m.addAction(self.actions[key])
        menus.append(m)
        return menus

    def add_menu_toolbar(self, window):
        """The drop-down menus repeated at the top of `window` (on macOS the
        menu bar is at the top of the screen) -- as in the DTS GUI."""
        tb = QtWidgets.QToolBar("Menus", window)
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.toggleViewAction().setEnabled(False)
        for m in self.menus:
            btn = QtWidgets.QToolButton(tb)
            btn.setText(m.title() + "  \u25be")
            btn.setMenu(m)
            btn.setPopupMode(E("QToolButton.ToolButtonPopupMode.InstantPopup"))
            btn.setToolButtonStyle(E("Qt.ToolButtonStyle.ToolButtonTextOnly"))
            btn.setStyleSheet("QToolButton { padding: 4px 12px; "
                              "font-size: 16px; font-weight: bold; } "
                              "QToolButton::menu-indicator { image: none; }")
            tb.addWidget(btn)
        window.addToolBar(E("Qt.ToolBarArea.TopToolBarArea"), tb)
        return tb

    def status(self, msg, ms=8000, log=False):
        """A message in the status bar (and in the log with log=True)."""
        self.statusBar().showMessage(msg, ms)
        if log:
            self.write_log(msg)

    # ---- log ------------------------------------------------------------ #
    def write_log(self, text, error=False, stamp=True):
        """Add a line (or lines) to the log panel, with the time."""
        text = str(text).rstrip("\n")
        if stamp:
            pad = " " * 10
            text = time.strftime("%H:%M:%S  ") + text.replace("\n",
                                                               "\n" + pad)
        self._log_append(text + "\n")
        if error and not self.log.isVisible():
            self.log.setVisible(True)

    def _log_append(self, text):
        cur = self.log.textCursor()
        cur.movePosition(E("QTextCursor.MoveOperation.End"))
        cur.insertText(text)
        self.log.setTextCursor(cur)
        self.log.ensureCursorVisible()

    def toggle_log(self):
        self.log.setVisible(not self.log.isVisible())

    def _start_log(self):
        """First lines of the log; printed text and uncaught errors are
        sent to the log from now on."""
        import matplotlib as _mpl
        from matplotlib.backends.qt_compat import QT_API
        self.write_log("%s GUI. Python %s (%s)" % (
            APP_NAME, sys.version.split()[0], sys.executable))
        self.write_log("Qt binding %s, numpy %s, matplotlib %s" % (
            QT_API, np.__version__, _mpl.__version__))
        for module in ("scipy", "h5py"):
            try:
                __import__(module)
            except ImportError:
                self.write_log("WARNING: the module '%s' is missing in this "
                               "Python: install it with  pip install %s"
                               % (module, module))
        self._streams = []
        if self.worker:                          # printed text: not here
            return
        for name in ("stdout", "stderr"):
            st = LogStream(getattr(sys, "__" + name + "__", None))
            st.text.connect(self._log_append)
            setattr(sys, name, st)
            self._streams.append(st)
        sys.excepthook = self._excepthook

    def _excepthook(self, etype, value, tb):
        """An uncaught error (e.g. in a button's slot): written in the log
        instead of being lost (or of closing the GUI)."""
        msg = "".join(traceback.format_exception(etype, value, tb))
        if sys.__stderr__ is not None:           # the terminal too
            sys.__stderr__.write(msg)
        try:
            self.write_log("ERROR (uncaught):\n" + msg.rstrip(), error=True)
        except Exception:                        # window already closed
            return
        self.status("Error: %s (details in the log)" % value)

    # ---- variables -------------------------------------------------------- #
    def _variables(self):
        """({name: value}, {name: description}, note, steps) for View > Show
        all variables and File > Save variables: the loaded run, the
        parameter file selected, the calibration in progress and the
        settings and results of every step; steps = [(step, [names])] (the
        groups shown, in the menu order)."""
        import collections
        out = collections.OrderedDict()
        help_ = dict(lib.VARIABLE_HELP)
        if self.data is not None:
            out.update(lib.run_variables(self.data))
            lp = self.load_panel
            out["setup_files_loaded"] = {
                k: (row.active if row.check.isChecked() else None)
                for k, row in lp.setup_rows()}
        C = self.calib_panel
        cal = {"image": C.image.currentData(C._role),
               "segment_px": (np.asarray(C.editor.points, float)
                              if C.editor.points else None),
               "length_px": lib.segment_length(C.editor.points or []),
               "length_mm": C.mm.value() or None,
               "scale_m_per_px (Scale)": C._new_scale(),
               "applied_m_per_px": C.scale}
        out["calibration_segment"] = cal
        self._step_variables(out, help_)
        note = ("" if self.data is not None else
                "No run loaded (File > Load data). ") + \
            "Select a row: its full description below. Right click: show " \
            "the value."
        return out, help_, note, self.VARIABLE_STEPS

    # the variables by step (Show / Save variables), in the menu order
    VARIABLE_STEPS = (
        ("File > Load data", (
            "run_name", "run_folder", "pivlab_result_file",
            "acquisition_log_file", "setup_files_loaded", "parameters",
            "velocity_type_loaded", "velocity_types_in_file", "x_grid_px",
            "z_grid_px", "u_velocity_px_per_frame",
            "v_velocity_px_per_frame", "rejected_vectors",
            "number_of_fields", "field_sources", "valid_vector_fraction",
            "mat_file_variables", "acquisition_log_last_row")),
        ("Pre-processing > Calibration", (
            "calibration_scales", "calibration_segment", "time_base",
            "velocity_calibrated_m_per_s")),
        ("Pre-processing > Define ROI", (
            "roi_and_line_coordinates", "roi_velocity_windows",
            "roi_derived_quantities", "line_derived_quantities")),
        ("Pre-processing > Mask", (
            "mask_applied", "mask_applied_name", "mask_applied_areas",
            "masks_defined")),
        ("Processing > Maps and Quiver", (
            "maps_display_options", "maps_figure_data", "maps_movie_info")),
        ("Processing > Time series", (
            "time_series_options", "time_series_data")),
        ("Processing > Profiles", (
            "profiles_options", "profiles_data", "profiles_movie_info")),
        ("Spectral analysis > FFT in time", (
            "fft_time_options", "fft_time_frequency_ranges",
            "fft_time_spectra", "fft_time_peaks",
            "fft_time_spectrum_per_point", "fft_time_integrated_map")),
        ("Spectral analysis > FFT in space", (
            "fft_space_options", "fft_space_spectrum", "fft_space_peaks",
            "fft_space_kt_diagram")),
        ("Advanced > Python script", (
            "script_results", "script_runs")),
    )

    def _step_variables(self, out, help_):
        """The settings and results of the steps (Define ROI, Maps and
        Quiver, Time series, Profiles, FFT in time) for the variables
        lists; a result is listed once its step was run (its figure made)."""
        d = self.data
        F = self.figures

        def add(name, fn, text):
            try:
                value = fn()
            except Exception as exc:             # never break the list
                value = "not available (%s)" % exc
            if value is None:
                return
            out[name] = value
            help_[name] = text

        def settings(panel):
            st = panel.get_state()
            return {k: v for k, v in st.items() if v is not None}

        def movie(box):
            m = getattr(box, "movie", None)
            if not m:
                return None
            return {k: v for k, v in m.items()
                    if isinstance(v, (str, int, float, bool))}
        if d is None:
            return
        R, T, P, X = (self.roi_panel, self.timeseries_panel,
                      self.profiles_panel, self.fft_time_panel)
        add("roi_and_line_coordinates", R._data,
            "Pre-processing > Define ROI: the corners of each ROI "
            "(ROI_ROIxx: [[x1, z1], [x2, z2]]) and the end points of each "
            "line (LINE_xx) as drawn in the lists, in m -- ticked or not "
            "(the same as PTS_ROI and LINES of the parameter file after "
            "Apply).")
        # Processing > Maps and Quiver
        if F.find(MapsPanel.STEP, MapsPanel.FIG) is not None:
            add("maps_display_options", self.load_panel.display_options,
                "Processing > Maps and Quiver: the display options in use -- "
                "velocity calibrated or not, the ROI field shown, the frame "
                "(index, mean or rms), the quiver settings (spacing, scale), "
                "the background map and its colour map, smoothing and "
                "shading.")
            add("maps_figure_data", self._display_data,
                "Processing > Maps and Quiver: the data of the figure shown "
                "-- grid x, z, velocity u, v and its norm of the frame and "
                "field displayed (with the units, the time base and the "
                "mask), and the background quantity when there is one.")
        add("maps_movie_info", lambda: movie(self.maps_panel.mv),
            "Processing > Maps and Quiver > Make Movie: the movie made -- "
            "its file, first and last frame, frames per second.")
        # Processing > Time series
        add("time_series_options", lambda: settings(T),
            "Processing > Time series: the statistic chosen (e.g. <U>, "
            "U_std, <|Vorticity|>) and which active ROIs are ticked.")
        if F.find(T.STEP, T.FIG) is not None:
            add("time_series_data", T._data,
                "Processing > Time series: for each ROI ticked, the time of "
                "each PIV field (t_s, s) and the statistic over the ROI at "
                "that time (values, unit), with the number of points; the "
                "definition of the statistic.")
        # Processing > Profiles
        add("profiles_options", lambda: settings(P),
            "Processing > Profiles: the line(s), the ROI, the extracted "
            "quantity and the frame (index, mean or rms) chosen.")
        if F.find(P.STEP, P.FIG) is not None:
            add("profiles_data", P._data,
                "Processing > Profiles: for each line, the points of the "
                "line inside the ROI (x, z in m), their coordinate along "
                "the line (s, m) and the extracted quantity there (values), "
                "with the units and the frame.")
        add("profiles_movie_info", lambda: movie(P.mv),
            "Processing > Profiles > Make Movie: the movie made -- its "
            "file, first and last frame, frames per second.")
        # Spectral analysis > FFT in time
        add("fft_time_options", lambda: settings(X),
            "Spectral analysis > FFT in time: every setting -- ROI, "
            "quantity, spectral function (FFT, power spectrum, "
            "periodogram), remove mean, detrend, segment length, order(s) "
            "of the calculation, display (frequency / period, fmin, fmax, "
            "log scale), number of peaks, map display.")
        add("fft_time_frequency_ranges", lambda: {
            "W%d" % (k + 1): {"fmin_Hz": w["fmin"], "fmax_Hz": w["fmax"],
                              "ticked": bool(w.get("on"))}
            for k, w in enumerate(X.windows)} or None,
            "Spectral analysis > FFT in time: the frequency ranges of the "
            "list (W1, W2, ...): lower and upper frequency in Hz, and "
            "which one is ticked (used by the peaks and the FFT map).")
        if self.script_results:
            add("script_results", lambda: {k: dict(v) for k, v in
                                           self.script_results.items()},
                "Advanced > Python script: what each script executed put in "
                "its `results` (one entry per script, named by its file), "
                "kept in memory and saved with the session")
        if self.script_runs:
            add("script_runs", lambda: {r["script"]: dict(r)
                                        for r in self.script_runs},
                "Advanced > Python script: the scripts executed -- file, "
                "time, duration, success or error, the results and "
                "figures they made")
        S = self.fft_space_panel
        add("fft_space_options", lambda: settings(S),
            "Spectral analysis > FFT in space: every setting -- region, "
            "direction (x, z or a line), quantity, spectral function, "
            "window, remove mean, detrend, time (one frame or the mean), "
            "display (x axis, kmin, kmax, log), peaks, k-t diagram")
        if S.res is not None:
            add("fft_space_spectrum", lambda: lib.fft_space_data(
                d, S.res, S.peaks),
                "Spectral analysis > FFT in space > Apply: the spatial "
                "spectrum -- f (cycles per m), k (rad/m), wavelength (m), "
                "amplitude (FFT; with phase, real and imaginary parts) or "
                "power spectral density, at the frame chosen or averaged "
                "over the frames; and kt_amplitude / kt_psd: the spectrum "
                "at every frame (nt x nk) with the times t")
        if S.peaks:
            add("fft_space_peaks", lambda: {
                "rank": np.array([p["rank"] for p in S.peaks]),
                "k_rad_per_m": np.array([2 * np.pi * p["f"]
                                         for p in S.peaks]),
                "wavelength": np.array([p["period"] for p in S.peaks]),
                "value": np.array([p["value"] for p in S.peaks])},
                "Spectral analysis > FFT in space > Find peaks: the n "
                "dominant peaks (rank, wavenumber k in rad/m, wavelength, "
                "amplitude / density)")
        if S.kt_res is not None:
            add("fft_space_kt_diagram", lambda: lib.fft_space_data(
                d, S.kt_res),
                "Spectral analysis > FFT in space > Plot k-t diagram: the "
                "spatial spectrum (k_x, k_z or k along a line) at every "
                "frame -- kt_amplitude / kt_psd (nt x nk), f, k, "
                "wavelength, t")
        res = getattr(X, "results", {}) or {}
        if res:
            add("fft_time_spectra", lambda: lib.fft_data(
                d, list(res.values()), X.windows, X.peaks),
                "Spectral analysis > FFT in time > Apply: one entry per "
                "order of the calculation -- 'average' (the quantity "
                "averaged over the ROI, then its spectrum) and 'fft' (the "
                "spectrum at each point, then averaged over the ROI). Each "
                "holds the frequencies f (Hz), the amplitude (FFT) or the "
                "power spectral density psd (power spectrum, periodogram), "
                "the phase, real and imaginary parts (FFT), the unit, the "
                "ROI, the number of points and the peaks found; plus the "
                "ranges and the frequencies used.")
            add("fft_time_peaks", lambda: {
                o: {"rank": np.array([p["rank"] for p in pk]),
                    "f": np.array([p["f"] for p in pk]),
                    "period": np.array([p["period"] for p in pk]),
                    "value": np.array([p["value"] for p in pk])}
                for o, pk in (X.peaks or {}).items() if pk} or None,
                "Spectral analysis > FFT in time > Find peaks: the n "
                "dominant peaks of each order -- rank (1 = highest), "
                "frequency f (Hz), period (s) and amplitude / density "
                "value; within the range ticked, if any.")
            r = res.get("fft")
            if r is not None and r.get("amp_points") is not None:
                add("fft_time_spectrum_per_point", lambda: {
                    "x": r.get("x"), "z": r.get("z"), "f": r["f"],
                    "amplitude": r["amp_points"], "unit": r.get("unit"),
                    "roi": r.get("roi"), "quantity": r.get("name")},
                    "Spectral analysis > FFT in time (order 'spectral "
                    "analysis first'): the amplitude (or density) of the "
                    "spectrum at each grid point of the ROI and each "
                    "frequency (ny x nx x nf), with the grid x, z (m) and "
                    "the frequencies f (Hz) -- the data of the FFT maps.")
                if X._ticked_range():
                    add("fft_time_integrated_map",
                        lambda: lib.fft_map_data(d, r, X.windows),
                        "Spectral analysis > FFT in time > Plot FFT maps: "
                        "at each grid point of the ROI, the amplitude (or "
                        "density) integrated over the frequency range "
                        "ticked (ny x nx), with the grid x, z (m) and the "
                        "frequencies used.")

    def show_variables(self):
        """View > Show all variables (as in the DTS GUI)."""
        import gui_variables
        dlg = getattr(self, "_variables_dialog", None)
        if dlg is None:
            dlg = gui_variables.VariablesDialog(self._variables, self)
            self._variables_dialog = dlg         # keep it alive
        dlg.fill()
        dlg.setModal(False)
        dlg.show()
        dlg.raise_()

    def _variables_to_compute(self):
        """For File > Save variables: the calibrated velocity (m, m/s, s;
        mask applied) of the full field and of each ROI field, computed only
        when saved (only the uncalibrated data are kept in memory)."""
        import gui_variables as gv
        d = self.data
        if d is None or not d.can_calibrate:
            return {}

        def cal(window=None):
            vw = lib.RunView(d, True, window)
            u, v = vw.uv()
            return {"x_m": np.asarray(vw.x, float),
                    "z_m": np.asarray(vw.z, float),
                    "u_m_per_s": np.asarray(u, np.float32),
                    "v_m_per_s": np.asarray(v, np.float32),
                    "t_s": np.asarray(vw.t, float),
                    "mask_applied": d.mask_name or "none"}
        ny, nx = d.X_px.shape
        out = {"Pre-processing > Calibration": {
            "velocity_calibrated_m_per_s": (
                gv.Lazy(cal, size="%d x %d x %d" % (ny, nx, d.nt)),
                "The %s velocity of the full field converted to physical "
                "units, computed only when saved (not kept in memory): grid "
                "x_m, z_m (m), velocity u_m_per_s, v_m_per_s (m/s, ny x nx "
                "x nt) and the time of each field t_s (s); XSCALE / YSCALE, "
                "pulse separation and time base applied, NaN in the mask "
                "applied." % d.default)}}
        rois = {}
        for name, info in d.roi_fields().items():
            rois["velocity_calibrated_m_per_s_%s" % lib.roi_name(
                info["number"])] = (
                gv.Lazy(lambda w=info["window"]: cal(w),
                        size="%d x %d x %d" % (info["shape"][0],
                                               info["shape"][1], d.nt)),
                "%s (field %s): the velocity inside the ROI converted to "
                "physical units, computed only when saved -- x_m, z_m (m), "
                "u_m_per_s, v_m_per_s (m/s, rows x columns of the ROI x "
                "nt), t_s (s); mask applied." % (info["roi"], name))
        if rois:
            out["Pre-processing > Define ROI"] = rois
        return out

    def show_save_variables(self):
        """File > Save variables (as Advanced > Save variables of the DTS
        GUI): tick the variables, then Save (file, folder, format)."""
        import gui_variables

        def folder():
            if FigureManager.last_save_folder:
                return FigureManager.last_save_folder
            if self.data is not None:
                return (getattr(self.data, "post_dir", None)
                        or lib.postproc_dir(self.data.run_dir)[0])
            return self.last_dir()

        def stem():
            return ("%s_variables" % self.data.run if self.data is not None
                    else "PIV_variables")

        def log(msg):
            self.write_log(msg)
            dlg = self._save_variables_dialog
            if getattr(dlg, "saved_folder", None):
                FigureManager.last_save_folder = dlg.saved_folder
        dlg = getattr(self, "_save_variables_dialog", None)
        if dlg is None:
            dlg = gui_variables.SaveVariablesDialog(
                self._variables, self, log, extra=self._variables_to_compute,
                folder=folder, stem=stem)
            self._save_variables_dialog = dlg
        dlg.fill()
        dlg.setModal(False)
        dlg.show()
        dlg.raise_()

    def show_apply_session(self):
        dlg = getattr(self, "_apply_dialog", None)
        if dlg is None:
            dlg = self._apply_dialog = ApplySessionDialog(self)
        dlg.setModal(False)
        dlg.show()
        dlg.raise_()
        return dlg

    def show_batch(self):
        dlg = getattr(self, "_batch_dialog", None)
        if dlg is None:
            dlg = self._batch_dialog = BatchDialog(self)
        dlg.setModal(False)
        dlg.show()
        dlg.raise_()
        return dlg

    def setup_folder(self):
        """The default folder of the calibration / ROI / mask files: the
        root folder of Load data (else the folder of the run, the last
        folder used)."""
        lp = self.load_panel
        if lp.root_dir:
            return lp.root_dir
        if self.data is not None:
            return os.path.dirname(self.data.run_dir)
        return self.last_dir()

    def save_selected_variables(self, paths, path, fmt="npz"):
        import gui_variables
        if fmt not in gui_variables.VARIABLE_FORMATS:  # no CSV
            fmt = "npz"
        return self._save_selected_variables(paths, path, fmt)

    def _save_selected_variables(self, paths, path, fmt="npz"):
        """The variables at the paths (as ticked in File > Save variables)
        written to path (its extension added); returns (files, the paths
        this run does not have)."""
        import gui_variables
        import gui_export
        sel, missing = gui_variables.select_values(
            self._variables, self._variables_to_compute, paths)
        path = gui_export.with_extension(path, fmt)
        files = gui_export.write(path, fmt, gui_export.to_tree(sel),
                                 title="PIV GUI variables: "
                                       + ", ".join(sel))
        self.write_log("%d variable(s) saved (%s): %s" % (
            len(sel), fmt, ", ".join(files)))
        return list(files), missing

    def _refresh_save_variables(self):
        dlg = getattr(self, "_save_variables_dialog", None)
        if dlg is not None and dlg.isVisible():
            dlg.fill()                           # new variables at once

    def last_dir(self):
        return (self.load_panel.root_dir or self._session_dir
                or os.path.expanduser("~"))

    def add_advanced_item(self, sid, label, panel):
        """A new item of the Advanced menu (made by a Python script): its
        panel on the left, its figures under `label` in the figure list.
        The panel may have open() (the item is opened) and data_loaded(d)
        (a run is loaded). Executing the script again replaces the item."""
        STEP_LABEL[sid] = label
        old = self.panel_of.get(sid)
        if old is not None:                      # replaced
            self.panels.removeWidget(old)
            old.deleteLater()
        self.panel_of[sid] = panel
        self.extra_panels[sid] = panel
        self.panels.addWidget(panel)
        if sid not in self.actions:
            mk = (QtWidgets.QAction if hasattr(QtWidgets, "QAction")
                  else QtGui.QAction)
            a = mk(label, self)
            a.triggered.connect(lambda _c=False, s=sid: self.set_step(s))
            self.actions[sid] = a
            for m in self.menus:
                if m.title() == "Advanced":
                    m.addAction(a)
        else:
            self.actions[sid].setText(label)
        self.write_log("Advanced menu: item '%s' added" % label)
        return panel

    def set_step(self, sid, focus_figures=True):
        """Open the parameter panel of a step and bring its figures."""
        self.step = sid
        if sid == "calibration":
            self.calib_panel.open()
        elif sid == "roi":
            self.roi_panel.open()
        elif sid == "mask":
            self.mask_panel.open()
        elif sid == "maps":
            self.maps_panel.open()
        elif sid == "profiles":
            self.profiles_panel.open()
        elif sid == "timeseries":
            self.timeseries_panel.open()
        elif sid == "fft_time":
            self.fft_time_panel.open()
        elif sid == "fft_space":
            self.fft_space_panel.open()
        elif sid == "python_script":
            self.script_panel.open()
        elif sid in self.extra_panels:           # added by a script
            p = self.panel_of[sid]
            if hasattr(p, "open"):
                p.open()
        self.panels.setCurrentWidget(self.panel_of[sid])
        self.panel_title.setText(STEP_LABEL[sid])
        if focus_figures:
            self.figures.focus_step(sid)
        self.raise_()

    # ----------------------------------------------------------------- #
    #  Load data
    # ----------------------------------------------------------------- #
    def _resolve_files(self, run_dir, params):
        """The PIV .mat and the acquisition log of a run, before reading:
        the files named in the parameter file in the run folder; else the
        ones recorded in its run_information.json; else asked to the user
        (select the .mat; select a log or skip it and enter the pulse
        separation and the camera fps). Returns dict(piv, log, tb) -- log
        False when skipped, tb the time base entered -- or None (cancel)."""
        rec = lib.read_run_info(run_dir)
        files, tb_rec = rec.get("files") or {}, rec.get("time_base") or {}
        run = os.path.basename(run_dir.rstrip("/"))
        # ---- the PIVlab .mat
        piv = os.path.join(run_dir, params["PIV_FILENAME"])
        if not os.path.isfile(piv):
            old = files.get("piv_file")
            if old and os.path.isfile(old):
                piv = old
                self.write_log("PIV file of %s: %s (from %s)"
                               % (run, old, lib.RUN_INFO_FILE))
            elif self._batch:                    # no question in a batch
                self.write_log("No %s in %s: run skipped"
                               % (params["PIV_FILENAME"], run))
                return None
            else:
                QtWidgets.QMessageBox.information(
                    self, "Load data",
                    "No %s in %s.\n\nSelect the PIVlab .mat file of this "
                    "run." % (params["PIV_FILENAME"], run))
                path, _f = QtWidgets.QFileDialog.getOpenFileName(
                    self, "PIVlab .mat file of %s" % run, run_dir,
                    "MATLAB files (*.mat);;All files (*)")
                if not path:
                    self.status("Load cancelled (no .mat file)", log=True)
                    return None
                piv = path
                self.write_log("PIV file of %s selected: %s" % (run, path))
        # ---- the acquisition log
        log, tb = os.path.join(run_dir, params["LOG_FILENAME"]), None
        if os.path.isfile(log):
            return dict(piv=piv, log=log, tb=None)
        old = files.get("log_file")
        same_run = self.data is not None and self.data.run_dir == run_dir
        if old and os.path.isfile(old):
            self.write_log("Log of %s: %s (from %s)"
                           % (run, old, lib.RUN_INFO_FILE))
            return dict(piv=piv, log=old, tb=None)
        if files.get("log_skipped"):
            if same_run:                         # the values in memory
                return dict(piv=piv, log=False, tb=None)
            if tb_rec.get("pulse_separation_s") and \
                    tb_rec.get("camera_fps_Hz"):
                tb = dict(pulse_separation=tb_rec["pulse_separation_s"],
                          camera_fps=tb_rec["camera_fps_Hz"])
                self.write_log("No log for %s: pulse separation %.6g s and "
                               "camera %.6g Hz entered before (%s)"
                               % (run, tb["pulse_separation"],
                                  tb["camera_fps"], lib.RUN_INFO_FILE))
                return dict(piv=piv, log=False, tb=tb)
        if self._batch:                          # no question in a batch
            self.write_log("No log for %s: the time base of the session is "
                           "used (pulse separation %.6g s, camera %.6g Hz)"
                           % (run, self.time_base["pulse_separation"],
                              self.time_base["camera_fps"]))
            return dict(piv=piv, log=False, tb=None)
        while True:                              # ask: select / skip
            box = QtWidgets.QMessageBox(self)
            box.setIcon(E("QMessageBox.Icon.Warning"))
            box.setWindowTitle("No acquisition log")
            box.setText("<b>%s</b>: no acquisition log (%s)."
                        % (run, params["LOG_FILENAME"]))
            box.setInformativeText(
                "Select the log file of this run, or skip it and enter the "
                "pulse separation and the camera fps.")
            sel = box.addButton("Select a log file...",
                                E("QMessageBox.ButtonRole.AcceptRole"))
            skip = box.addButton("Skip", E("QMessageBox.ButtonRole.NoRole"))
            box.addButton(E("QMessageBox.StandardButton.Cancel"))
            _exec(box)
            if box.clickedButton() is sel:
                path, _f = QtWidgets.QFileDialog.getOpenFileName(
                    self, "Acquisition log of %s" % run, run_dir,
                    "Log files (*.txt *.log *.tsv *.csv);;All files (*)")
                if path:
                    self.write_log("Log of %s selected: %s" % (run, path))
                    return dict(piv=piv, log=path, tb=None)
            elif box.clickedButton() is skip:
                dlg = TimeBaseDialog(self, run,
                                     self.time_base["pulse_separation"],
                                     self.time_base["camera_fps"])
                if _exec(dlg) and dlg.values:
                    tb = dict(pulse_separation=dlg.values[0],
                              camera_fps=dlg.values[1])
                    self.write_log("No log for %s (skipped): pulse "
                                   "separation %.6g s, camera %.6g Hz "
                                   "entered" % (run, dlg.values[0],
                                                dlg.values[1]))
                    return dict(piv=piv, log=False, tb=tb)
            else:
                self.status("Load cancelled (no log)", log=True)
                return None

    def load_run(self, run_dir, params, param_file=None, calibrate=True,
                 velocity=None):
        if self._thread is not None:
            self.status("Already loading...")
            return
        files = self._resolve_files(run_dir, params)
        if files is None:
            return
        if files["tb"]:                          # entered by hand (no log)
            self.time_base = lib.full_time_base(dict(self.time_base,
                                                     **files["tb"]))
        self._param_file = param_file
        self.status("Loading %s ..." % os.path.basename(run_dir), 0)
        QtWidgets.QApplication.setOverrideCursor(
            E("Qt.CursorShape.WaitCursor"))
        self.load_panel.load_btn.setEnabled(False)
        if velocity is None:
            velocity = self.load_panel.velocity.currentText()
        self.write_log("Loading %s  (parameters %s, calibration %s, velocity "
                       "%s)" % (run_dir, os.path.basename(param_file or "--"),
                                "on" if calibrate else "off", velocity))
        self._thread = LoadThread(run_dir, params, calibrate, velocity,
                                  dict(self.time_base), files["piv"],
                                  files["log"])
        self._thread.done.connect(self._loaded)
        self._thread.failed.connect(self._load_failed)
        self._thread.finished.connect(self._load_finished)
        self._thread.start()

    def _load_finished(self):
        QtWidgets.QApplication.restoreOverrideCursor()
        self._thread = None
        self.load_panel.load_btn.setEnabled(bool(self.load_panel.run_dir))

    def _load_failed(self, msg):
        self.status("Loading failed")
        self.write_log("ERROR while loading:\n%s" % msg.rstrip(), error=True)
        r, self._replay = self._replay, None
        if r is not None and r.get("callback"):  # a session: tell it
            QtCore.QTimer.singleShot(0, lambda: r["callback"](
                False, "loading failed: %s" % msg.strip().splitlines()[-1]))
        if not self._batch:
            QtWidgets.QMessageBox.critical(self, "Load data", msg[:3000])

    def _loaded(self, d):
        self.data = d
        try:                                     # <run>/PostProcessing
            d.post_dir, created = lib.postproc_dir(d.run_dir)
            if created:
                self.write_log("Created %s" % d.post_dir)
        except Exception as exc:
            self.write_log("WARNING: could not create %s in %s: %s"
                           % (lib.POSTPROC_DIR, d.run_dir, exc))
        self.setWindowTitle("%s -- %s" % (APP_NAME, d.run))
        self.load_panel.data_loaded(d)
        pf = getattr(self.load_panel, "_pending_frame", None)
        if pf is not None:
            self.load_panel.frame.setValue(min(int(pf), d.nt - 1))
            self.load_panel._pending_frame = None
        info = self.show_info(d)
        self.draw_field()
        self.calib_panel.refresh()
        self.roi_panel.data_loaded(d)
        self.mask_panel.data_loaded(d)
        self.maps_panel.data_loaded(d)
        self.profiles_panel.data_loaded(d)
        self.profiles_panel.apply_pending()
        self.timeseries_panel.data_loaded(d)
        self.timeseries_panel.apply_pending()
        self.fft_time_panel.data_loaded(d)
        self.fft_time_panel.apply_pending()
        self.fft_space_panel.data_loaded(d)
        self.fft_space_panel.apply_pending()
        if getattr(self, "_script_run_dir", None) != d.run_dir:
            self.script_results, self.script_runs = {}, []   # another run
        self._script_run_dir = d.run_dir
        self.script_panel.data_loaded(d)
        for sid, p in list(self.extra_panels.items()):   # script items
            if hasattr(p, "data_loaded"):
                try:
                    p.data_loaded(d)
                except Exception as exc:
                    self.write_log("WARNING: %s: %s" % (sid, exc))
        self._log_loaded(d)
        self._apply_setup_files(d)
        after, self._after_load_step = self._after_load_step, None
        if after and "roi_apply" in self.journal and d.can_calibrate and \
                (d.params.get("PTS_ROI") or d.params.get("LINES")):
            # the run read again by a step (calibration, time base): the
            # ROI quantities of the active ROIs calculated again
            self.roi_panel.calculate_quantities()
        focus_page, self._after_load_focus = self._after_load_focus, None
        if after:                                 # reloaded by a step: stay
            self.set_step(after)
        if focus_page:
            page = self.figures.find(*focus_page)
            if page is not None:
                self.figures.focus(page)
        elif not after:
            self.figures.focus(info)
        self.status("Loaded %s: %d fields, dt = %.4g s, %s, PIV fields "
                    "%.4g Hz" % (d.run, d.nt, d.acq.dt,
                                 "camera %.4g Hz" % d.acq.cam_fps
                                 if d.acq.ok else "no log file",
                                 d.acq.fps))
        if not d.acq.log_ok and d.acq.source == "memory (no log)" and \
                not after and self._replay is None:   # a log not read
            QtCore.QTimer.singleShot(0, lambda: self.no_log_prompt(d))
        if self._replay is not None:             # a session: go on
            QtCore.QTimer.singleShot(0, self._replay_continue)

    def setup_file_used(self, kind, path, show=True):
        """A calibration / ROI / mask file saved or loaded (the step run
        from it): kept for the run information (table and
        run_information.json, updated and saved now) and the session."""
        self.setup_files[kind] = path
        if show and self.data is not None:
            try:
                self.show_info(self.data)
            except Exception as exc:
                self.write_log("WARNING: run information not updated: %s"
                               % exc)

    def _apply_setup_files(self, d):
        """After a Load data with calibration / ROI / mask files: each step
        run from its file (the calibration, the ROIs and the masks are
        already in the run's parameters)."""
        pending, self._pending_setup = getattr(
            self, "_pending_setup", None) or {}, None
        if not pending:
            return
        for kind, panel in (("calibration", self.calib_panel),
                            ("roi", self.roi_panel),
                            ("mask", self.mask_panel)):
            if kind not in pending:
                continue
            path, rec = pending[kind]
            try:
                if kind == "roi":
                    panel.load_rois(path)
                else:
                    panel.after_load(path, rec)
            except Exception as exc:
                self.write_log("WARNING: %s step from %s: %s"
                               % (kind, path, exc), error=True)
        self.show_info(d)

    def show_info(self, d, save=True):
        """The Run information table of a loaded run, saved in
        <run>/run_information.json (called again whenever something it
        shows changes)."""
        self._refresh_save_variables()
        sections = d.info_sections()
        pf = None
        files = getattr(self, "setup_files", {}) or {}
        for k, (kind, label) in enumerate((("calibration", "calibration file"),
                                           ("roi", "ROI file"),
                                           ("mask", "mask file"))):
            sections[0][1].insert(1 + k, (label, files.get(kind) or "--"))
        if self.script_runs:                     # Advanced > Python script
            sections.append(("Advanced: Python scripts", [
                (r["script"], "%s %s (%.2f s) -- %s%s%s" % (
                    "executed" if r["ok"] else "FAILED", r["time"],
                    r["seconds"], r["path"],
                    "; results: " + ", ".join(r["results"])
                    if r["results"] else "; no result",
                    "; error: " + r["error"] if r.get("error") else ""))
                for r in self.script_runs]))
            try:
                path = lib.write_run_info(d, sections, pf,
                                          setup_files=self.setup_files)
                self.write_log("Run information saved: %s" % path)
            except Exception as exc:
                self.write_log("WARNING: run information not saved in %s: "
                               "%s" % (d.run_dir, exc))
        return self.figures.add_table(
            "load", "Run information", sections,
            data_fn=lambda: lib.sections_data(
                self.figures.find("load", "Run information").sections),
            update_fn=self.update_info)

    def _log_loaded(self, d):
        a = d.acq
        self.write_log(
            "Loaded %s: %d fields, grid %d x %d (z x x); velocity %s only "
            "(%.1f %% valid, %.1f MB); %s; time base from %s: pulse "
            "separation %.6g s, camera %.6g Hz, TSCALE %.6g s"
            % (d.run, d.nt, d.ny, d.nx, d.default, 100 * d.valid,
               (d.U_px.nbytes + d.V_px.nbytes) / 1e6,
               ("calibration XSCALE %s m/px, shown %s"
                % (d.params["XSCALE"],
                   CALIBRATED if d.calibrated else UNCALIBRATED))
               if d.can_calibrate else "no calibration (px, px/frame)",
               a.source, a.dt, a.cam_fps, a.tscale))
        if not a.log_ok:
            self.write_log("WARNING: %s" % a.message)
        if d.velocity_asked not in (None, d.default):
            self.write_log("WARNING: %s is not in the .mat, %s loaded"
                           % (d.velocity_asked, d.default))

    def no_log_prompt(self, d):
        """The run has no (readable) acquisition log: say so, and offer the
        Calibration panel, where the time base can be typed."""
        box = QtWidgets.QMessageBox(self)
        box.setIcon(E("QMessageBox.Icon.Warning"))
        box.setWindowTitle("No acquisition log")
        box.setText("<b>%s</b>: %s." % (d.run, d.acq.message))
        box.setInformativeText(
            "The time between frames and the sampling frequency of the PIV "
            "fields are unknown; the time base in memory is used for now "
            "(pulse separation %.6g s, camera %.6g Hz).\n\n"
            "Enter the real values in Pre-processing > Calibration."
            % (d.acq.dt, d.acq.cam_fps))
        go = box.addButton("Go to Calibration",
                           E("QMessageBox.ButtonRole.AcceptRole"))
        box.addButton("Later", E("QMessageBox.ButtonRole.RejectRole"))
        _qexec(box)()
        if box.clickedButton() is go:
            self.set_step("calibration")

    def update_params(self, path, values):
        """Update of the Parameter file table: write the values, select
        that file in Load data, load the run again (the table stays in
        front)."""
        lib.write_param_file(path, values)
        lp = self.load_panel
        lp.refresh(select_param=os.path.basename(path))
        self.status("Saved %s -- loading the run again" % path, log=True)
        self._after_load_focus = ("load", "Parameter file")
        if self.data is not None and lp.run_dir:
            lp.load()

    def _display_data(self):
        o = self.load_panel.display_options()
        return lib.display_data(self.data, o["frame"], o["field"], o["roi"])

    def update_info(self):
        """Update of the run information: the run read again with the
        parameters in memory."""
        d = self.data
        if d is None:
            self.load_panel.load()
            return
        self.load_run(d.run_dir, dict(d.params), None,
                      calibrate=d.calibration_asked, velocity=d.default)

    def draw_field(self, force=False):
        """The figure of Processing > Maps and Quiver, drawn with the
        options of its panel; created when the step is opened (force), then
        updated by every change."""
        if self.data is None:
            return
        opts = self.load_panel.display_options()
        opts.pop("mask", None)                   # applied to the run
        page = self.figures.find(MapsPanel.STEP, MapsPanel.FIG)
        if page is None and not force and self.step != MapsPanel.STEP:
            page = False
        try:
            if page is False:
                pass                             # not shown yet
            elif page is None:
                fig = lib.display_figure(self.data,
                                         fig=lib.new_figure(), **opts)
                page = self.figures.add(
                    MapsPanel.STEP, MapsPanel.FIG, fig,
                    data_fn=self._display_data,
                    update_fn=self.draw_field)
            else:
                lib.display_figure(self.data, fig=page.figure, **opts)
            if page:
                page.redraw()
        except Exception as exc:
            self.status("Figure error: %s" % exc, log=True)
        # Define ROI / Mask on the current frame follow the frame of Load
        for rp in (self.roi_panel, self.mask_panel):
            if rp.source.currentData(rp._role) == "frame" and \
                    self.figures.find(rp.STEP, rp.FIG) is not None:
                rp.draw(keep_view=True)

    # ----------------------------------------------------------------- #
    #  Sessions
    # ----------------------------------------------------------------- #
    # ---- sessions ---------------------------------------------------------- #
    STEP_ACTIONS = {
        "calibration": "Calibration > Apply (scale in memory)",
        "roi_apply": "Define ROI > Apply (ROIs, lines, ROI quantities)",
        "fft_apply": "FFT in time > Apply",
        "fft_peaks": "FFT in time > Find peaks",
        "fft_map": "FFT in time > Plot FFT maps",
        "fftspace_apply": "FFT in space > Apply",
        "fftspace_peaks": "FFT in space > Find peaks",
        "fftspace_kt": "FFT in space > Plot k-t diagram",
    }

    def mark_done(self, action):
        """A step performed (kept in order, the last time it was done):
        Save session records it, Load session performs it again."""
        if action in self.journal:
            self.journal.remove(action)
        self.journal.append(action)

    def _widget_panels(self):
        """(key, object, exclude) of everything whose widgets a session
        keeps (the Load data files are kept by its own state)."""
        return [
            ("load", self.load_panel, ("root", "run_combo", "velocity")),
            ("maps_movie", self.maps_panel.mv, ()),
            ("roi", self.roi_panel, ()),
            ("roi_style", self.roi_panel.style, ()),
            ("mask", self.mask_panel, ()),
            ("mask_style", self.mask_panel.style, ()),
            ("profiles", self.profiles_panel, ()),
            ("profiles_movie", self.profiles_panel.mv, ()),
            ("timeseries", self.timeseries_panel, ()),
            ("fft_time", self.fft_time_panel, ()),
            ("fft_space", self.fft_space_panel, ()),
            ("fft_map_style", self.fft_time_panel.map_style, ()),
        ]

    def session_state(self):
        """Everything needed to come back to this point: the files, the
        state of every panel and widget, the steps performed (in order)
        and the figures shown."""
        widgets = {k: widget_states(o, ex)
                   for k, o, ex in self._widget_panels()}
        figures = [[p.step, p.name] for p in self.figures.pages
                   if self.figures.is_ticked(p)]
        return dict(version=2, step=self.step,
                    time_base=dict(self.time_base),
                    load=self.load_panel.get_state(),
                    calibration=self.calib_panel.get_state(),
                    roi=self.roi_panel.get_state(),
                    mask=self.mask_panel.get_state(),
                    profiles=self.profiles_panel.get_state(),
                    timeseries=self.timeseries_panel.get_state(),
                    fft_time=self.fft_time_panel.get_state(),
                    fft_space=self.fft_space_panel.get_state(),
                    scripts=dict(panel=self.script_panel.get_state(),
                                 runs=[dict(r) for r in self.script_runs]),
                    advanced_items={sid: widget_states(p, ())
                                    for sid, p in self.extra_panels.items()},
                    widgets=widgets,
                    executed=list(self.journal),
                    figures=figures,
                    params=(lib._jsonable(dict(self.data.params))
                            if self.data is not None else None),
                    setup_files=dict(self.setup_files),
                    loaded_run=(self.data.run_dir if self.data else None))

    def save_session(self, path=None):
        if not path:
            path, _f = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save session", os.path.join(self.last_dir(),
                                                   "session.pivsession.json"),
                SESSION_EXT)
        if not path:
            return None
        st = self.session_state()
        if self.script_results:                  # the data of the scripts
            npz = os.path.splitext(path)[0] + ".scripts.npz"
            try:
                flat = {}
                for name, res in self.script_results.items():
                    for k, v in res.items():
                        flat["%s/%s" % (name, k)] = (
                            np.asarray(v) if not isinstance(v, dict)
                            else np.array(v, dtype=object))
                np.savez_compressed(npz, **flat)
                st["scripts"]["results_file"] = npz
                self.write_log("Script results saved with the session: %s"
                               % npz)
            except Exception as exc:
                self.write_log("WARNING: script results not saved: %s" % exc)
        with open(path, "w") as fh:
            json.dump(lib._jsonable(st), fh, indent=2)
        self._session_dir = os.path.dirname(path)
        self.status("Session saved: %s (%d step(s) performed: %s; %d "
                    "figure(s))" % (path, len(st["executed"]), ", ".join(
                        st["executed"]) or "none", len(st["figures"])),
                    log=True)
        return path

    def load_session(self, path=None, done=None):
        """Load session: the files and panels as saved, the run loaded
        again, then every widget set back, the steps performed run again in
        their order (a step that loads the run again waits for it) and the
        figures shown again."""
        if not path:
            path, _f = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load session", self.last_dir(), SESSION_EXT)
        if not path:
            return
        try:
            with open(path) as fh:
                s = json.load(fh)
        except Exception as exc:
            if done:
                done(False, str(exc))
            else:
                QtWidgets.QMessageBox.warning(self, "Load session", str(exc))
            return
        self._start_session(s, path, done=done)

    def apply_session(self, path, run_dir, done=None):
        """File > Apply session steps to another dataset (and Batch
        processing): the session file read, its paths changed to the run
        folder run_dir (root = its parent folder), the run loaded with the
        SAME parameter file as the session, then every widget set back and
        the steps performed again, as by Load session. The run information
        (table and run_information.json) is that of the new run. done(ok,
        message) is called at the end. Returns False when it cannot start."""
        try:
            with open(path) as fh:
                s = json.load(fh)
        except Exception as exc:
            if done:
                done(False, "session file: %s" % exc)
            return False
        run_dir = os.path.normpath(run_dir)
        if not os.path.isdir(run_dir):
            if done:
                done(False, "not a folder: %s" % run_dir)
            return False
        self.write_log("Apply session %s to %s" % (path, run_dir))
        return self._start_session(s, path, run_dir=run_dir, done=done)

    def _session_param_file(self, s):
        """The parameter file of a session (full path)."""
        pf = s.get("param_file")
        if pf and os.path.isfile(pf):
            return pf
        ld = s.get("load", {})
        if ld.get("root") and ld.get("params"):
            pf = os.path.join(ld["root"], ld["params"])
            if os.path.isfile(pf):
                return pf
        return None

    @staticmethod
    def _session_setup_files(s):
        """The calibration / ROI / mask files the steps of a session were
        run from (older sessions: the files ticked in its Load data)."""
        files = dict(s.get("setup_files") or {})
        if not files:
            for k, st in ((s.get("load") or {}).get("files") or {}).items():
                if st and st.get("on") and st.get("path"):
                    files[k] = st["path"]
        return {k: v for k, v in files.items() if v}

    def _start_session(self, s, path, run_dir=None, done=None):
        """Load session / Apply session: the panels as saved, the run
        loaded (run_dir: another run, with the session's parameter file),
        then _replay_continue. Returns False when it cannot start."""
        def fail(msg):
            self._replay = None
            self.write_log("Session: %s" % msg, error=True)
            if done:
                done(False, msg)
            elif not self._batch:
                QtWidgets.QMessageBox.warning(self, "Session", msg)
            return False
        if self._thread is not None:
            return fail("a run is being loaded: wait for it")
        try:
            P = s.get("params")                  # the parameters in memory
            if not P:                            # older session: its file
                pf = self._session_param_file(s)
                P = lib.read_param_file(pf) if pf else None
            if P:
                P = dict(lib.default_params(), **P)
            if run_dir is not None:              # another dataset
                s = json.loads(json.dumps(s))
                s.setdefault("load", {})
                s["load"]["root"] = os.path.dirname(run_dir)
                s["load"]["run"] = os.path.basename(run_dir)
                s["loaded_run"] = run_dir
            if s.get("time_base"):
                self.time_base = lib.full_time_base(s["time_base"])
            self.load_panel.set_state(s.get("load", {}))
            self.calib_panel.set_state(s.get("calibration", {}))
            self.roi_panel.set_state(s.get("roi", {}))
            self.mask_panel.set_state(s.get("mask", {}))
            self.profiles_panel.set_state(s.get("profiles", {}))
            self.timeseries_panel.set_state(s.get("timeseries", {}))
            self.fft_time_panel.set_state(s.get("fft_time", {}))
            self.fft_space_panel.set_state(s.get("fft_space", {}))
            self.script_panel.set_state((s.get("scripts") or {}).get(
                "panel") or {})
            self.script_results, self.script_runs = {}, []   # from the
            self._script_run_dir = None          # session: executed again
            self._session_dir = os.path.dirname(path)
            self.journal = []
            self.write_log("Session %s: %d step(s) to perform again: %s" % (
                path, len(s.get("executed") or []),
                ", ".join(s.get("executed") or []) or "none"))
            rd = run_dir or (s.get("loaded_run") and self.load_panel.run_dir)
            if not rd:
                self.set_step(s.get("step") or "load")
                self.status("Session loaded: %s (no run)" % path, log=True)
                if done:
                    done(True, "no run in the session")
                return True
            self._replay = dict(s=s, path=path, stage="restore",
                                queue=list(s.get("executed") or []),
                                done=[], callback=done)
            self.setup_files = self._session_setup_files(s)
            if not P and run_dir is None:
                self.load_panel.load()           # then _replay_continue
            else:                                # the session's parameters
                ld = s.get("load", {})
                field = lib.OLD_FIELD_NAMES.get(ld.get("field"),
                                                ld.get("field"))
                self._pending_setup = None
                P = P or lib.default_params()
                self.load_run(rd, P, None, calibrate=bool(ld.get(
                    "calibrate", True)) and lib.is_calibrated(P),
                    velocity=field or None)
            if self._thread is None and self._replay is not None and \
                    self._replay["stage"] == "restore":
                return fail("the run %s could not be loaded" % rd)
            return True
        except Exception as exc:
            return fail(str(exc))

    def _replay_continue(self):
        """After a Load of the session: the widgets back, then the steps
        performed, one after the other (a step that loads the run again
        -- Calibration > Apply -- resumes here when the run is loaded)."""
        r = self._replay
        if r is None or self.data is None:
            return
        s = r["s"]
        if r["stage"] == "restore":
            r["stage"] = "steps"
            roi = s.get("roi", {})
            if not roi.get("dirty"):             # the lists as applied
                self.roi_panel.restore_lists(roi.get("lists"), False)
            self.mask_panel.restore_shown(s.get("mask", {}).get(
                "masks_shown"))
            n = 0
            for key, obj, ex in self._widget_panels():
                n += restore_widget_states(obj, (s.get("widgets") or {}).get(
                    key), ex)
            self.write_log("Session: %d widget(s) set back" % n)
        while r["queue"]:
            action = r["queue"].pop(0)
            self.write_log("Session: performing again %s" % (
                self.STEP_ACTIONS.get(action, action)))
            try:
                reload = self._perform(action)
            except Exception as exc:
                self.write_log("Session: %s failed: %s" % (action, exc),
                               error=True)
                reload = False
            r["done"].append(action)
            if reload:
                return                           # resumed by _loaded
        self._replay = None
        self._restore_script_results(s, r)
        for sid, st in (s.get("advanced_items") or {}).items():
            p = self.extra_panels.get(sid)       # made again by its script
            if p is not None:
                try:
                    restore_widget_states(p, st, ())
                    if hasattr(p, "open") and self.step == sid:
                        p.open()
                except Exception as exc:
                    self.write_log("Session: %s settings: %s" % (sid, exc))
        roi = s.get("roi", {})
        if roi.get("dirty"):                     # edits not applied: after
            self.roi_panel.restore_lists(roi.get("lists"), True)
        steps = []
        for st, _name in s.get("figures") or []:
            if st not in steps and st in self.panel_of:
                steps.append(st)
        for st in steps:                         # the figures shown again
            try:
                self.set_step(st)
            except Exception as exc:
                self.write_log("Session: figures of %s: %s" % (st, exc))
        self.set_step(s.get("step") or "load")
        # the run information of the run, updated and saved at the end (with
        # the calibration / ROI / mask files the steps were run from)
        try:
            self.show_info(self.data)
        except Exception as exc:
            self.write_log("WARNING: run information not updated: %s" % exc)
        msg = ("Session restored: %s on %s -- %d step(s) performed again "
               "(%s), %d step figure(s)" % (
                   r["path"], self.data.run, len(r["done"]),
                   ", ".join(r["done"]) or "none", len(steps)))
        self.status(msg, log=True)
        if r.get("callback"):
            r["callback"](True, msg)

    def _restore_script_results(self, s, r):
        """After a session replay: the results of the scripts that were not
        executed again (file missing) taken from the session's
        .scripts.npz -- for the same run only."""
        sc = s.get("scripts") or {}
        npz = sc.get("results_file")
        same = os.path.normpath(s.get("loaded_run") or "") == \
            os.path.normpath(self.data.run_dir)
        if not npz or not os.path.isfile(npz) or not same:
            return
        try:
            z = np.load(npz, allow_pickle=True)
        except Exception as exc:
            self.write_log("WARNING: %s not read: %s" % (npz, exc))
            return
        n = 0
        done = set(self.script_results)          # executed again
        for key in z.files:
            name, _s, k = key.partition("/")
            if name in done:
                continue
            v = z[key]
            self.script_results.setdefault(name, {})[k] = (
                v.item() if v.dtype == object and v.shape == () else v)
            n += 1
        for run in sc.get("runs") or []:
            if run["script"] not in {x["script"] for x in self.script_runs} \
                    and run["script"] in self.script_results:
                self.script_runs.append(dict(run, restored=npz))
        if n:
            self.write_log("Session: %d script result(s) restored from %s"
                           % (n, npz))
            self.script_panel._fill_results()
            self.show_info(self.data)

    def _perform(self, action):
        """One step of a session again; True when it loads the run again
        (the replay goes on when it is loaded)."""
        if action == "calibration":
            if self.calib_panel._new_scale() is None:
                self.write_log("Session: no calibration segment / length: "
                               "Calibration > Apply skipped")
                return False
            self.calib_panel.apply(save=False)
            return self._thread is not None
        if action == "roi_apply":
            self.roi_panel.update_param_file()
        elif action == "fft_apply":
            self.fft_time_panel.apply()
        elif action == "fft_peaks":
            self.fft_time_panel.find_peaks()
        elif action == "fft_map":
            self.fft_time_panel.plot_map()
        elif action.startswith("script:"):
            path = action[len("script:"):]
            if os.path.isfile(path):
                self.script_panel.execute(path)
            else:
                self.write_log("Session: script %s not found (its saved "
                               "results are used)" % path)
        elif action == "fftspace_apply":
            self.fft_space_panel.apply()
        elif action == "fftspace_peaks":
            self.fft_space_panel.find_peaks()
        elif action == "fftspace_kt":
            self.fft_space_panel.plot_kt()
        else:
            self.mark_done(action)
        return False

    # ----------------------------------------------------------------- #
    def _restore_settings(self):
        st = QtCore.QSettings("PIVlab", "PIV_GUI")
        root = st.value("root", "")
        if root:
            self.load_panel.root.setText(root)
            self.load_panel.refresh()

    def new_session(self, ask=True):
        """File > New session: everything in memory removed -- the run and
        its variables, the figures (and their windows), the steps
        performed, the settings of every step, the log -- and a fresh
        window started (the root folder kept). Files on disk are not
        changed. Asks first: Save session first / New session / Cancel.
        Returns the new window (None when cancelled)."""
        if self._thread is not None:
            QtWidgets.QMessageBox.information(
                self, "New session", "A run is being loaded: wait for the "
                "end of the Load first.")
            return None
        if ask and (self.data is not None or self.figures.pages):
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("New session")
            box.setText("Start a new session?")
            box.setInformativeText(
                "All the variables in memory, the figures and the steps "
                "performed are removed, and every step starts afresh. Files "
                "on disk (data, parameter file, run information, saved "
                "figures, movies and sessions) are not changed.")
            save = box.addButton("Save session first...",
                                 E("QMessageBox.ButtonRole.AcceptRole"))
            new = box.addButton("New session",
                                E("QMessageBox.ButtonRole.DestructiveRole"))
            box.addButton(E("QMessageBox.StandardButton.Cancel"))
            box.setDefaultButton(new)
            _qexec(box)()
            clicked = box.clickedButton()
            if clicked is save:
                if not self.save_session():
                    return None                  # the save was cancelled
            elif clicked is not new:
                return None
        root = self.load_panel.root.text()
        geom = self.saveGeometry()
        # this window: everything dropped (memory freed), then closed
        self._replaced = True
        self.maps_panel.close()                  # unsaved movies removed
        self.profiles_panel.close()
        self.figures.close_all_windows()
        for page in list(self.figures.pages):
            try:
                self.figures.remove(page)
            except Exception:
                pass
        self.data = None
        self.journal = []
        self._replay = None
        dlg = getattr(self, "_save_variables_dialog", None)
        if dlg is not None:
            dlg.close()
        nw = MainWindow()                        # the fresh session
        nw.restoreGeometry(geom)
        if root:
            nw.load_panel.root.setText(root)
            nw.load_panel.refresh()
        _WINDOWS.append(nw)
        nw.show()
        nw.write_log("New session (the previous one closed: variables, "
                     "figures and steps removed; files not changed)")
        nw.status("New session")
        self.setAttribute(E("Qt.WidgetAttribute.WA_DeleteOnClose"))
        self.close()
        if self in _WINDOWS:
            _WINDOWS.remove(self)
        import gc
        gc.collect()
        return nw

    def closeEvent(self, ev):
        if not self.worker and not getattr(self, "_replaced", False) and \
                not getattr(self, "_quit_asked", False) and \
                (self.data is not None or self.figures.pages):
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Quit")
            box.setText("Save the current session before quitting?")
            box.setInformativeText(
                "The variables in memory, the figures and the steps "
                "performed are lost when the GUI is closed. Files on disk "
                "(data, run information, calibration / ROI / mask files, "
                "saved figures, movies and sessions) are not changed.")
            save = box.addButton("Save session first...",
                                 E("QMessageBox.ButtonRole.AcceptRole"))
            quit_ = box.addButton("Quit without saving",
                                  E("QMessageBox.ButtonRole.DestructiveRole"))
            box.addButton(E("QMessageBox.StandardButton.Cancel"))
            box.setDefaultButton(save)
            _qexec(box)()
            clicked = box.clickedButton()
            if clicked is save:
                if not self.save_session():      # the save was cancelled
                    ev.ignore()
                    return
            elif clicked is not quit_:
                ev.ignore()                      # Cancel: stay
                return
            self._quit_asked = True
            bd = getattr(self, "_batch_dialog", None)
            if bd is not None and bd.running:    # the batch stops too
                bd._stop = True
        if not self.worker:
            QtCore.QSettings("PIVlab", "PIV_GUI").setValue(
                "root", self.load_panel.root.text())
        self.maps_panel.close()
        self.profiles_panel.close()
        if not getattr(self, "_replaced", False):  # not by a New session
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
            sys.excepthook = sys.__excepthook__
        self.figures.close_all_windows()
        ev.accept()


_WINDOWS = []                    # the main window(s) alive (New session)


def main(argv):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    app.setApplicationName(APP_NAME)
    w = MainWindow()
    _WINDOWS.append(w)
    w.show()
    if len(argv) > 1 and argv[1].endswith(".json"):
        w.load_session(argv[1])
    return _qexec(app)()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
