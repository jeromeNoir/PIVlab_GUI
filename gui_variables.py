"""View > Show all variables: the variables of the loaded run kept in memory
by the GUI, with a short description, their format and their size. Right
click on a variable: Show value.

Qt is taken from matplotlib's qt_compat, so the module works with PySide6
and PyQt5 like PIV_GUI.py.
"""
import numpy as np
from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets



def _qexec(obj):
    """obj.exec (PySide6, PyQt6, recent PyQt5), else obj.exec_ (old PyQt5) --
    exec_ is deprecated in PySide6 (DeprecationWarning)."""
    f = getattr(obj, "exec", None)
    return f if callable(f) else obj.exec_

def _E(path):
    """A Qt enum value, scoped (Qt6) or unscoped (Qt5)."""
    parts = path.split(".")
    obj = getattr(QtCore, parts[0], None) or getattr(QtWidgets, parts[0],
                                                     None) \
        or getattr(QtGui, parts[0])
    try:
        for p in parts[1:]:
            obj = getattr(obj, p)
        return obj
    except AttributeError:
        obj = getattr(QtCore, parts[0], None) or getattr(QtWidgets,
                                                         parts[0], None) \
            or getattr(QtGui, parts[0])
        return getattr(obj, parts[-1])


def human_bytes(n):
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%.0f %s" % (n, unit) if unit == "B" \
                else "%.1f %s" % (n, unit)
        n /= 1024.0


def memory(value, depth=0, seen=None):
    """Approximate memory used by a value [bytes] (an array counted once
    even when it appears twice, e.g. U and VELOCITY_FIELDS)."""
    seen = set() if seen is None else seen
    if isinstance(value, np.ndarray):
        base = value.base if value.base is not None else value
        if id(base) in seen:
            return 0
        seen.add(id(base))
        return value.nbytes
    if depth > 6:
        return 0
    if isinstance(value, dict):
        return sum(memory(v, depth + 1, seen) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(memory(v, depth + 1, seen) for v in value)
    if isinstance(value, str):
        return len(value)
    return 8


def describe(value):
    """(format, size) of a value."""
    if isinstance(value, np.ndarray):
        return ("array %s" % value.dtype,
                " x ".join(str(n) for n in value.shape) or "scalar")
    if isinstance(value, dict):
        return "dict", "%d entries" % len(value)
    if isinstance(value, (list, tuple)):
        return type(value).__name__, "%d items" % len(value)
    if value is None:
        return "None", ""
    if isinstance(value, (bool, np.bool_)):
        return "bool", str(bool(value))
    if isinstance(value, str):
        return "text", value if len(value) <= 60 else value[:57] + "..."
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return type(value).__name__, "%.6g" % float(value)
        except (TypeError, ValueError):
            return type(value).__name__, str(value)
    return type(value).__name__, ""


def value_text(value, depth=0):
    """Readable text of a value: arrays with their statistics and
    (shortened) content, dicts and lists entry by entry."""
    pad = "    " * depth
    if isinstance(value, np.ndarray):
        lines = ["array %s, shape %s" % (value.dtype, value.shape)]
        if value.size and value.dtype.kind in "fiu":
            finite = value[np.isfinite(value)] if value.dtype.kind == "f" \
                else value
            n_nan = value.size - finite.size
            if finite.size:
                lines.append("min %.6g, max %.6g, mean %.6g, NaN %d"
                             % (finite.min(), finite.max(), finite.mean(),
                                n_nan))
            else:
                lines.append("all NaN (%d)" % n_nan)
        elif value.size and value.dtype.kind == "b":
            lines.append("True: %d of %d" % (int(value.sum()), value.size))
        with np.printoptions(threshold=200, edgeitems=5, linewidth=100,
                             precision=5):
            lines.append(np.array2string(value))
        return "\n".join(pad + ln for ln in "\n".join(lines).split("\n"))
    if isinstance(value, dict):
        if depth > 3:
            return pad + "dict with %d entries" % len(value)
        out = []
        for key, child in value.items():
            text = value_text(child, depth + 1)
            if "\n" in text or isinstance(child, (dict, np.ndarray, list,
                                                  tuple)):
                out.append("%s%s:\n%s" % (pad, key, text))
            else:
                out.append("%s%s: %s" % (pad, key, text.strip()))
        return "\n".join(out) if out else pad + "{} (empty)"
    if isinstance(value, (list, tuple)):
        if depth > 3 or len(value) > 50:
            head = ", ".join(str(v) for v in list(value)[:20])
            return pad + "%s of %d: %s%s" % (type(value).__name__,
                                             len(value), head,
                                             " ..." if len(value) > 20
                                             else "")
        if not any(isinstance(v, (dict, np.ndarray, list, tuple))
                   for v in value):
            return pad + repr(list(value) if isinstance(value, list)
                              else tuple(value))
        return "\n".join("%s[%d]:\n%s" % (pad, i, value_text(v, depth + 1))
                         for i, v in enumerate(value))
    return pad + repr(value)


def _entry_help():
    try:
        from piv_gui_lib import ENTRY_HELP
        return dict(ENTRY_HELP)
    except Exception:                            # another GUI
        return {}


ENTRY_HELP = _entry_help()


# the formats of Save variables: those of gui_export without CSV
VARIABLE_FORMATS = ("npz", "netcdf", "hdf5", "txt", "mat")


def variable_formats():
    import gui_export
    return {k: v for k, v in gui_export.FORMATS.items()
            if k in VARIABLE_FORMATS}


def variable_filter():
    return ";;".join(f for f, _ext in variable_formats().values())


def variable_format(path, chosen_filter=""):
    """The format of Save variables (from the filter, else the extension;
    CSV is not offered: npz then)."""
    import gui_export
    fmt = gui_export.format_of(path, chosen_filter)
    return fmt if fmt in VARIABLE_FORMATS else "npz"


def short_text(description):
    """The description for the column: without its leading 'menu > step
    ...:' (the step is the group already; the full text is in the details
    and the tool tip)."""
    head, sep, rest = description.partition(": ")
    if sep and " > " in head and len(head) < 70:
        return rest[:1].upper() + rest[1:]
    return description


def entry_description(key):
    """Description of an entry inside a variable (by its name): the
    ENTRY_HELP of the GUI, else from patterns ('<quantity>_ROIxx', W1)."""
    import re
    key = str(key)
    if key in ENTRY_HELP:
        return ENTRY_HELP[key]
    m = re.match(r"^(.*)_(ROI\d+)$", key)
    if m:
        q = m.group(1).replace("_", " ")
        if q in ("Validated", "Unvalidated", "Smoothed"):
            return "%s velocity field of %s" % (q, m.group(2))
        return "%s calculated in %s" % (q, m.group(2))
    m = re.match(r"^W(\d+)$", key)
    if m:
        return "frequency range %s" % m.group(1)
    m = re.match(r"^\[(\d+)\]$", key)
    if m:
        return "item %s of the list" % m.group(1)
    return ""


class _StepTree(object):
    """Shared by Show variables and Save variables: the search field (all
    the keywords in the name, the parents, the step or the description) and
    the details box of the row selected."""

    def _make_search(self, lay):
        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(QtWidgets.QLabel("Search"))
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText(
            "keywords (e.g. fft amplitude, ROI01, vorticity): the variables "
            "and entries whose name, step or description hold them all")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(lambda *_: self.apply_filter())
        srow.addWidget(self.search, 1)
        self.found = QtWidgets.QLabel("")
        self.found.setStyleSheet("color: gray;")
        srow.addWidget(self.found)
        lay.addLayout(srow)

    def _make_details(self, lay):
        self.details = QtWidgets.QTextBrowser()
        self.details.setMaximumHeight(130)
        self.details.setPlaceholderText(
            "Select a row: its full description here.")
        lay.addWidget(self.details)
        self.tree.currentItemChanged.connect(
            lambda cur, _prev: self.show_details(cur))
        self.tree.setWordWrap(True)

    def show_details(self, item):
        if item is None:
            self.details.clear()
            return
        path, it = [], item
        while it is not None:
            path.insert(0, it.text(0).split("  (")[0])
            it = it.parent()
        desc = item.toolTip(1) or item.text(1)
        parent = item.parent()
        while not desc and parent is not None:   # an entry: its variable
            desc = parent.toolTip(1)
            parent = parent.parent()
        from html import escape
        html = "<b>%s</b>" % " &gt; ".join(escape(p) for p in path)
        info = [x for x in (item.text(2), item.text(3) and "size " +
                            item.text(3), item.text(4) and "memory " +
                            item.text(4)) if x]
        if info:
            html += " &nbsp; <span style='color: gray;'>%s</span>" % \
                escape(", ".join(info))
        own = item.toolTip(1) or item.text(1)
        html += "<p>%s</p>" % escape(own or "")
        if not own and desc:
            html += "<p style='color: gray;'>In: %s</p>" % escape(desc)
        self.details.setHtml(html)

    def apply_filter(self, *_):
        """Show the entries whose name, step (category), parent names or
        description hold every keyword of the search field (case ignored);
        their parents are shown and expanded, and a match shows all its
        entries."""
        words = self.search.text().lower().split()

        def show(item):
            item.setHidden(False)
            for i in range(item.childCount()):
                show(item.child(i))

        def visit(item, context):
            text = context + " " + item.text(0).lower() + " " + \
                item.text(1).lower() + " " + item.toolTip(1).lower()
            if all(w in text for w in words):
                show(item)
                return True
            hit = False
            for i in range(item.childCount()):
                hit = visit(item.child(i), text) or hit
            item.setHidden(not hit)
            if hit and words:
                item.setExpanded(True)
            return hit
        n = 0
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            if not words:
                show(top)
            elif visit(top, ""):
                n += 1
        self.found.setText("%d step(s) match" % n if words else "")

    def _shown(self, item):
        while item is not None:
            if item.isHidden():
                return False
            item = item.parent()
        return True

    def _step_item(self, category, variables, size):
        top = QtWidgets.QTreeWidgetItem(self.tree, [
            "%s  (%d)" % (category, len(variables)), "", "", "",
            human_bytes(size) if size else ""])
        f = top.font(0)
        f.setBold(True)
        top.setFont(0, f)
        top.setToolTip(1, "the variables of the step %s" % category)
        return top


class VariablesDialog(QtWidgets.QDialog, _StepTree):
    """Tree of the variables in memory, grouped by step (as Save
    variables): dicts open to their entries; a search field; the full
    description of the row selected below.

    source()  -> ({name: value}, {name: description}, footnote[, steps])
    """

    COLUMNS = ("Variable", "Description", "Format", "Size", "Memory")

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self.source = source
        self.values = {}                     # id(item) -> (name, value)
        self.setWindowTitle("All variables in memory")
        self.resize(1050, 650)
        lay = QtWidgets.QVBoxLayout(self)
        self._make_search(lay)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(list(self.COLUMNS))
        self.tree.setAlternatingRowColors(True)
        self.tree.setContextMenuPolicy(
            _E("Qt.ContextMenuPolicy.CustomContextMenu"))
        self.tree.customContextMenuRequested.connect(self.context_menu)
        lay.addWidget(self.tree, 1)
        self._make_details(lay)
        self.total = QtWidgets.QLabel("")
        self.total.setWordWrap(True)
        self.total.setStyleSheet("color: gray;")
        lay.addWidget(self.total)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.accept)
        row = QtWidgets.QHBoxLayout()
        for text, slot in (("Expand all", lambda: self.tree.expandAll()),
                           ("Collapse", lambda: self.tree.collapseAll()),
                           ("Refresh", self.fill)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(close)
        lay.addLayout(row)
        self.fill()

    def _item(self, parent, name, value, description, level):
        fmt, size = describe(value)
        description = description or entry_description(name)
        item = QtWidgets.QTreeWidgetItem(
            parent, [name, short_text(description), fmt, size,
                     human_bytes(memory(value))])
        item.setToolTip(1, description)
        self.values[id(item)] = (name, value)
        if level < 3 and isinstance(value, dict):
            for key, child in value.items():
                self._item(item, str(key), child, "", level + 1)
        elif (level < 3 and isinstance(value, (list, tuple))
              and 0 < len(value) <= 12
              and any(isinstance(v, (dict, np.ndarray, list, tuple))
                      for v in value)):
            for i, child in enumerate(value):
                self._item(item, "[%d]" % i, child, "", level + 1)
        return item

    def fill(self):
        expanded = {self.tree.topLevelItem(i).text(0).split("  (")[0]
                    for i in range(self.tree.topLevelItemCount())
                    if self.tree.topLevelItem(i).isExpanded()}
        self.tree.clear()
        self.values = {}
        entries, desc = all_entries(self.source)
        note = self.source()[2]
        seen = set()
        n = total = 0
        for category, variables in entries.items():
            size = sum(memory(v) for v in variables.values())
            top = self._step_item(category, variables, size)
            for name, value in variables.items():
                self._item(top, name, value, desc.get(name, ""), 0)
                total += memory(value, seen=seen)
                n += 1
            if not variables:
                top.setDisabled(True)
                top.setText(1, "nothing yet (run this step first)")
            top.setExpanded(category in expanded)
        for col, width in enumerate((340, 420, 120, 110, 70)):
            self.tree.setColumnWidth(col, width)
        self.total.setText("%d variables, about %s in memory. %s"
                           % (n, human_bytes(total), note))
        self.apply_filter()

    def context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None or id(item) not in self.values:
            return
        menu = QtWidgets.QMenu(self)
        act = menu.addAction("Show value")
        run = _qexec(menu)
        if run(self.tree.viewport().mapToGlobal(pos)) is act:
            self.show_value(item)

    def show_value(self, item):
        name, value = self.values[id(item)]
        path = [item.text(0)]
        parent = item.parent()
        while parent is not None:
            path.insert(0, parent.text(0))
            parent = parent.parent()
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(" > ".join(path))
        dlg.resize(800, 560)
        lay = QtWidgets.QVBoxLayout(dlg)
        fmt, size = describe(value)
        lay.addWidget(QtWidgets.QLabel("<b>%s</b> -- %s%s" % (
            " &gt; ".join(path), fmt, ", " + size if size else "")))
        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        text.setFont(QtGui.QFontDatabase.systemFont(
            _E("QFontDatabase.SystemFont.FixedFont")))
        try:
            content = value_text(value)
        except Exception as exc:                  # noqa: BLE001
            content = "(value cannot be shown: %s)" % exc
        if len(content) > 400000:
            content = content[:400000] + "\n... (cut)"
        text.setPlainText(content)
        lay.addWidget(text, 1)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(dlg.accept)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close)
        lay.addLayout(row)
        dlg.show()
        self._value_dialogs = getattr(self, "_value_dialogs", []) + [dlg]



# --------------------------------------------------------------------------- #
#  File > Save variables 
# --------------------------------------------------------------------------- #
class Lazy(object):
    """A variable computed only when it is saved (e.g. the calibrated
    velocity: only the uncalibrated data are kept in memory)."""

    def __init__(self, fn, fmt="computed when saved", size=""):
        self.fn, self.fmt, self.size = fn, fmt, size

    def __call__(self):
        return self.fn()


CATEGORIES = ("Run and velocity (px, px/frame)",
              "Calibrated (computed when saved)",
              "ROI fields and ROI quantities",
              "Mask",
              "Calibration and time base",
              "Parameters",
              "Other")
_CAT_OF = {
    "Run and velocity (px, px/frame)": (
        "RUN", "RUN_DIR", "PIV_FILE", "LOG_FILE", "VELOCITY",
        "VELOCITIES_IN_FILE", "X_PX", "Z_PX", "U_PX", "V_PX", "REJECT", "NT",
        "NOTES", "VALID_FRACTION", "MAT_VARIABLES", "LOG_ROW"),
    "ROI fields and ROI quantities": ("ROI_FIELDS", "ROI_QUANTITIES"),
    "Mask": ("MASK", "MASK_NAME", "MASK_AREAS", "MASKS"),
    "Calibration and time base": ("CALIBRATION", "TIME_BASE",
                                  "CALIBRATION_SEGMENT"),
    "Parameters": ("PARAMS", "PARAM_FILE"),
}
MAX_DEPTH = 3


def category_of(name, groups=None):
    for cat, names in (groups or _CAT_OF.items()):
        if name in names:
            return cat
    return "Other"


def all_entries(source, extra=None):
    """({category: {name: value}}, {name: description}) of the variables in
    memory (source) and those computed when saved (extra). A source giving
    a 4th item [(category, [names])] sets the categories and their order
    (e.g. one per step of the GUI); every category is listed, even empty."""
    got = source()
    data, help_ = got[0], got[1]
    groups = list(got[3]) if len(got) > 3 and got[3] else None
    cats = [c for c, _n in groups] if groups else list(CATEGORIES)
    if "Other" not in cats:
        cats.append("Other")
    out = {c: {} for c in cats}
    desc = dict(help_)
    for name, value in data.items():
        out[category_of(name, groups)][name] = value
    for cat, items in (extra() if extra else {}).items():
        for name, (value, d) in items.items():
            out.setdefault(cat, {})[name] = value
            desc[name] = d
    if groups:                                   # the order of the steps
        rank = {n: k for _c, names in groups for k, n in enumerate(names)}
        for c in out:
            out[c] = dict(sorted(out[c].items(),
                                 key=lambda kv: rank.get(kv[0], 999)))
    if not out["Other"]:
        del out["Other"]
    return out, desc


def select_values(source, extra, paths):
    """(selection, missing): {name: value} of the entries at the paths
    (variable, key, key, ...) -- a dictionary keeps only the entries
    asked; 'computed when saved' values computed -- and the paths that do
    not exist (e.g. an ROI that this run does not have)."""
    entries, _desc = all_entries(source, extra)
    values = {}
    for variables in entries.values():
        values.update(variables)
    out, missing = {}, []
    for path in paths:
        path = tuple(path)
        try:
            value = values[path[0]]
            for key in path[1:]:
                value = value[key]
        except (KeyError, IndexError, TypeError):
            missing.append(path)
            continue
        if isinstance(value, Lazy):
            value = value()
        target = out
        for key in path[:-1]:
            target = target.setdefault(key, {})
        target[path[-1]] = value
    return out, missing


class SaveVariablesDialog(QtWidgets.QDialog, _StepTree):
    """File > Save variables :
    the variables in categories, each category with a tick box selecting
    all its variables, and every variable -- and every entry of a
    dictionary variable -- with its own tick box. Save asks the file name,
    folder and format (npz, NetCDF, HDF5, txt, MATLAB).

    source() -> ({name: value}, {name: description}, note) (as for View >
    Show all variables); extra() -> {category: {name: (Lazy, description)}}
    (values computed only when saved); folder() -> the folder proposed;
    stem() -> the file name proposed."""

    COLUMNS = ("Variable", "Description", "Format", "Size", "Memory")

    def __init__(self, source, parent=None, log=print, extra=None,
                 folder=None, stem=None, select_only=False, ticked=None):
        """select_only: the window only chooses the variables (batch
        processing): its button 'Use this selection' keeps the paths
        ticked in self.selected_paths and closes it (accepted)."""
        super().__init__(parent)
        self.select_only = select_only
        self.selected_paths = None
        self.source, self.extra = source, extra or (lambda: {})
        self.folder = folder or (lambda: "")
        self.stem = stem or (lambda: "PIV_variables")
        self.log = log
        self.paths = {}            # id(item) -> path (name, key, key, ...)
        self.last_files = []
        self.setWindowTitle("Save variables")
        self.resize(1050, 700)
        lay = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Tick the variables to save -- they are grouped by step (menu > "
            "step); a step ticks all its variables, a variable ticks all "
            "its entries; All / None act on the entries shown (Search) -- "
            "then press "
            "Save: the file name, the folder and the format (npz, NetCDF, "
            "HDF5, txt, MATLAB) are asked. The data in memory are "
            "uncalibrated (px, px/frame); the calibrated velocity (m, m/s, "
            "s, mask applied) is computed when saved. The list is updated "
            "after a Load and every change of the run.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self._make_search(lay)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(list(self.COLUMNS))
        self.tree.setAlternatingRowColors(True)
        self.tree.itemChanged.connect(self.update_total)
        lay.addWidget(self.tree, 1)
        self._make_details(lay)
        self.total = QtWidgets.QLabel("")
        self.total.setStyleSheet("color: gray;")
        lay.addWidget(self.total)
        row = QtWidgets.QHBoxLayout()
        for text, state in (("All", True), ("None", False)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda checked=False, on=state:
                              self.tick_all(on))
            row.addWidget(b)
        for text, slot in (("Expand all", lambda: self.tree.expandAll()),
                           ("Collapse", self.collapse),
                           ("Refresh", self.fill)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        self.save_button = QtWidgets.QPushButton(
            "Use this selection" if select_only else "Save...")
        self.save_button.setStyleSheet("font-weight: bold; "
                                       "padding: 4px 16px;")
        self.save_button.clicked.connect(
            self._use if select_only else (lambda: self.save()))
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.close)
        row.addWidget(self.save_button)
        row.addWidget(close)
        lay.addLayout(row)
        self.fill()
        if ticked:                               # a selection made before
            self.set_ticked_paths(ticked)
        if select_only:
            self.setWindowTitle("Select the variables to save (batch)")

    def _use(self):
        self.selected_paths = [list(p) for p in self.ticked_paths()]
        self.accept()

    def set_ticked_paths(self, paths):
        """Tick the entries at these paths (a variable: all its entries)."""
        want = {tuple(p) for p in paths}
        self.tree.blockSignals(True)
        for it in self._leaves():
            p = tuple(self.paths[id(it)])
            on = any(p[:len(w)] == w for w in want)
            it.setCheckState(0, _E("Qt.CheckState.Checked") if on
                             else _E("Qt.CheckState.Unchecked"))
        self.tree.blockSignals(False)
        self.fill()

    # ---- content ---------------------------------------------------------- #
    def entries(self):
        """({category: {name: value}}, {name: description})."""
        return all_entries(self.source, self.extra)

    def _add(self, parent, path, value, description, level, ticked):
        if isinstance(value, Lazy):
            fmt, size, mem = value.fmt, value.size, ""
        else:
            fmt, size = describe(value)
            mem = human_bytes(memory(value))
        description = description or entry_description(path[-1])
        item = QtWidgets.QTreeWidgetItem(parent, [str(path[-1]),
                                                  short_text(description),
                                                  fmt, size, mem])
        item.setToolTip(1, description)
        children = isinstance(value, dict) and value and level < MAX_DEPTH
        flags = item.flags() | _E("Qt.ItemFlag.ItemIsUserCheckable")
        if children:
            flags |= _E("Qt.ItemFlag.ItemIsAutoTristate")
        item.setFlags(flags)
        self.paths[id(item)] = path
        if children:
            for key, child in value.items():
                self._add(item, path + (key,), child, "", level + 1, ticked)
        else:
            item.setCheckState(0, _E("Qt.CheckState.Checked")
                               if path in ticked or path[:1] in ticked
                               else _E("Qt.CheckState.Unchecked"))
        return item

    def fill(self):
        ticked = set(self.ticked_paths()) if self.tree.topLevelItemCount() \
            else set()
        expanded = {self.tree.topLevelItem(i).text(0).split("  (")[0]
                    for i in range(self.tree.topLevelItemCount())
                    if self.tree.topLevelItem(i).isExpanded()}

        self.tree.blockSignals(True)
        self.tree.clear()
        self.paths = {}
        entries, desc = self.entries()
        for category, variables in entries.items():
            size = sum(memory(v) for v in variables.values()
                       if not isinstance(v, Lazy))
            top = self._step_item(category, variables, size)
            top.setFlags(top.flags() | _E("Qt.ItemFlag.ItemIsUserCheckable")
                         | _E("Qt.ItemFlag.ItemIsAutoTristate"))
            self.paths[id(top)] = ("__category__", category)
            for name in variables:
                self._add(top, (name,), variables[name], desc.get(name, ""),
                          0, ticked)
            if not variables:
                top.setCheckState(0, _E("Qt.CheckState.Unchecked"))
                top.setDisabled(True)
                top.setText(1, "nothing yet (run this step first)")
            top.setExpanded(category in expanded)  # steps collapsed first
        self.tree.blockSignals(False)
        for column, width in enumerate((340, 420, 120, 110, 70)):
            self.tree.setColumnWidth(column, width)
        self.apply_filter()
        self.update_total()

    def collapse(self):
        self.tree.collapseAll()
        for i in range(self.tree.topLevelItemCount()):
            self.tree.topLevelItem(i).setExpanded(True)

    # ---- ticks ------------------------------------------------------------ #
    def _leaves(self, item=None):
        items = ([self.tree.topLevelItem(i)
                  for i in range(self.tree.topLevelItemCount())]
                 if item is None else
                 [item.child(i) for i in range(item.childCount())])
        for child in items:
            if child.childCount():
                for x in self._leaves(child):
                    yield x
            elif self.paths.get(id(child), ("__category__",))[0] \
                    != "__category__":
                yield child

    def ticked_paths(self):
        """Paths (variable, key, ...) of the entries ticked."""
        return [self.paths[id(it)] for it in self._leaves()
                if it.checkState(0) == _E("Qt.CheckState.Checked")]

    def ticked(self):
        return sorted({p[0] for p in self.ticked_paths()})

    def tick_all(self, on):
        """All / None: every entry shown (with a search: those found)."""
        self.tree.blockSignals(True)
        for it in self._leaves():
            if not self._shown(it):
                continue
            it.setCheckState(0, _E("Qt.CheckState.Checked") if on
                             else _E("Qt.CheckState.Unchecked"))
        self.tree.blockSignals(False)
        self.fill()                              # parents follow (tristate)

    def tick_category(self, category, on=True):
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            if self.paths.get(id(top)) == ("__category__", category):
                top.setCheckState(0, _E("Qt.CheckState.Checked") if on
                                  else _E("Qt.CheckState.Unchecked"))

    def set_ticked(self, names):
        self.tree.blockSignals(True)
        for it in self._leaves():
            it.setCheckState(0, _E("Qt.CheckState.Checked")
                             if self.paths[id(it)][0] in names
                             else _E("Qt.CheckState.Unchecked"))
        self.tree.blockSignals(False)
        self.fill()

    def selection(self):
        """{name: value} of the entries ticked; a dictionary keeps only its
        entries ticked (nested); 'computed when saved' values computed."""
        return select_values(self.source, self.extra,
                             self.ticked_paths())[0]

    def update_total(self, *_):
        paths = self.ticked_paths()
        names = {p[0] for p in paths}
        self.total.setText("%d variable(s) ticked (%d entries)"
                           % (len(names), len(paths)))
        self.save_button.setEnabled(bool(paths))

    # ---- save ------------------------------------------------------------- #
    def save(self, path=None, chosen_filter=""):
        """Write the entries ticked; returns the files written."""
        import os
        import gui_export
        if not self.ticked_paths():
            return []
        if path is None:
            folder = self.folder() or os.path.expanduser("~")
            if isinstance(folder, (tuple, list)):  # (path, created)
                folder = folder[0]
            folder = str(folder)
            path, chosen_filter = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save variables", os.path.join(folder, self.stem()),
                variable_filter())
            if not path:
                return []
        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(_E("Qt.CursorShape.WaitCursor")))
        try:
            selection = self.selection()
            fmt = variable_format(path, chosen_filter)
            path = gui_export.with_extension(
                os.path.splitext(path)[0] if path.lower().endswith(".csv")
                else path, fmt)
            tree = gui_export.to_tree(selection)
            files = gui_export.write(path, fmt, tree,
                                     title="PIV GUI variables: "
                                           + ", ".join(selection))
        except Exception as error:               # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(
                self, "Save variables",
                "Could not save the variables:\n%s" % error)
            return []
        QtWidgets.QApplication.restoreOverrideCursor()
        self.last_files = list(files)
        self.saved_folder = os.path.dirname(files[0])
        self.log("%d variable(s) saved (%s): %s" % (
            len(selection), fmt, ", ".join(files)))
        return files
