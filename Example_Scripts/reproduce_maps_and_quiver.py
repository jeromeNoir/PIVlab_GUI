"""Example script (Advanced > Python script): adds an item to the Advanced
menu, 'Reproduce maps and quiver', that reproduces Processing > Maps and
Quiver -- the same Display options, the same figure (lib.display_figure)
and the same Make Movie section -- built by a script with the GUI's own
pieces. It shows how a script can add a whole step to the GUI.

Execute it once: the item appears at the end of the Advanced menu (and the
step opens). Its figure is listed under 'Reproduce maps and quiver' in the
figure list. The item is made again when a session that executed this
script is loaded. Velocity (calibrated / uncalibrated) and Mask are those
of the run, shared with Maps and Quiver (as in the original step).

Names used (given to every script): gui (the main window), G (the GUI
module: its classes and constants), lib, np, QtWidgets, QtCore, QtGui,
add_advanced_item(id, label, panel), d (the loaded run).
"""

ITEM_ID = "repro_maps"                     # the id of the new step
ITEM_LABEL = "Reproduce maps and quiver"   # its name in the menu
FIGURE = "Maps and Quiver (reproduced)"    # its figure

E = G.E                                    # Qt enums (Qt5 and Qt6)
ROLE = E("Qt.ItemDataRole.UserRole")


class ReproMapsPanel(QtWidgets.QWidget):
    """The panel of the new step: the Display box of Maps and Quiver
    (velocity, frame, ROI, mask, quiver, background, smoothing, shading,
    colour map), its figure, and Make Movie."""

    STEP = ITEM_ID
    FIG = FIGURE
    MOVIE = "Movie (reproduced)"
    KIND = "maps"                          # the movie draws display_figure

    def __init__(self, main):
        super().__init__()
        self.main = main
        lay = QtWidgets.QVBoxLayout(self)
        self.note = QtWidgets.QLabel("Load a run first (File > Load data).")
        lay.addWidget(self.note)
        box = QtWidgets.QGroupBox("Display")
        g = QtWidgets.QGridLayout(box)
        g.setColumnStretch(1, 1)
        r = 0
        # Velocity: calibrated / uncalibrated -- the run's choice (the same
        # combo as Maps and Quiver, mirrored)
        self.units = QtWidgets.QComboBox()
        g.addWidget(QtWidgets.QLabel("Velocity"), r, 0)
        g.addWidget(self.units, r, 1, 1, 2); r += 1
        # Frame: single frame / time mean / rms, slider + number
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["single frame", "time mean", "rms"])
        self.frame = QtWidgets.QSlider(E("Qt.Orientation.Horizontal"))
        self.frame_spin = QtWidgets.QSpinBox()
        self.frame_info = QtWidgets.QLabel("")
        g.addWidget(QtWidgets.QLabel("Frame"), r, 0)
        g.addWidget(self.mode, r, 1, 1, 2); r += 1
        g.addWidget(self.frame, r, 1)
        g.addWidget(self.frame_spin, r, 2); r += 1
        g.addWidget(self.frame_info, r, 1, 1, 2); r += 1
        # ROI: Full + the active ROIs (the GUI's own ROI drop-down)
        self.roi = G.RoiCombo(main)
        g.addWidget(QtWidgets.QLabel("ROI"), r, 0)
        g.addWidget(self.roi, r, 1, 1, 2); r += 1
        # Mask: the run's mask applied (mirrors Maps and Quiver's)
        self.mask = QtWidgets.QComboBox()
        g.addWidget(QtWidgets.QLabel("Mask"), r, 0)
        g.addWidget(self.mask, r, 1, 1, 2); r += 1
        # Quiver
        self.quiver = QtWidgets.QCheckBox("Quiver")
        self.quiver.setChecked(True)
        g.addWidget(self.quiver, r, 0, 1, 3); r += 1
        self.qcolour = QtWidgets.QComboBox()
        self.qcolour.addItems(lib.QUIVER_COLOURS)
        self.qcolour.setCurrentText("black")
        self.skip = QtWidgets.QSpinBox()
        self.skip.setRange(1, 50)
        self.skip.setPrefix("every ")
        self.skip.setSuffix(" vector(s)")
        g.addWidget(QtWidgets.QLabel("   Quiver"), r, 0)
        g.addWidget(self.qcolour, r, 1)
        g.addWidget(self.skip, r, 2); r += 1
        self.vscale = QtWidgets.QDoubleSpinBox()
        self.vscale.setRange(0.05, 50.0)
        self.vscale.setSingleStep(0.25)
        self.vscale.setValue(1.0)
        self.vscale.setPrefix("x ")
        g.addWidget(QtWidgets.QLabel("   Vector scale"), r, 0)
        g.addWidget(self.vscale, r, 1, 1, 2); r += 1
        # Background: a map or an image
        self.bkg_on = QtWidgets.QCheckBox("Background")
        g.addWidget(self.bkg_on, r, 0, 1, 3); r += 1
        self.bkg = QtWidgets.QComboBox()
        self.bkg.addItems(G.BACKGROUNDS)
        g.addWidget(self.bkg, r, 1, 1, 2); r += 1
        self.image = QtWidgets.QComboBox()
        self.image_browse = QtWidgets.QPushButton("Browse...")
        g.addWidget(QtWidgets.QLabel("   image"), r, 0)
        g.addWidget(self.image, r, 1)
        g.addWidget(self.image_browse, r, 2); r += 1
        # Smoothing, shading, colour map (as every map of the GUI)
        self.smooth_on = QtWidgets.QCheckBox("Smoothing")
        self.smooth = G._smooth_spin()
        g.addWidget(self.smooth_on, r, 0)
        g.addWidget(self.smooth, r, 1, 1, 2); r += 1
        self.shading_on = QtWidgets.QCheckBox("Shading")
        self.shading = QtWidgets.QComboBox()
        self.shading.addItems(lib.MAP_SHADINGS)
        g.addWidget(self.shading_on, r, 0)
        g.addWidget(self.shading, r, 1, 1, 2); r += 1
        self.cmap = QtWidgets.QComboBox()
        self.cmap.addItems(lib.MAP_CMAPS)
        g.addWidget(QtWidgets.QLabel("Colour map"), r, 0)
        g.addWidget(self.cmap, r, 1, 1, 2); r += 1
        lay.addWidget(box)
        # Make Movie: the GUI's own movie box, driven by this panel
        self.mv = G.MovieBox(main, self)
        lay.addWidget(self.mv)
        lay.addStretch(1)

        redraw = lambda *_: self.draw()
        for w in (self.mode, self.roi, self.qcolour, self.bkg, self.cmap,
                  self.image, self.shading):
            w.currentIndexChanged.connect(redraw)
        for w in (self.skip, self.vscale, self.smooth):
            w.valueChanged.connect(redraw)
        for w in (self.quiver, self.bkg_on, self.smooth_on, self.shading_on):
            w.toggled.connect(redraw)
        self.frame.valueChanged.connect(self.frame_spin.setValue)
        self.frame_spin.valueChanged.connect(self.frame.setValue)
        self.frame.valueChanged.connect(
            lambda *_: None if self.frame.isSliderDown() else self.draw())
        self.frame.sliderReleased.connect(redraw)
        self.units.activated.connect(self._units_chosen)
        self.mask.activated.connect(self._mask_chosen)
        self.image_browse.clicked.connect(self._browse)
        self.bkg.currentIndexChanged.connect(lambda *_: self._enable())
        self.bkg_on.toggled.connect(lambda *_: self._enable())
        self._enable()

    # ---- the run's choices, shared with Maps and Quiver ------------------ #
    def _mirror(self, mine, theirs):
        mine.blockSignals(True)
        mine.clear()
        for i in range(theirs.count()):
            mine.addItem(theirs.itemIcon(i), theirs.itemText(i),
                         theirs.itemData(i, ROLE))
        mine.setCurrentIndex(theirs.currentIndex())
        mine.blockSignals(False)

    def _units_chosen(self, i):
        self.main.load_panel.units.setCurrentIndex(i)   # the run's choice
        self.sync()

    def _mask_chosen(self, i):
        self.main.load_panel.mask_combo.setCurrentIndex(i)
        self.sync()

    def _browse(self):
        lp = self.main.load_panel
        path, _f = QtWidgets.QFileDialog.getOpenFileName(
            self, "Background image", lp.run_dir or lp.root_dir or "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.mat)")
        if path:
            self.image.addItem(G.os.path.basename(path), path)
            self.image.setCurrentIndex(self.image.count() - 1)

    def _enable(self):
        img = self.bkg.currentText() == "image"
        self.bkg.setEnabled(self.bkg_on.isChecked())
        for w in (self.image, self.image_browse):
            w.setEnabled(self.bkg_on.isChecked() and img)

    def sync(self):
        """The run's lists: velocity, mask, ROIs, images, frames."""
        d = self.main.data
        lp = self.main.load_panel
        self.note.setVisible(d is None)
        self._mirror(self.units, lp.units)
        self._mirror(self.mask, lp.mask_combo)
        self.roi.refill()
        if d is not None and getattr(self, "_movie_run", None) is not d:
            self.mv.data_loaded(d)        # the movie's frame range
            self._movie_run = d
        if d is not None:
            for w in (self.frame, self.frame_spin):
                w.setRange(0, d.nt - 1)
            keep = self.image.currentData(ROLE)
            self.image.blockSignals(True)
            self.image.clear()
            for p in lib.find_background_images(lp.run_dir, lp.root_dir):
                self.image.addItem(G.os.path.basename(p), p)
            i = self.image.findData(keep, ROLE)
            self.image.setCurrentIndex(max(i, 0))
            self.image.blockSignals(False)

    # ---- the options and the figure -------------------------------------- #
    def options(self):
        """The options of lib.display_figure -- those of Maps and Quiver."""
        rp = self.main.roi_panel
        return dict(
            frame={1: "mean", 2: "rms"}.get(self.mode.currentIndex(),
                                            self.frame.value()),
            roi=self.roi.currentData(ROLE) or "Full",
            quiver=self.quiver.isChecked(),
            quiver_colour=self.qcolour.currentText(),
            skip=self.skip.value(), vec_scale=self.vscale.value(),
            smoothing=(self.smooth.value() if self.smooth_on.isChecked()
                       else 0),
            background=(self.bkg.currentText() if self.bkg_on.isChecked()
                        else None),
            image=self.image.currentData(ROLE),
            cmap=self.cmap.currentText(),
            shading=(self.shading.currentText()
                     if self.shading_on.isChecked() else "nearest"),
            rois_shown=rp.shown("rect"), lines_shown=rp.shown("line"))

    def draw(self, force=False):
        d = self.main.data
        if d is None:
            return
        frame = self.frame.value()
        self.frame_info.setText(d.frame_text(frame)
                                if self.mode.currentIndex() == 0 else "")
        F = self.main.figures
        page = F.find(self.STEP, self.FIG)
        if page is None and not force:
            return
        try:
            if page is None:
                fig = lib.display_figure(d, fig=lib.new_figure(),
                                         **self.options())
                page = F.add(self.STEP, self.FIG, fig,
                             data_fn=lambda: lib.display_data(
                                 self.main.data, self.options()["frame"],
                                 None, self.options()["roi"]),
                             update_fn=lambda: self.draw(force=True))
            else:
                lib.display_figure(d, fig=page.figure, **self.options())
                page.redraw()
        except Exception as exc:
            self.main.status("Figure error: %s" % exc, log=True)

    # ---- what the GUI calls ----------------------------------------------- #
    def open(self):                       # the item is opened
        self.sync()
        self.draw(force=True)

    def data_loaded(self, d):             # a run is loaded
        self.sync()
        self.draw()

    # ---- what Make Movie calls -------------------------------------------- #
    def movie_options(self):
        return dict(self.options(), field=self.main.data.default)

    def movie_note(self, o):
        return "%s%s" % (o.get("roi"), ", background %s" % o["background"]
                         if o.get("background") else "")

    def movie_stem(self, o):
        what = (o.get("background") or "quiver").replace(" ", "_")
        roi = "" if o.get("roi") in (None, "Full") else "_" + o["roi"]
        return "repro_" + what + roi


panel = add_advanced_item(ITEM_ID, ITEM_LABEL, ReproMapsPanel(gui))
gui.set_step(ITEM_ID)                     # open it now
results["item"] = ITEM_LABEL
log("'%s' added to the Advanced menu" % ITEM_LABEL)
