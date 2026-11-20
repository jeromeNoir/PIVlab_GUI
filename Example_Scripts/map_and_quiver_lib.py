"""Example script (Advanced > Python script): the figure of Processing >
Maps and Quiver, made by the same function as the GUI
(lib.display_figure), with its options as parameters. The figure appears
in the figure list under 'Python script'; its data (lib.display_data) are
kept in `results`.
"""

# ---- parameters -------------------------------------------------------- #
ROI = "Full"                  # "Full", or an ROI of Define ROI: "ROI01", ...
FRAME = 0                     # a frame index, "mean" or "rms"
BACKGROUND = "velocity norm"  # None, "velocity norm", "U", "V",
                              # "divergence", "vorticity", "kinetic energy",
                              # "image"
IMAGE = None                  # the image file, for BACKGROUND = "image"
QUIVER = True
QUIVER_COLOUR = "black"       # "|u|", "black", "white", "red", ...
SKIP = 2                      # every SKIP-th vector
VECTOR_SCALE = 1.0
SMOOTHING = 0                 # Gaussian smoothing of the map (sigma, in
                              # vector spacings; 0: none)
SHADING = "nearest"           # "nearest", "gouraud", "bilinear", ...
CMAP = "auto"
FIGURE_NAME = "Maps and Quiver (lib)"
# ------------------------------------------------------------------------ #

if not d.can_calibrate:
    raise RuntimeError("calibrate the run first (Pre-processing > "
                       "Calibration)")
if not d.calibrated:
    d.set_calibrated(True)               # m, m/s, s

field = roi_field(ROI)
rp = gui.roi_panel                       # the ROIs / lines ticked there
opts = dict(frame=FRAME, roi=field, quiver=QUIVER,
            quiver_colour=QUIVER_COLOUR, skip=SKIP, vec_scale=VECTOR_SCALE,
            background=BACKGROUND, image=IMAGE, smoothing=SMOOTHING,
            shading=SHADING, cmap=CMAP, rois_shown=rp.shown("rect"),
            lines_shown=rp.shown("line"))
fig = figure(FIGURE_NAME)
lib.display_figure(d, fig=fig, **opts)
results.update(lib.display_data(d, FRAME, None, field))
log("Maps and Quiver of %s, frame %s, background %s" % (ROI, FRAME,
                                                        BACKGROUND))
