"""Example script (Advanced > Python script): a map of a quantity with the
velocity vectors on top -- the figure of Processing > Maps and Quiver --
drawn here with plain matplotlib from the calibrated data, so that every
step can be read and changed.

Change the parameters below, then press Execute. The figure appears in the
figure list under 'Python script'; the arrays are kept in `results`
(Show / Save variables, saved with the session).
"""

# ---- parameters -------------------------------------------------------- #
ROI = "ROI01"                 # "Full", or an ROI of Define ROI: "ROI01", ...
FRAME = "mean"                # a frame index (0, 1, ...), "mean" or "rms"
BACKGROUND = "vorticity"      # None, "velocity norm", "U", "V",
                              # "divergence", "vorticity", "kinetic energy"
QUIVER = True                 # draw the velocity vectors
SKIP = 1                      # one vector out of SKIP in each direction
VECTOR_SCALE = 1.0            # the 99th-percentile speed is drawn
                              # VECTOR_SCALE vector spacings long
QUIVER_COLOUR = "black"
CMAP = "auto"                 # "auto": RdBu_r for signed maps, else viridis
FIGURE_NAME = "Map and quiver (script)"
# ------------------------------------------------------------------------ #

import warnings

if not d.can_calibrate:
    raise RuntimeError("the run is not calibrated (Pre-processing > "
                       "Calibration): this script works in m and m/s")

field = roi_field(ROI)                   # "ROI01" -> "Validated_ROI01"
vw = view(field, calibrated=True)        # the ROI (or full) field, in m
x, z = vw.x, vw.z                        # grid (ny, nx), m


def at_frame(a):
    """(ny, nx, nt) -> (ny, nx) for FRAME (an index, the mean or rms)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if FRAME == "mean":
            return np.nanmean(a, axis=2)
        if FRAME == "rms":
            return np.sqrt(np.nanmean(a * a, axis=2))
        return a[:, :, int(FRAME)]


# the velocity (m/s), mask applied
u_all, v_all = vw.uv()
u, v = at_frame(u_all), at_frame(v_all)
speed = np.hypot(u, v)

fig = figure(FIGURE_NAME)
ax = fig.add_subplot(111)

# the background map: the stored quantity of the ROI (Define ROI > Apply)
# or of the full field
if BACKGROUND:
    q = at_frame(quantity(BACKGROUND, field))
    signed = BACKGROUND in ("U", "V", "divergence", "vorticity")
    cmap = CMAP if CMAP != "auto" else ("RdBu_r" if signed else "viridis")
    good = q[np.isfinite(q)]
    if signed and good.size:                     # centred on 0
        lim = np.percentile(np.abs(good), 99.5)
        vmin, vmax = -lim, lim
    elif good.size:
        vmin, vmax = np.percentile(good, (0.5, 99.5))
    else:
        vmin = vmax = None
    im = ax.pcolormesh(x, z, q, shading="nearest", cmap=cmap,
                       vmin=vmin, vmax=vmax)
    unit = lib.quantity_unit(vw, BACKGROUND).replace("$^2$", "²")
    fig.colorbar(im, ax=ax, label="%s (%s)" % (BACKGROUND, unit))
    results["background"] = q

# the vectors: every SKIP-th one; the 99th-percentile speed drawn
# VECTOR_SCALE vector spacings long
if QUIVER:
    s = (slice(None, None, SKIP), slice(None, None, SKIP))
    spacing = float(np.nanmedian(np.abs(np.diff(x[0, :])))) * SKIP
    u99 = float(np.nanpercentile(speed, 99)) if np.isfinite(speed).any() \
        else 1.0
    ax.quiver(x[s], z[s], u[s], v[s], color=QUIVER_COLOUR, angles="xy",
              scale_units="xy", scale=u99 / (VECTOR_SCALE * spacing),
              width=0.0025)
    ax.set_title("arrow of %.3g m/s = %.3g m" % (u99, VECTOR_SCALE * spacing),
                 fontsize=8, loc="right")

ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
frame_txt = FRAME if isinstance(FRAME, str) else d.frame_text(FRAME)
fig.suptitle("%s\n%s, %s%s" % (d.run, ROI, frame_txt,
                               ", " + BACKGROUND if BACKGROUND else ""),
             fontsize=9)

results.update(x=x, z=z, u=u, v=v, speed=speed)
log("map and quiver of %s (%s, %s) drawn" % (ROI, frame_txt, BACKGROUND))
