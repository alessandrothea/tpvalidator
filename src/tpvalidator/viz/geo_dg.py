from pathlib import Path

import matplotlib as mpl
from matplotlib import animation
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal, Sequence

from tpvalidator.detgeometry import FDVDGeometry, GeoData, Range1D


def _get_edges(tpcs: tuple, axis: str) -> np.ndarray:
    """Collect unique bin edges for one axis, merging floating-point near-duplicates."""
    vals = set()
    for t in tpcs:
        r: Range1D = getattr(t, f'{axis}_range')
        vals.add(round(r.min, 3))
        vals.add(round(r.max, 3))
    return np.array(sorted(vals))


def _find_bin(edges: np.ndarray, val: float) -> int:
    return int(np.searchsorted(edges, round(val, 3), side='left'))


_DETECTOR_DEFAULTS = {
    'vd': dict(elev=-45,  azim=-45, roll=55, opdet_color_axis='x'),
    'hd': dict(elev=50,  azim=35,  roll=120, opdet_color_axis='y'),
}

_ANNOTATION_DEFAULTS = {
    'beam_length': 400,      # cm
    'beam_tip_z': -200,      # cm — z position of the arrowhead
    'beam_color': 'tomato',
    'beam_linewidth': 2,
    'beam_arrow_length_ratio': 0.2,
    'beam_fontsize': 10,
    'drift_color': 'steelblue',
    'drift_linewidth': 2,
    'drift_arrow_length_ratio': 0.1,
    'drift_fontsize': 10,
    'drift_y_offset': -200,  # cm below cryostat y_min
    'dim_color': '#1a3a5c',
    'dim_linewidth': 0.8,
    'dim_arrow_length_ratio': 0.06,
    'dim_fontsize': 10,
    'beam_anchor_x': 0,     # cm — transverse x position of beam arrow
    'beam_anchor_y': 0,     # cm — transverse y position of beam arrow
    'drift_anchor_z': None, # cm — z position of drift arrow; None = cryostat z midpoint
}

# Maps plane name → (horizontal axis, vertical axis, depth/colour axis)
_PLANE_AXES: dict[str, tuple[str, str, str]] = {
    'xy': ('x', 'y', 'z'),
    'xz': ('x', 'z', 'y'),
    'yz': ('y', 'z', 'x'),
}


def _plot_box_edges(ax, x_range: Range1D, y_range: Range1D, z_range: Range1D, **kwargs):
    """Draw the 12 edges of a rectangular box on a 3-D axes.

    A ``label`` kwarg is applied only to the first edge so the legend shows one
    entry per box rather than one per edge.
    """
    x0, x1 = x_range.min, x_range.max
    y0, y1 = y_range.min, y_range.max
    z0, z1 = z_range.min, z_range.max
    edges = [
        # 4 edges along x
        ([x0, x1], [y0, y0], [z0, z0]),
        ([x0, x1], [y1, y1], [z0, z0]),
        ([x0, x1], [y0, y0], [z1, z1]),
        ([x0, x1], [y1, y1], [z1, z1]),
        # 4 edges along y
        ([x0, x0], [y0, y1], [z0, z0]),
        ([x1, x1], [y0, y1], [z0, z0]),
        ([x0, x0], [y0, y1], [z1, z1]),
        ([x1, x1], [y0, y1], [z1, z1]),
        # 4 edges along z
        ([x0, x0], [y0, y0], [z0, z1]),
        ([x1, x1], [y0, y0], [z0, z1]),
        ([x0, x0], [y1, y1], [z0, z1]),
        ([x1, x1], [y1, y1], [z0, z1]),
    ]
    label = kwargs.pop('label', '_nolegend_')
    for i, (xs, ys, zs) in enumerate(edges):
        ax.plot(xs, ys, zs, label=(label if i == 0 else '_nolegend_'), **kwargs)


def _plot_box_dimensions(ax, x_range: Range1D, y_range: Range1D, z_range: Range1D,
                          color, **text_kwargs):
    """Draw engineering-style dimension lines along all three axes of a box.

    Lines are placed on the 3-D axis spines (at the current axis-limit
    boundaries) so they sit outside all geometry.  Arrowheads mark both ends
    and a ``"NNN.N cm"`` label sits at the midpoint of each line.

    Must be called *after* ``ax.set_box_aspect`` so that the axis limits are
    already finalised.
    """
    x0, x1 = x_range.min, x_range.max
    y0, y1 = y_range.min, y_range.max
    z0, z1 = z_range.min, z_range.max
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0

    xl = ax.get_xlim()[0]
    yl = ax.get_ylim()[0]
    zl = ax.get_zlim()[0]

    ann = _ANNOTATION_DEFAULTS
    txt = dict(fontsize=ann['dim_fontsize'], color=color)
    txt.update(text_kwargs)

    arrow = dict(color=color, alpha=0.9, linewidth=ann['dim_linewidth'],
                 arrow_length_ratio=ann['dim_arrow_length_ratio'], linestyle='solid')

    ax.quiver(x0, yl, zl,  dx, 0, 0, **arrow)
    ax.quiver(x1, yl, zl, -dx, 0, 0, **arrow)
    ax.text((x0 + x1) / 2, yl, zl, f'{dx:.1f} cm', ha='center', va='top', **txt)

    ax.quiver(xl, y0, zl, 0,  dy, 0, **arrow)
    ax.quiver(xl, y1, zl, 0, -dy, 0, **arrow)
    ax.text(xl, (y0 + y1) / 2, zl, f'{dy:.1f} cm', ha='right', va='center', **txt)

    ax.quiver(xl, yl, z0, 0, 0,  dz, **arrow)
    ax.quiver(xl, yl, z1, 0, 0, -dz, **arrow)
    ax.text(xl, yl, (z0 + z1) / 2, f'{dz:.1f} cm', ha='right', va='bottom', **txt)


def _resolve(geo: GeoData | FDVDGeometry) -> GeoData:
    return geo.geo() if isinstance(geo, FDVDGeometry) else geo


def plot_geometry_3d(
    geo: GeoData | FDVDGeometry,
    cmap_name: str = 'tab20',
    show_tpcs: bool = True,
    show_opdets: bool = True,
    show_cryostat: bool = True,
    show_cryostat_dimensions: bool = False,
    show_axes: bool = True,
    show_beam: bool = False,
    show_drift: bool = False,
    drift_type: str = 'vd',
    annotation_overrides: dict | None = None,
    show_grid: bool = True,
    **fig_kwargs
) -> tuple[plt.Figure, plt.Axes]:
    """Plot TPC volumes and/or optical detector positions from a GeoData object.

    Parameters
    ----------
    geo:
        A :class:`~tpvalidator.detgeometry.GeoData` instance, or an
        :class:`~tpvalidator.detgeometry.FDVDGeometry` whose ``.geo()`` method
        will be called to obtain one.
    cmap_name:
        Matplotlib colormap used to colour TPC voxels by z-module index.
    show_tpcs:
        Draw TPC volumes as voxels.
    show_opdets:
        Draw optical detector centres as scatter markers.
    show_cryostat:
        Draw the cryostat boundary as a transparent box.
    show_cryostat_dimensions:
        Annotate the cryostat with dimension lines.  Requires ``show_cryostat``.
    show_axes:
        When ``False``, hide axis lines, tick marks, tick labels, and grid planes.
    show_beam:
        Draw a beam arrow along the z axis.
    show_drift:
        Draw a drift arrow along the x axis.
    drift_type:
        ``'vd'`` or ``'hd'``. Controls the initial view angles and opdet colour axis.
    annotation_overrides:
        Dict merged over :data:`_ANNOTATION_DEFAULTS`.
    show_grid:
        When ``False``, hide background panes and grid lines.

    Returns
    -------
    fig, ax
    """
    if drift_type not in _DETECTOR_DEFAULTS:
        raise ValueError(f"drift_type must be 'vd' or 'hd', got {drift_type!r}")
    det_cfg = _DETECTOR_DEFAULTS[drift_type]
    geo = _resolve(geo)
    tpcs = geo.tpcs

    x_edges = _get_edges(tpcs, 'x')
    y_edges = _get_edges(tpcs, 'y')
    z_edges = _get_edges(tpcs, 'z')

    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    nz = len(z_edges) - 1

    filled = np.zeros((nx, ny, nz), dtype=bool)
    for tpc in tpcs:
        ix = _find_bin(x_edges, tpc.x_range.min)
        iy = _find_bin(y_edges, tpc.y_range.min)
        iz = _find_bin(z_edges, tpc.z_range.min)
        filled[ix, iy, iz] = True

    X, Y, Z = np.meshgrid(x_edges, y_edges, z_edges, indexing='ij')

    colors = np.empty(filled.shape, dtype=object)
    cmap = plt.colormaps[cmap_name]
    for iz in range(nz):
        rgba = cmap(iz / max(nz - 1, 1))
        hex_color = '#{:02x}{:02x}{:02x}30'.format(
            int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        )
        colors[:, :, iz] = hex_color

    if 'figsize' not in fig_kwargs or fig_kwargs['figsize'] is None:
        fig_kwargs['figsize'] = (10, 10)

    fig = plt.figure(**fig_kwargs)
    ax = fig.add_subplot(projection='3d')

    if show_tpcs:
        ax.voxels(X, Y, Z, filled, facecolors=colors, edgecolor='k', linewidth=0.3)

    if show_cryostat:
        cryo = geo.cryostat
        _plot_box_edges(ax, cryo.x_range, cryo.y_range, cryo.z_range,
                        color='#1a3a5c', linewidth=0.5, alpha=0.6)

    if show_opdets and geo.opdets:
        ox = [o.origin.x for o in geo.opdets]
        oy = [o.origin.y for o in geo.opdets]
        oz = [o.origin.z for o in geo.opdets]
        color_axis = det_cfg['opdet_color_axis']
        c_values = ox if color_axis == 'x' else oy
        ax.scatter(ox, oy, oz, marker='*', s=40, c=c_values, cmap='coolwarm',
                   depthshade=False, zorder=5)

    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_zlabel('z [cm]')
    ax.set_title(f'{geo.detector_name}\n{len(tpcs)} TPCs — colour = z-module index')

    x_lo, x_hi = x_edges[0], x_edges[-1]
    y_lo, y_hi = y_edges[0], y_edges[-1]
    z_lo, z_hi = z_edges[0], z_edges[-1]

    ann = {**_ANNOTATION_DEFAULTS, **(annotation_overrides or {})}
    _beam_x = ann['beam_anchor_x']
    _beam_y = ann['beam_anchor_y']
    _drift_z = ann['drift_anchor_z']

    drift_y = None
    if show_drift:
        drift_y = geo.cryostat.y_range.min + ann['drift_y_offset']
        y_lo = min(y_lo, drift_y)
    beam_z = None
    if show_beam:
        beam_z = ann['beam_tip_z'] - ann['beam_length']
        z_lo = min(z_lo, beam_z)

    ax.set_box_aspect((x_hi - x_lo, y_hi - y_lo, z_hi - z_lo))
    ax.view_init(elev=det_cfg['elev'], azim=det_cfg['azim'], roll=det_cfg['roll'])

    if show_cryostat_dimensions:
        cryo = geo.cryostat
        _plot_box_dimensions(ax, cryo.x_range, cryo.y_range, cryo.z_range,
                             color=ann['dim_color'])

    if show_beam:
        ax.quiver(_beam_x, _beam_y, beam_z, 0, 0, ann['beam_length'],
                  color=ann['beam_color'], linewidth=ann['beam_linewidth'],
                  arrow_length_ratio=ann['beam_arrow_length_ratio'])
        ax.text(_beam_x, _beam_y, beam_z, 'beam', color=ann['beam_color'],
                fontsize=ann['beam_fontsize'], ha='center', va='top')

    if show_drift:
        cryo = geo.cryostat
        x0, x1 = cryo.x_range.min, cryo.x_range.max
        dx = x1 - x0
        z_pos = _drift_z if _drift_z is not None else (cryo.z_range.min + cryo.z_range.max) / 2
        ax.quiver(x0, drift_y, z_pos, dx, 0, 0,
                  color=ann['drift_color'], linewidth=ann['drift_linewidth'],
                  arrow_length_ratio=ann['drift_arrow_length_ratio'])
        ax.text(x0, drift_y, z_pos, 'drift', color=ann['drift_color'],
                fontsize=ann['drift_fontsize'], ha='center', va='top')

    if not show_axes:
        ax.set_axis_off()

    if not show_grid:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        ax.grid(False)

    return fig, ax


def compare_cryostats_3d(
    geos: list[GeoData | FDVDGeometry],
    labels: list[str] | None = None,
    cmap_name: str = 'tab10',
    drift_type: str | None = None,
    elev: float = 30,
    azim: float = 45,
    roll: float = 0,
    show_cryostat_dimensions: bool = False,
    show_axes: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Overlay cryostat wireframes from multiple geometry objects for visual comparison.

    Parameters
    ----------
    geos:
        List of :class:`~tpvalidator.detgeometry.GeoData` or
        :class:`~tpvalidator.detgeometry.FDVDGeometry` instances.
    labels:
        Legend labels, one per geometry. Defaults to each object's ``detector_name``.
    cmap_name:
        Colormap used to assign a distinct colour to each cryostat.
    drift_type:
        ``'vd'`` or ``'hd'``. When set, overrides ``elev``, ``azim``, and ``roll``
        with the detector defaults from :data:`_DETECTOR_DEFAULTS`.
    elev, azim, roll:
        Initial camera angles in degrees. Ignored when ``drift_type`` is given.
    show_cryostat_dimensions:
        Annotate each cryostat with dimension lines in its matching colour.
    show_axes:
        When ``False``, hide axis lines, tick marks, tick labels, and grid planes.

    Returns
    -------
    fig, ax
    """
    resolved = [_resolve(g) for g in geos]
    if labels is None:
        labels = [g.detector_name for g in resolved]
    if len(labels) != len(resolved):
        raise ValueError('labels must have the same length as geos')
    if drift_type is not None:
        if drift_type not in _DETECTOR_DEFAULTS:
            raise ValueError(f"drift_type must be 'vd' or 'hd', got {drift_type!r}")
        det_cfg = _DETECTOR_DEFAULTS[drift_type]
        elev, azim, roll = det_cfg['elev'], det_cfg['azim'], det_cfg['roll']

    cmap = plt.colormaps[cmap_name]
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection='3d')

    all_x, all_y, all_z = [], [], []
    cryostats_colors = []
    for i, (geo, label) in enumerate(zip(resolved, labels)):
        cryo = geo.cryostat
        color = cmap(i / len(resolved))
        _plot_box_edges(ax, cryo.x_range, cryo.y_range, cryo.z_range,
                        color=color, linewidth=1.0, alpha=0.8, label=label)
        all_x += [cryo.x_range.min, cryo.x_range.max]
        all_y += [cryo.y_range.min, cryo.y_range.max]
        all_z += [cryo.z_range.min, cryo.z_range.max]
        cryostats_colors.append((cryo, color))

    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_zlabel('z [cm]')
    ax.set_title('Cryostat geometry comparison')

    x_span = max(all_x) - min(all_x)
    y_span = max(all_y) - min(all_y)
    z_span = max(all_z) - min(all_z)
    ax.set_box_aspect((x_span, y_span, z_span))
    ax.view_init(elev=elev, azim=azim, roll=roll)

    if show_cryostat_dimensions:
        for cryo, color in cryostats_colors:
            _plot_box_dimensions(ax, cryo.x_range, cryo.y_range, cryo.z_range,
                                 color=color)

    if not show_axes:
        ax.set_axis_off()

    return fig, ax


def plot_geometry_2d(
    geo: GeoData | FDVDGeometry,
    plane: Literal['xy', 'yz', 'xz'] = 'yz',
    cmap_name: str = 'tab20',
    show_tpcs: bool = True,
    show_cryostat: bool = True,
    show_opdets: bool = False,
    **fig_kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot 2-D projections of TPC volumes onto one of the three coordinate planes.

    Parameters
    ----------
    geo:
        A :class:`~tpvalidator.detgeometry.GeoData` or
        :class:`~tpvalidator.detgeometry.FDVDGeometry` instance.
    plane:
        Projection plane: ``'xy'``, ``'xz'``, or ``'yz'``.
    cmap_name:
        Colormap for colouring TPC rectangles by their depth-axis bin index.
    show_tpcs:
        Fill TPC rectangles (semi-transparent, edge-outlined).
    show_cryostat:
        Draw the cryostat boundary as a dashed rectangle.
    show_opdets:
        Scatter-plot optical detector positions projected onto the plane.

    Returns
    -------
    fig, ax
    """
    if plane not in _PLANE_AXES:
        raise ValueError(f"plane must be one of {list(_PLANE_AXES)}, got {plane!r}")
    h_ax, v_ax, c_ax = _PLANE_AXES[plane]
    geo = _resolve(geo)
    tpcs = geo.tpcs

    c_edges = _get_edges(tpcs, c_ax)
    n_c = len(c_edges) - 1
    cmap = plt.colormaps[cmap_name]

    if 'figsize' not in fig_kwargs or fig_kwargs['figsize'] is None:
        fig_kwargs['figsize'] = (8, 8)

    fig, ax = plt.subplots(**fig_kwargs)

    if show_tpcs:
        for tpc in tpcs:
            h_range: Range1D = getattr(tpc, f'{h_ax}_range')
            v_range: Range1D = getattr(tpc, f'{v_ax}_range')
            c_range: Range1D = getattr(tpc, f'{c_ax}_range')
            ic = _find_bin(c_edges, c_range.min)
            rgba = cmap(ic / max(n_c - 1, 1))
            ax.add_patch(Rectangle(
                (h_range.min, v_range.min), h_range.length, v_range.length,
                facecolor=(*rgba[:3], 0.25), edgecolor=(*rgba[:3], 0.9), linewidth=0.6,
            ))

            draw_tpc_id = False
            if draw_tpc_id:
                ax.text(h_range.center, v_range.center, f"{tpc.id}", horizontalalignment='center', verticalalignment='center')

    if show_cryostat:
        cryo = geo.cryostat
        h_range: Range1D = getattr(cryo, f'{h_ax}_range')
        v_range: Range1D = getattr(cryo, f'{v_ax}_range')
        ax.add_patch(Rectangle(
            (h_range.min, v_range.min), h_range.length, v_range.length,
            facecolor='none', edgecolor='#1a3a5c', linewidth=1.0, linestyle='--',
        ))

    if show_opdets and geo.opdets:
        oh = [getattr(o.origin, h_ax) for o in geo.opdets]
        ov = [getattr(o.origin, v_ax) for o in geo.opdets]
        ax.scatter(oh, ov, marker='*', s=30, c='orange', zorder=5, label='opdets')
        ax.legend(fontsize=8)

    ax.set_xlabel(f'{h_ax} [cm]')
    ax.set_ylabel(f'{v_ax} [cm]')
    # ax.set_title(f'{geo.detector_name} — {plane} projection  (colour = {c_ax}-module)')
    ax.set_aspect('equal')
    ax.autoscale()

    return fig, ax


def draw_tpc_outlines(
    ax: plt.Axes,
    geo: GeoData | FDVDGeometry,
    tpc_ids: list[int] | None = None,
    plane: str = 'xz',
    c: str | tuple | Sequence | None = None,
    colormap: str | mpl.colors.Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 0.8,
    linewidth: float = 1.2,
    fill: bool = False
) -> mpl.cm.ScalarMappable | None:
    """Draw the outlines of selected TPCs on an existing 2-D axes.

    Parameters
    ----------
    ax:
        A matplotlib axes object (e.g. from :func:`plot_geometry_2d`).
    geo:
        A :class:`~tpvalidator.detgeometry.GeoData` or
        :class:`~tpvalidator.detgeometry.FDVDGeometry` instance.
    tpc_ids:
        Indices into ``geo.tpcs`` identifying which TPCs to outline.
    plane:
        Projection plane: ``'xy'``, ``'xz'``, or ``'yz'``.
    c:
        Edge colour(s) or numeric values to map through *colormap*.
        When *colormap* is ``None``: a single matplotlib colour applies to all
        rectangles; a list of colours (one per entry in *tpc_ids*) colours each
        individually; ``None`` defaults to the current axes colour cycle.
        When *colormap* is given: *c* must be an array-like of numbers that are
        normalised and mapped to colours via the chosen colormap.
    colormap:
        Colormap name or object used when *c* is numeric.  When provided, *c*
        is treated as numeric values; the function returns a
        :class:`~matplotlib.cm.ScalarMappable` suitable for adding a colorbar.
    vmin, vmax:
        Explicit normalization range for the colormap.  Defaults to the min/max
        of *c* when not given.
    alpha:
        Edge opacity (0–1).
    linewidth:
        Edge line width in points.

    Returns
    -------
    ScalarMappable or None
        A :class:`~matplotlib.cm.ScalarMappable` when *colormap* is used
        (pass to ``fig.colorbar(sm, ax=ax)``), otherwise ``None``.
    """
    if plane not in _PLANE_AXES:
        raise ValueError(f"plane must be one of {list(_PLANE_AXES)}, got {plane!r}")
    h_ax, v_ax, _ = _PLANE_AXES[plane]
    geo = _resolve(geo)

    if tpc_ids is None:
        tpc_ids = [tpc.id for tpc in geo.tpcs]

    sm: mpl.cm.ScalarMappable | None = None
    colors: list

    if colormap is not None:
        if c is None:
            raise ValueError("c must be provided when colormap is given")
        c_vals = np.asarray(c, dtype=float)
        if len(c_vals) != len(tpc_ids):
            raise ValueError(
                f"c length ({len(c_vals)}) must match tpc_ids length ({len(tpc_ids)})"
            )
        cmap = mpl.colormaps[colormap] if isinstance(colormap, str) else colormap
        norm = mpl.colors.Normalize(
            vmin=vmin if vmin is not None else c_vals.min(),
            vmax=vmax if vmax is not None else c_vals.max(),
        )
        colors = [cmap(norm(v)) for v in c_vals]
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
    elif c is None:
        prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = [prop_cycle[i % len(prop_cycle)] for i in range(len(tpc_ids))]
    elif isinstance(c, list):
        if len(c) != len(tpc_ids):
            raise ValueError(
                f"c list length ({len(c)}) must match tpc_ids length ({len(tpc_ids)})"
            )
        colors = c
    else:
        colors = [c] * len(tpc_ids)

    for tpc_id, col in zip(tpc_ids, colors):
        tpc = geo.tpc_by_id(tpc_id)
        h_range: Range1D = getattr(tpc, f'{h_ax}_range')
        v_range: Range1D = getattr(tpc, f'{v_ax}_range')
        ax.add_patch(Rectangle(
            (h_range.min, v_range.min), h_range.length, v_range.length,
            facecolor=col, edgecolor=col, linewidth=linewidth, alpha=alpha, fill=fill,
        ))

        # ax.text(h_range.center, v_range.center, f"{tpc.id}", horizontalalignment='center', verticalalignment='center')


    return sm


def _rodrigues(v: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
    """Rotate vector v by angle theta (radians) around unit axis using Rodrigues' formula."""
    n = axis / np.linalg.norm(axis)
    return v * np.cos(theta) + np.cross(n, v) * np.sin(theta) + n * np.dot(n, v) * (1 - np.cos(theta))


def _camera_angles(d: np.ndarray, u: np.ndarray) -> tuple[float, float, float]:
    """Convert camera look-direction d and up-vector u to (elev, azim, roll) in degrees."""
    elev = np.degrees(np.arcsin(np.clip(d[2], -1.0, 1.0)))
    azim = np.degrees(np.arctan2(d[1], d[0]))
    e, a = np.radians(elev), np.radians(azim)
    up0 = np.array([-np.sin(e) * np.cos(a), -np.sin(e) * np.sin(a), np.cos(e)])
    right = np.cross(up0, d)
    if np.linalg.norm(right) < 1e-9:
        return elev, azim, 0.0
    right /= np.linalg.norm(right)
    up0 /= np.linalg.norm(up0)
    roll = np.degrees(np.arctan2(np.dot(u, right), np.dot(u, up0)))
    return elev, azim, roll


def animate_tpc_volumes(
    geo: GeoData | FDVDGeometry,
    output_path: str | Path = 'tpc_volumes.gif',
    rotation_axis: tuple[float, float, float] = (0, 0, 1),
    n_frames: int = 72,
    fps: int = 15,
    cmap_name: str = 'tab20',
    show_tpcs: bool = True,
    show_opdets: bool = True,
    drift_type: str = 'vd',
) -> None:
    """Rotate the TPC volume plot 360° around a physical axis and save as a GIF.

    Parameters
    ----------
    geo:
        A :class:`~tpvalidator.detgeometry.GeoData` or
        :class:`~tpvalidator.detgeometry.FDVDGeometry` instance.
    output_path:
        Destination path for the GIF file.
    rotation_axis:
        Physical (x, y, z) rotation axis vector. Defaults to ``(0, 0, 1)`` (world z).
    n_frames:
        Number of frames for a full 360° rotation (default 72 → 5° per frame).
    fps:
        Frames per second of the output GIF.
    cmap_name, show_tpcs, show_opdets, drift_type:
        Forwarded to :func:`plot_geometry`.
    """
    was_interactive = plt.isinteractive()
    plt.ioff()
    try:
        fig, ax = plot_geometry_3d(
            geo,
            cmap_name=cmap_name,
            show_tpcs=show_tpcs,
            show_opdets=show_opdets,
            drift_type=drift_type,
            figsize=(10, 10)
        )

        det_cfg = _DETECTOR_DEFAULTS[drift_type]
        elev0 = np.radians(det_cfg['elev'])
        azim0 = np.radians(det_cfg['azim'])

        d0 = np.array([np.cos(elev0) * np.cos(azim0),
                       np.cos(elev0) * np.sin(azim0),
                       np.sin(elev0)])
        u0 = np.array([-np.sin(elev0) * np.cos(azim0),
                       -np.sin(elev0) * np.sin(azim0),
                       np.cos(elev0)])

        roll0 = np.radians(det_cfg['roll'])
        if roll0 != 0.0:
            u0 = _rodrigues(u0, d0, -roll0)

        if isinstance(rotation_axis, str):
            aliases = {
                'azim': np.array([0.0, 0.0, 1.0]),
                'elev': np.cross(d0, u0),
                'roll': d0.copy(),
            }
            if rotation_axis not in aliases:
                raise ValueError(
                    f"rotation_axis string must be 'azim', 'elev', or 'roll'; got {rotation_axis!r}"
                )
            axis = aliases[rotation_axis]
        else:
            axis = np.asarray(rotation_axis, dtype=float)

        def _update(i):
            theta = i * 2 * np.pi / n_frames
            d_rot = _rodrigues(d0, axis, theta)
            u_rot = _rodrigues(u0, axis, theta)
            e, a, r = _camera_angles(d_rot, u_rot)
            ax.view_init(elev=e, azim=a, roll=r)

        print(f'Rendering {n_frames} frames…')
        anim = animation.FuncAnimation(fig, _update, frames=n_frames, interval=1000 / fps)
        anim.save(output_path, writer='pillow', fps=fps)
        print(f'Saved to {output_path}')
        plt.close(fig)
    finally:
        if was_interactive:
            plt.ion()
