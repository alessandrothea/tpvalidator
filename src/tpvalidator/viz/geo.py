import json
from pathlib import Path

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np


def _get_edges(tpcs: list[dict], axis: str) -> np.ndarray:
    """Collect unique bin edges for one axis, merging floating-point near-duplicates."""
    vals = set()
    for t in tpcs:
        vals.add(round(t[f'{axis}_range']['min'], 3))
        vals.add(round(t[f'{axis}_range']['max'], 3))
    return np.array(sorted(vals))


def _find_bin(edges: np.ndarray, val: float) -> int:
    return int(np.searchsorted(edges, round(val, 3), side='left'))


_DETECTOR_DEFAULTS = {
    'vd': dict(elev=30,  azim=135, roll=120, opdet_color_axis='x'),
    'hd': dict(elev=50,  azim=35,  roll=120, opdet_color_axis='y'),
}


def plot_tpc_volumes(
    geo_file: str | Path,
    cmap_name: str = 'tab20',
    show_tpcs: bool = True,
    show_opdets: bool = True,
    drift_type: str = 'vd',
) -> tuple[plt.Figure, plt.Axes]:
    """Plot TPC volumes and/or optical detector positions from a geometry JSON file.

    Parameters
    ----------
    geo_file:
        Path to the geometry JSON (must contain a ``tpcs`` list).
    cmap_name:
        Matplotlib colormap used to colour TPC voxels by z-module index.
    show_tpcs:
        Draw TPC volumes as voxels.
    show_opdets:
        Draw optical detector centres as scatter markers (requires ``opdets`` in the JSON).
    drift_type:
        ``'vd'`` or ``'hd'``. Controls the initial view angles and the axis used
        to colour opdet markers (x for VD, y for HD).

    Returns
    -------
    fig, ax
    """
    if drift_type not in _DETECTOR_DEFAULTS:
        raise ValueError(f"drift_type must be 'vd' or 'hd', got {drift_type!r}")
    det_cfg = _DETECTOR_DEFAULTS[drift_type]
    with open(geo_file) as f:
        geo = json.load(f)
    tpcs = geo['tpcs']

    x_edges = _get_edges(tpcs, 'x')
    y_edges = _get_edges(tpcs, 'y')
    z_edges = _get_edges(tpcs, 'z')

    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    nz = len(z_edges) - 1

    filled = np.zeros((nx, ny, nz), dtype=bool)
    for tpc in tpcs:
        ix = _find_bin(x_edges, tpc['x_range']['min'])
        iy = _find_bin(y_edges, tpc['y_range']['min'])
        iz = _find_bin(z_edges, tpc['z_range']['min'])
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

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection='3d')

    if show_tpcs:
        ax.voxels(X, Y, Z, filled, facecolors=colors, edgecolor='k', linewidth=0.3)

    if show_opdets and 'opdets' in geo:
        opdets = geo['opdets']
        ox = [o['origin']['x'] for o in opdets]
        oy = [o['origin']['y'] for o in opdets]
        oz = [o['origin']['z'] for o in opdets]
        color_axis = det_cfg['opdet_color_axis']
        c_values = ox if color_axis == 'x' else oy
        sc = ax.scatter(ox, oy, oz, marker='*', s=40, c=c_values, cmap='coolwarm',
                        depthshade=False, zorder=5)
        plt.colorbar(sc, ax=ax, label=f'{color_axis} [cm]', shrink=0.5, pad=0.1)

    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_zlabel('z [cm]')
    detector_name = geo.get('detector_name', geo_file.stem if isinstance(geo_file, Path) else Path(geo_file).stem)
    ax.set_title(f'{detector_name}\n{len(tpcs)} TPCs — colour = z-module index')

    x_span = x_edges[-1] - x_edges[0]
    y_span = y_edges[-1] - y_edges[0]
    z_span = z_edges[-1] - z_edges[0]
    ax.set_box_aspect((x_span, y_span, z_span))
    ax.view_init(elev=det_cfg['elev'], azim=det_cfg['azim'], roll=det_cfg['roll'])

    return fig, ax


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
    geo_file: str | Path,
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
    geo_file:
        Path to the geometry JSON.
    output_path:
        Destination path for the GIF file.
    rotation_axis:
        Physical (x, y, z) rotation axis vector. Defaults to ``(0, 0, 1)`` (world z),
        which is equivalent to sweeping the azimuth.
    n_frames:
        Number of frames for a full 360° rotation (default 72 → 5° per frame).
    fps:
        Frames per second of the output GIF.
    cmap_name, show_tpcs, show_opdets, drift_type:
        Forwarded to :func:`plot_tpc_volumes`.
    """
    was_interactive = plt.isinteractive()
    plt.ioff()
    try:
        fig, ax = plot_tpc_volumes(
            geo_file,
            cmap_name=cmap_name,
            show_tpcs=show_tpcs,
            show_opdets=show_opdets,
            drift_type=drift_type,
        )

        det_cfg = _DETECTOR_DEFAULTS[drift_type]
        elev0 = np.radians(det_cfg['elev'])
        azim0 = np.radians(det_cfg['azim'])

        # Initial camera look-direction and up-vector in world coordinates
        d0 = np.array([np.cos(elev0) * np.cos(azim0),
                       np.cos(elev0) * np.sin(azim0),
                       np.sin(elev0)])
        u0 = np.array([-np.sin(elev0) * np.cos(azim0),
                       -np.sin(elev0) * np.sin(azim0),
                       np.cos(elev0)])

        # Apply the initial roll so frame 0 matches the view set by plot_tpc_volumes
        roll0 = np.radians(det_cfg['roll'])
        if roll0 != 0.0:
            u0 = _rodrigues(u0, d0, -roll0)

        # ── resolve rotation axis ───────────────────────────────────────
        if isinstance(rotation_axis, str):
            aliases = {
                'azim': np.array([0.0, 0.0, 1.0]),        # world z → sweeps azimuth
                'elev': np.cross(d0, u0),                  # camera right → sweeps elevation
                'roll': d0.copy(),                          # look direction → changes roll
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
