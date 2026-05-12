
import numpy as np
import matplotlib.pyplot as plt
import hist
import particle
import mplhep as hep

from rich.table import Table
from typing import Literal, Optional, Tuple
from matplotlib.colors import LogNorm

from ..workspace import TriggerPrimitivesWorkspace


class MCPlotter:
    """Plots and tables for a single MC collection (mctruths or mcparticles).

    Parameters
    ----------
    ws:
        The workspace providing access to MC data.
    collection:
        Which collection this instance operates on: ``'mctruths'`` (default)
        or ``'mcparticles'``.  All ``self.df`` references throughout the class
        resolve to the chosen collection.  Methods that are inherently
        truth-level (``simulated_mc_time``, ``make_generators_table``,
        ``make_generator_rates_table``) always use ``ws.mctruths`` regardless
        of this setting.
    """

    def __init__(
        self,
        ws: TriggerPrimitivesWorkspace,
        collection: Literal['mctruths', 'mcparticles'] = 'mctruths',
    ):
        """Initialise the plotter.

        Parameters
        ----------
        ws:
            Workspace instance.
        collection:
            ``'mctruths'`` or ``'mcparticles'``.
        """
        self.ws = ws
        self.collection = collection
        self.df = getattr(ws, collection)
        
        # Private variables
        self._groups_by_generator = None

    # ── Collection helpers ────────────────────────────────────────────────────

    @property
    def _generator_by(self) -> str:
        """Column name used to group the active collection by generator.

        Returns ``'generator_name'`` for mctruths (the column is present
        directly) and ``'truth_block_id'`` for mcparticles (resolved via
        ``ws.mctruth_blocks_map`` inside :meth:`_make_groups`).
        """
        return 'generator_name' if self.collection == 'mctruths' else 'truth_block_id'
        

    @property
    def groups_by_generator(self) -> dict:
        if self._groups_by_generator is None:
            self._groups_by_generator = self._make_groups(self.df, self._generator_by)
        
        return self._groups_by_generator


    def _make_groups(self, df, by: str,
                        n_top: Optional[int] = None) -> dict[str, object]:
        """Return a ``{label: sub-DataFrame}`` dict grouped by *by*.

        Parameters
        ----------
        by:
            Grouping mode:

            * ``'pdg'`` — group by PDG id, label = PDG name from the
              ``particle`` package.
            * ``'generator_name'`` — group by MCTruth generator name directly.
            * ``'truth_block_id'`` — group MCParticles by truth block id,
              label resolved via ``ws.mctruth_blocks_map``.
        n_top:
            If given, return only the *n_top* largest groups by row count.
            ``None`` keeps all groups.
        """
        match by:
            case 'pdg':
                groups = {particle.Particle.from_pdgid(p).pdg_name: sub
                          for p, sub in df.groupby('pdg')}
            case 'generator_name':
                groups = {name: sub for name, sub in df.groupby('generator_name')}
            case 'truth_block_id':
                block_map = self.ws.mctruth_blocks_map
                groups = {block_map[n]: sub for n, sub in df.groupby('truth_block_id')}
            case _:
                raise ValueError(f"Unknown groupby: {by!r}")
        groups = dict(sorted(groups.items(), key=lambda x: len(x[1]), reverse=True))
        if n_top is not None:
            groups = dict(list(groups.items())[:n_top])
        return groups
    

    # ── Bookkeeping ───────────────────────────────────────────────────────────

    def simulated_mc_time(self) -> float:
        """Return total simulated time in seconds.

        Computed as ``2 × readout_window × num_entries × 0.5 µs``, where the
        factor of 2 accounts for pre- and post-spill readout windows.
        Always derived from ``ws.mctruths`` regardless of the active
        collection.
        """
        
        sampling_time = 0.5e-6  # Sampling time 1/2 usec
        ro_win = self.df.extra_info['readout_window']
        num_entries = self.df.extra_info['num_entries']
        return 2 * ro_win * sampling_time * num_entries

    def make_generators_table(self) -> Table:
        """Return a rich Table listing background generator names and their block ids."""
        t = Table('generator name', 'id', title='Background generators')
        for gid, name in self.ws.mctruth_blocks_map.items():
            t.add_row(name, str(gid))
        return t

    def make_generator_rates_table(self) -> Table:
        """Return a rich Table of generator names, entry counts, and activity rates in Hz.

        Always derived from ``ws.mctruths`` regardless of the active
        collection.
        """
        simu_time = self.simulated_mc_time()

        h_counts = self.make_generator_counts_hist()


        t = Table('generator name', 'entries', 'rate [Hz]', title='Backgrounds generators by activity')
        for bin_label, count in zip(h_counts.axes[0], h_counts.values()):
            t.add_row(bin_label, f'{int(count)}', f'{count / simu_time :.2e}')

        return t


    # ── Generator activity ────────────────────────────────────────────────────

    def make_generator_counts_hist(self, query: Optional[str] = None) -> hist.Hist:
        """Return a 1D StrCategory hist of generator counts for the active collection.

        For ``mctruths`` the groups are keyed by ``generator_name``.  For
        ``mcparticles`` they are keyed by ``truth_block_id`` resolved via
        ``ws.mctruth_blocks_map``.

        Parameters
        ----------
        query:
            Optional pandas query string applied to the DataFrame before
            filling (e.g. ``'pdg == 11'`` to select electrons only).
        """
        df = self.df
        if query:
            df = df.query(query)
        groups = self._make_groups(df, self._generator_by)
        categories = list(groups.keys())
        h = hist.Hist(hist.axis.StrCategory(categories, name='generator', label='Generator'))
        for label, sub in groups.items():
            h.fill(generator=np.full(len(sub), label))
        return h

    def plot_generator_activity(self, norm: str = 'counts', pdg_id: Optional[int] = None,
                                figsize: tuple = (10, 10)):
        """Plot generator activity as a filled bar histogram.

        Parameters
        ----------
        norm:
            ``'counts'`` for raw event counts or ``'rate'`` for activity in Hz.
            Rates are computed using :meth:`simulated_mc_time` (truth-level).
        pdg_id:
            If given, restrict to entries with this PDG id.
        figsize:
            Figure size passed to ``plt.subplots``.
        """
        match norm:
            case 'counts':
                ylabel = 'counts'
                title_units = 'counts'
                norm_factor = 1
            case 'rate':
                ylabel = 'Rates [Hz]'
                title_units = 'rates'
                norm_factor = 1 / self.simulated_mc_time()
            case _:
                raise RuntimeError(f"norm ({norm!r}) must be 'counts' or 'rate'")

        query = f'pdg == {pdg_id}' if pdg_id is not None else None
        h = self.make_generator_counts_hist(query)
        h *= norm_factor

        fig, ax = plt.subplots(figsize=figsize)
        hep.histplot(h, yerr=False, histtype='fill', ax=ax)
        ax.grid(visible=True)
        ax.tick_params(axis='x', rotation=90)
        ax.set_ylabel(ylabel)
        ax.set_yscale('log')

        title = f'Radioactive backgrounds {title_units}'
        if pdg_id is not None:
            pp = particle.Particle.from_pdgid(pdg_id)
            title += f" producing ${pp.latex_name}$"

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    # ── KE spectra — private helpers ──────────────────────────────────────────

    def _build_ke_hist(self, groups: dict[str, object], n_bins: int,
                       max_ke: float) -> hist.Hist:
        """Build a 2D ``hist.Hist`` with axes ``(StrCategory[group] × Regular[ke])``.

        Parameters
        ----------
        groups:
            Output of :meth:`_make_groups`.
        n_bins:
            Number of KE bins.
        max_ke:
            Upper edge of the KE axis in MeV. Passed by the caller so that
            paired histograms (mct / mcp) always share the same binning.
        """
        ke_axis = hist.axis.Regular(n_bins, 0, max_ke, name='ke', label=r'$E_{kin}$ [MeV]')
        labels = list(groups.keys())
        h = hist.Hist(hist.axis.StrCategory(labels, name='group'), ke_axis)
        for label, sub in groups.items():
            h.fill(group=label, ke=sub.kinetic_energy.values * 1000)
        return h

    def _build_pos_hist(self, pos_var: Literal['x', 'y', 'z'], groups: dict[str, object],
                        n_bins: int, p_min: float, p_max: float) -> hist.Hist:
        """Build a 2D ``hist.Hist`` with axes ``(StrCategory[group] × Regular[pos])``.

        Parameters
        ----------
        pos_var:
            Name of the position column (``'x'``, ``'y'``, or ``'z'``).
        groups:
            Output of :meth:`_make_groups`.
        n_bins:
            Number of position bins.
        p_min:
            Lower edge of the position axis in cm.
        p_max:
            Upper edge of the position axis in cm.
        """
        pos_axis = hist.axis.Regular(n_bins, p_min, p_max, name=pos_var, label=f'{pos_var} [cm]')
        labels = list(groups.keys())
        h = hist.Hist(hist.axis.StrCategory(labels, name='group'), pos_axis)
        for label, sub in groups.items():
            h.fill(group=label, **{pos_var: sub[pos_var].values})
        return h

    def _build_pos2d_hist(self, df, x_axis_id: str, y_axis_id: str,
                          bin_width: int,
                          x_rng: Optional[tuple[float, float]] = None,
                          y_rng: Optional[tuple[float, float]] = None) -> hist.Hist:
        """Build a 2D ``hist.Hist`` of position in the plane ``(x_axis_id, y_axis_id)``.

        Parameters
        ----------
        df:
            DataFrame containing the position columns.
        x_axis_id:
            Column name for the horizontal axis (e.g. ``'x'``).
        y_axis_id:
            Column name for the vertical axis (e.g. ``'y'``).
        bin_width:
            Width of each bin in cm; used to compute bin counts from the range.
        x_rng:
            ``(min, max)`` for the x axis in cm.  Defaults to the column
            extremes of *df*.
        y_rng:
            ``(min, max)`` for the y axis in cm.  Defaults to the column
            extremes of *df*.

        Returns
        -------
        hist.Hist
            Filled 2D histogram.
        """
        if x_rng is None:
            x_rng = df[x_axis_id].min(), df[x_axis_id].max()
        if y_rng is None:
            y_rng = df[y_axis_id].min(), df[y_axis_id].max()

        x_n_bin = int((x_rng[1] - x_rng[0]) // bin_width)
        y_n_bin = int((y_rng[1] - y_rng[0]) // bin_width)

        x_axis = hist.axis.Regular(x_n_bin, *x_rng, name=f"{x_axis_id} [cm]")
        y_axis = hist.axis.Regular(y_n_bin, *y_rng, name=f"{y_axis_id} [cm]")
        h = hist.Hist(x_axis, y_axis)
        h.fill(df[x_axis_id], df[y_axis_id])
        return h

    def _plot_ke_panel(self, ax, h: hist.Hist, title: str,
                       color_map: Optional[dict] = None):
        """Render all group slices of a 2D KE hist as overlaid step histograms."""
        for label in h.axes['group']:
            kw = dict(label=label, histtype='step', yerr=False, ax=ax)
            if color_map:
                kw['color'] = color_map.get(label)
            hep.histplot(h[label, :], **kw)
        ax.legend()
        ax.set_yscale('log')
        ax.set_title(title)
        ax.grid(visible=True)

    # ── KE spectra — histograms ───────────────────────────────────────────────

    def make_ke_spectra_by_pdg_hist(self, n_bins: int = 100,
                                    n_top: Optional[int] = None) -> hist.Hist:
        """Return a 2D KE histogram with groups keyed by PDG name.

        Parameters
        ----------
        n_bins:
            Number of KE bins.
        n_top:
            If given, keep only the *n_top* most populous PDG species.
        """
        max_ke = self.df.kinetic_energy.max() * 1000
        return self._build_ke_hist(self._make_groups(self.df, 'pdg', n_top), n_bins, max_ke)

    def make_ke_spectra_by_generator_hist(self, n_top: Optional[int] = None,
                                          n_bins: int = 100) -> hist.Hist:
        """Return a 2D KE histogram with groups keyed by generator name.

        For ``mctruths`` groups by ``generator_name``; for ``mcparticles``
        groups by ``truth_block_id`` resolved via ``ws.mctruth_blocks_map``.

        Parameters
        ----------
        n_top:
            If given, keep only the *n_top* most active generators.
        n_bins:
            Number of KE bins.
        """
        max_ke = self.df.kinetic_energy.max() * 1000
        return self._build_ke_hist(
            self._make_groups(self.df, self._generator_by, n_top), n_bins, max_ke
        )

    # ── KE spectra — plots ────────────────────────────────────────────────────

    def plot_ke_spectra_by_pdg(self, n_bins: int = 100, n_top: Optional[int] = None,
                               figsize: tuple = (8, 6)):
        """Plot KE spectra of the active collection grouped by PDG id.

        Parameters
        ----------
        n_bins:
            Number of KE bins.
        n_top:
            If given, show only the *n_top* most populous PDG species.
        figsize:
            Figure size passed to ``plt.subplots``.
        """
        h = self.make_ke_spectra_by_pdg_hist(n_bins, n_top)
        fig, ax = plt.subplots(figsize=figsize)
        self._plot_ke_panel(ax, h, f'{self.collection} — KE by PDG')
        fig.tight_layout()
        return fig

    def plot_ke_spectra_by_generator(self, n_top: Optional[int] = 10, n_bins: int = 100,
                                     figsize: tuple = (8, 6)):
        """Plot KE spectra of the active collection grouped by generator name.

        Parameters
        ----------
        n_top:
            If given, show only the *n_top* most active generators.
        n_bins:
            Number of KE bins.
        figsize:
            Figure size passed to ``plt.subplots``.
        """
        h = self.make_ke_spectra_by_generator_hist(n_top, n_bins)
        fig, ax = plt.subplots(figsize=figsize)
        self._plot_ke_panel(ax, h, f'{self.collection} — KE by generator')
        fig.tight_layout()
        return fig

    # ── Origin — plots ────────────────────────────────────────────────────

    def plot_generator_pos(self, axis: Literal['x', 'y', 'z'], n_top: Optional[int] = 10,
                                 c_scale: Literal['lin', 'log'] = 'lin', figsize: tuple = (8, 8)):
        """Plot a generator × position 2D histogram for the active collection.

        Parameters
        ----------
        axis:
            Position coordinate to plot on the horizontal axis (``'x'``,
            ``'y'``, or ``'z'``).
        n_top:
            Keep only the *n_top* most populous generators.
        c_scale:
            Colour scale: ``'lin'`` for linear, ``'log'`` for logarithmic.
        figsize:
            Figure size passed to ``plt.subplots``.
        """
        df = self.df
        groups = self._make_groups(df, self._generator_by, n_top=n_top)

        v_min = df[axis].min()
        v_max = df[axis].max()
        bin_width = 50  # cm
        n_bins_v = int((v_max - v_min) // bin_width)
        hist_v = self._build_pos_hist(axis, groups, n_bins_v, v_min, v_max)

        fig, ax = plt.subplots(figsize=figsize)
        match c_scale:
            case "log":
                c_norm = LogNorm(vmin=1, vmax=hist_v.values().max())
            case "lin":
                c_norm = None
            case _:
                raise RuntimeError(f"Unexpected c_scale value {c_scale!r} ('lin', 'log')")
        hep.hist2dplot(hist_v, ax=ax, norm=c_norm)
        ax.tick_params(axis='x', rotation=90)
        fig.tight_layout()
        return fig

    def plot_generator_origin_2d(self, name: str, plane: Literal['xy', 'yz', 'xz'],
                                       bin_width: int = 50, c_scale: str = 'lin',
                                       figsize: tuple = (8, 8)):
        """Plot the 2D spatial origin of a single generator in the active collection.

        Parameters
        ----------
        name:
            Generator name (must be a value in ``ws.mctruth_blocks_map``).
        plane:
            Projection plane: ``'xy'``, ``'yz'``, or ``'xz'``.
        bin_width:
            Bin width in cm.
        c_scale:
            Colour scale: ``'lin'`` for linear, ``'log'`` for logarithmic.
        figsize:
            Figure size passed to ``plt.subplots``.
        """
        if name not in self.ws.mctruth_blocks_map.values():
            raise RuntimeError(f"Generator {name} not found.")

        match plane:
            case 'xy':
                x_axis_id, y_axis_id = 'x', 'y'
            case 'yz':
                x_axis_id, y_axis_id = 'y', 'z'
            case 'xz':
                x_axis_id, y_axis_id = 'x', 'z'
            case _:
                raise ValueError(f"Plane can be 'xy', 'yz', 'xz'. Found {plane!r}")

        groups = self._make_groups(self.df, self._generator_by)
        df = groups[name]
        h = self._build_pos2d_hist(df, x_axis_id, y_axis_id, bin_width)

        fig, ax = plt.subplots(figsize=figsize)
        match c_scale:
            case "log":
                norm = LogNorm(vmin=1, vmax=h.values().max())
            case "lin":
                norm = None
            case _:
                raise RuntimeError(f"Unexpected c_scale value {c_scale!r} ('lin', 'log')")
        hep.hist2dplot(h, ax=ax, norm=norm)
        fig.tight_layout()
        return fig

    def plot_generator_pos_ke(self, name: str, figsize: tuple = (8, 8)):
        """Plot x, y, z position distributions and KE spectrum for a single generator.

        Produces a 2×2 grid: x (top-left), y (top-right), z (bottom-left),
        KE on a log scale (bottom-right).

        Parameters
        ----------
        name:
            Generator name (must be a value in ``ws.mctruth_blocks_map``).
        figsize:
            Figure size passed to ``plt.subplots``.
        """
        if name not in self.ws.mctruth_blocks_map.values():
            raise RuntimeError(f"Generator {name} not found.")

        df = self.df
        groups = self._make_groups(df, self._generator_by)

        h_pos = {}
        for v in ['x', 'y', 'z']:
            v_min = df[v].min()
            v_max = df[v].max()
            n_bins_v = int((v_max - v_min) // 20)
            h_pos[v] = self._build_pos_hist(v, groups, n_bins_v, v_min, v_max)

        max_ke = df.kinetic_energy.max() * 1000
        h_ke = self._build_ke_hist(groups, 100, max_ke)

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        ax = axes[0][0]
        hep.histplot(h_pos['x'][name, :], histtype='fill', ax=ax)
        ax.set_ylabel('counts')
        ax.grid(visible=True)

        ax = axes[0][1]
        hep.histplot(h_pos['y'][name, :], histtype='fill', ax=ax)
        ax.set_ylabel('counts')
        ax.grid(visible=True)

        ax = axes[1][0]
        hep.histplot(h_pos['z'][name, :], histtype='fill', ax=ax)
        ax.set_ylabel('counts')
        ax.grid(visible=True)

        ax = axes[1][1]
        hep.histplot(h_ke[name, :], histtype='fill', ax=ax)
        ax.set_ylabel('counts')
        ax.set_yscale('log')
        ax.grid(visible=True)

        fig.tight_layout()
        return fig

    # ── Distributions ─────────────────────────────────────────────────────────

    def plot_distributions(self, bins: int = 100, figsize: tuple = (14, 14)):
        """Plot histograms of all numeric columns in the active collection.

        Parameters
        ----------
        bins:
            Number of bins for each histogram.
        figsize:
            Figure size passed to ``plt.subplots``.
        """
        fig, ax = plt.subplots(figsize=figsize)
        self.df.hist(ax=ax, bins=bins)
        fig.suptitle(f"{self.collection} distributions")
        fig.tight_layout()
        return fig
