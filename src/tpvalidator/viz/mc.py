
import numpy as np
import matplotlib.pyplot as plt
import hist
import particle
import mplhep as hep

from rich.table import Table
from typing import Optional, Tuple
from matplotlib.colors import LogNorm

from ..workspace import TriggerPrimitivesWorkspace


class MCPlotter:

    def __init__(self, ws: TriggerPrimitivesWorkspace):
        self.ws = ws
        # self.groups = {}


    # ── Bookkeeping ───────────────────────────────────────────────────────────

    def simulated_mc_time(self) -> float:
        """Return total simulated time in seconds.

        Computed as ``2 × readout_window × num_entries × 0.5 µs``, where the
        factor of 2 accounts for pre- and post-spill readout windows.
        """
        ro_win = self.ws.mctruths.extra_info['readout_window']
        num_entries = self.ws.mctruths.extra_info['num_entries']
        return 2 * ro_win * 0.5e-6 * num_entries

    def make_generators_table(self) -> Table:
        """Return a rich Table listing background generator names and their block ids."""
        t = Table('generator name', 'id', title='Background generators')
        for gid, name in self.ws.mctruth_blocks_map.items():
            t.add_row(name, str(gid))
        return t

    def make_generator_rates_table(self) -> Table:
        """Return a rich Table of generator names, entry counts, and activity rates in Hz."""
        simu_time = self.simulated_mc_time()
        t = Table('generator name', 'entries', 'rate [Hz]', title='Backgrounds generators by activity')
        part_by_gen = sorted([(n, _df) for n, _df in self.ws.mctruths.groupby('generator_name')],
                             reverse=True, key=lambda x: len(x[1]))
        for _gen_id, _df in part_by_gen:
            t.add_row(_gen_id, str(len(_df)), f'{len(_df) / simu_time :.2e}')
        return t

    # ── Generator activity ────────────────────────────────────────────────────

    def make_generator_counts_hist(self, query: Optional[str] = None) -> hist.Hist:
        """Return a 1D StrCategory hist of MCTruth generator counts.

        Parameters
        ----------
        query:
            Optional pandas query string applied to mctruths before filling
            (e.g. ``'pdg == 11'`` to select electrons only).
        """
        df = self.ws.mctruths
        if query:
            df = df.query(query)
        categories = sorted(df.generator_name.unique().tolist())
        h = hist.Hist(hist.axis.StrCategory(categories, name='generator', label='Generator'))
        h.fill(generator=df.generator_name.values)
        return h

    def plot_generator_activity(self, norm: str = 'counts', pdg_id: Optional[int] = None,
                                figsize: tuple = (10, 10)):
        """Plot MCTruth generator activity as a filled bar histogram.

        Parameters
        ----------
        norm:
            ``'counts'`` for raw event counts or ``'rate'`` for activity in Hz.
        pdg_id:
            If given, restrict to MCTruths with this PDG id.
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


    # TODO: static/class method?
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
    
    # TODO: static/class method?
    def _build_pos_hist(self, pos_var:str, groups: dict[str, object], n_bins: int, p_min: float, p_max:float):
        """Build a 2D ``hist.Hist`` with axes ``(StrCategory[group] × Regular[pos])``.

        Args:
            pos_var (str): _description_
            groups (dict[str, object]): _description_
            n_bins (int): _description_
            p_min (float): _description_
            p_max (float): _description_

        Returns:
            _type_: _description_
        """
        pos_axis = hist.axis.Regular(n_bins, p_min, p_max, name=pos_var, label=f'{pos_var} [cm]')
        labels = list(groups.keys())
        h = hist.Hist(hist.axis.StrCategory(labels, name='group'), pos_axis)
        for label, sub in groups.items():
            h.fill(group=label, **{pos_var:sub[pos_var].values})
        return h

    # TODO: tuple -> Tuple
    # TODO: static/class method?
    def _build_pos2d_hist(self, df,x_axis_id:str, y_axis_id:str, bin_width:int, x_rng:Optional[tuple[float, float]]=None, y_rng:Optional[tuple[float, float]]=None):
        if x_rng is None:
            x_rng = df[x_axis_id].min(), df[x_axis_id].max()
        if y_rng is None:
            y_rng = df[y_axis_id].min(), df[y_axis_id].max()

        x_n_bin = int((x_rng[1]-x_rng[0])//bin_width)
        y_n_bin = int((y_rng[1]-y_rng[0])//bin_width)

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

    def make_ke_spectra_by_pdg_hists(self, n_bins: int = 100,
                                      n_top: Optional[int] = None) -> tuple[hist.Hist, hist.Hist]:
        """Return ``(h_mct, h_mcp)`` 2D KE histograms with groups keyed by PDG name.

        Both histograms share the same KE axis, with upper edge set to the
        maximum kinetic energy across mct and mcp.

        Parameters
        ----------
        n_bins:
            Number of KE bins.
        n_top:
            If given, keep only the *n_top* most populous PDG species.
        """
        mct, mcp = self.ws.mctruths, self.ws.mcparticles
        max_ke = max(mct.kinetic_energy.max(), mcp.kinetic_energy.max()) * 1000
        return (
            self._build_ke_hist(self._make_groups(mct, 'pdg', n_top), n_bins, max_ke),
            self._build_ke_hist(self._make_groups(mcp, 'pdg', n_top), n_bins, max_ke),
        )

    def make_ke_spectra_by_generator_hists(self, n_top: Optional[int] = None,
                                            n_bins: int = 100) -> tuple[hist.Hist, hist.Hist]:
        """Return ``(h_mct, h_mcp)`` 2D KE histograms with groups keyed by generator name.

        MCTruths are grouped by ``generator_name``; MCParticles by
        ``truth_block_id`` resolved via ``ws.mctruth_blocks_map``.
        Both histograms share the same KE axis.

        Parameters
        ----------
        n_top:
            If given, keep only the *n_top* most active generators.
        n_bins:
            Number of KE bins.
        """
        mct, mcp = self.ws.mctruths, self.ws.mcparticles
        max_ke = max(mct.kinetic_energy.max(), mcp.kinetic_energy.max()) * 1000
        return (
            self._build_ke_hist(self._make_groups(mct, 'generator_name', n_top), n_bins, max_ke),
            self._build_ke_hist(self._make_groups(mcp, 'truth_block_id', n_top), n_bins, max_ke),
        )

    # ── KE spectra — plots ────────────────────────────────────────────────────

    def plot_ke_spectra_by_pdg(self, n_bins: int = 100, n_top: Optional[int] = None,
                               figsize: tuple = (14, 6)):
        """Plot KE spectra of MCTruths and MCParticles grouped by PDG id.

        PDG species are assigned consistent colours across both panels.
        """
        h_mct, h_mcp = self.make_ke_spectra_by_pdg_hists(n_bins, n_top)

        all_labels = sorted(set(h_mct.axes['group']) | set(h_mcp.axes['group']))
        colors = plt.cm.tab10.colors
        color_map = {label: colors[i % len(colors)] for i, label in enumerate(all_labels)}

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        self._plot_ke_panel(axes[0], h_mct, 'MCTruths particles', color_map)
        self._plot_ke_panel(axes[1], h_mcp, 'MCParticles (G4)', color_map)
        fig.tight_layout()
        return fig


    def plot_ke_spectra_by_generator(self, n_top: Optional[int] = 10, n_bins: int = 100,
                                     figsize: tuple = (14, 6)):
        """Plot KE spectra of MCTruths and MCParticles grouped by generator name."""
        h_mct, h_mcp = self.make_ke_spectra_by_generator_hists(n_top, n_bins)

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        self._plot_ke_panel(axes[0], h_mct, 'MCTruths particles')
        self._plot_ke_panel(axes[1], h_mcp, 'MCParticles (G4)')
        fig.tight_layout()
        return fig
    
    # ── Origin — plots ────────────────────────────────────────────────────

    def plot_truth_generator_pos(self, axis:str, n_top: Optional[int] = 10, c_scale: str='lin', figsize: tuple = (8,8)):

        mct = self.ws.mctruths
        mct_g = self._make_groups(mct, 'generator_name', n_top=n_top)

        v = axis
        v_min = mct[v].min()
        v_max = mct[v].max()

        bin_width=50 #cm
        n_binx_v = int((v_max-v_min)//bin_width)
        hist_v = self._build_pos_hist(v, mct_g, n_binx_v, v_min, v_max) 

        fig, ax = plt.subplots(figsize=figsize)
        match c_scale:
            case "log":
                norm = LogNorm(vmin=1, vmax=hist_v.values().max())
            case "lin":
                norm = None
            case _:
                raise RuntimeError(f"Unexpected norm value '{norm}' ('lin', 'log')")
        hep.hist2dplot(hist_v, ax=ax, norm=norm)
        ax.tick_params(axis='x', rotation=90)

        fig.tight_layout()
        return fig
    

    def plot_truth_generator_origin_2d(self, name:str, plane:str, bin_width=50, c_scale: str='lin', figsize: tuple = (8,8)):
        if name not in self.ws.mctruth_blocks_map.values():
            raise RuntimeError(f"Generator {name} not found.")

        mct = self.ws.mctruths

        # NOTE ---- make this into a build_pos_hist2d
        match plane:
            case 'xy':
                x_axis_id = 'x'
                y_axis_id = 'y'

            case 'yz':
                x_axis_id = 'y'
                y_axis_id = 'z'
            case 'xz':
                x_axis_id = 'x'
                y_axis_id = 'z'
            case _:
                raise ValueError(f"Plane can be 'xy', 'yz', 'xz'. Found {plane}")




        # NOTE: could use the samle xmin/xmax here?
        # x_axis_min = mct[x_axis_id].min()
        # x_axis_max = mct[x_axis_id].max()
        # y_axis_min = mct[y_axis_id].min()
        # y_axis_max = mct[y_axis_id].max()
        # x_n_bin = int((x_axis_max-x_axis_min)//bin_width)
        # y_n_bin = int((y_axis_max-y_axis_min)//bin_width)

        # x_axis = hist.axis.Regular(x_n_bin, x_axis_min, x_axis_max, name=f"{x_axis_id} [cm]")
        # y_axis = hist.axis.Regular(y_n_bin, y_axis_min, y_axis_max, name=f"{y_axis_id} [cm]")

        # h = hist.Hist(x_axis, y_axis)


        mct_g = self._make_groups(mct, 'generator_name')
        df = mct_g[name]
        h = _build_pos2d_hist(df, x_axis_id, y_axis_id, bin_width)

        # h.fill(df[x_axis_id], df[y_axis_id])

        fig, ax = plt.subplots(figsize=figsize)
        match c_scale:
            case "log":
                norm = LogNorm(vmin=1, vmax=h.values().max())
            case "lin":
                norm = None
            case _:
                raise RuntimeError(f"Unexpected norm value '{norm}' ('lin', 'log')")
        hep.hist2dplot(h, ax=ax, norm=norm)

        fig.tight_layout()
        return fig


    def plot_truth_generator_pos_ke(self, name:str, figsize: tuple = (8,8)) :
        if name not in self.ws.mctruth_blocks_map.values():
            raise RuntimeError(f"Generator {name} not found.")

        mct = self.ws.mctruths
        mct_g = self._make_groups(mct, 'generator_name')

        h_pos = {}
        for v in ['x', 'y', 'z']:
            v_min = mct[v].min()
            v_max = mct[v].max()

            n_binx_v = int((v_max-v_min)//20)
            hist_v = self._build_pos_hist(v, mct_g, n_binx_v, v_min, v_max)
            h_pos[v] = hist_v
        
        n_bins=100
        max_ke = mct.kinetic_energy.max() * 1000
        h_ke = self._build_ke_hist(self._make_groups(mct, 'generator_name'), n_bins, max_ke)
    
        fig, axes = plt.subplots(2,2, figsize=figsize)


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

    def plot_mc_truth_distributions(self, bins: int = 100, figsize: tuple = (14, 14)):
        """Plot histograms of all numeric MCTruth columns."""
        mct = self.ws.mctruths
        fig, ax = plt.subplots(figsize=figsize)
        mct.hist(ax=ax, bins=bins)
        fig.suptitle("MCTruth distributions")
        fig.tight_layout()
        return fig
