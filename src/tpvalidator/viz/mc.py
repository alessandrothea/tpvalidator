
import numpy as np
import matplotlib.pyplot as plt
import hist
import particle
import mplhep as hep

from rich.table import Table
from typing import Optional

from ..workspace import TriggerPrimitivesWorkspace


class MCPlotter:

    def __init__(self, ws: TriggerPrimitivesWorkspace):

        # if ws.simides is None:
            # raise RuntimeError(f"No IDE data available in '{ws._data_path}'")

        self.ws = ws

    def simulated_mc_time(self):
        ro_win = self.ws.mctruths.extra_info['readout_window']
        num_entries = self.ws.mctruths.extra_info['num_entries']

        # print('ro_win',ro_win, 'num_entries',num_entries)

        simulated_time = 2 * ro_win * 0.5e-6 * num_entries
        return simulated_time


    def make_generators_table(self) -> Table:
        """Return a rich Table listing background generator names and their block ids."""
        t = Table('generator name', 'id', title='Background generators')
        for gid, name in self.ws.mctruth_blocks_map.items():
            t.add_row(name, str(gid))
        return t


    def make_generator_rates_table(self) -> Table:
        simu_time = self.simulated_mc_time()

        t = Table('generator name', 'entries', 'rate [Hz]', title='Backgrounds generators by activity')
        part_by_gen = sorted([(n, _df) for n, _df in self.ws.mctruths.groupby('generator_name')], reverse=True, key=lambda x: len(x[1]))
        for _gen_id, _df in part_by_gen:
            t.add_row(_gen_id, str(len(_df)), f'{len(_df) / simu_time :.2e}')
        return t

    def plot_distributions(self, bins: int = 100, figsize: tuple = (14, 14)):
        mct = self.ws.mctruths
        fig, ax  = plt.subplots(figsize=figsize)
        mct.hist(ax=ax, bins=bins )

        fig.tight_layout()
        return fig



    def make_ke_spectra_by_pdg_hists(self, n_bins: int = 100) -> tuple[hist.Hist, hist.Hist]:
        """Return (h_mct, h_mcp) KE histograms keyed by PDG name."""
        mct, mcp = self.ws.mctruths, self.ws.mcparticles
        max_ke = max(mct.kinetic_energy.max(), mcp.kinetic_energy.max()) * 1000
        ke_axis = hist.axis.Regular(n_bins, 0, max_ke, name='ke', label=r'$E_{kin}$ [MeV]')

        def build(df):
            labels = sorted({particle.Particle.from_pdgid(p).pdg_name for p in df.pdg.unique()})
            h = hist.Hist(hist.axis.StrCategory(labels, name='group'), ke_axis)
            for pdg_id, sub in df.groupby('pdg'):
                h.fill(group=particle.Particle.from_pdgid(pdg_id).pdg_name,
                       ke=sub.kinetic_energy.values * 1000)
            return h

        return build(mct), build(mcp)

    def make_ke_spectra_by_generator_hists(self, n_top: int = 10, n_bins: int = 100) -> tuple[hist.Hist, hist.Hist]:
        """Return (h_mct, h_mcp) KE histograms keyed by generator name (top-N by count)."""
        mct, mcp = self.ws.mctruths, self.ws.mcparticles
        block_map = self.ws.mctruth_blocks_map
        max_ke = max(mct.kinetic_energy.max(), mcp.kinetic_energy.max()) * 1000
        ke_axis = hist.axis.Regular(n_bins, 0, max_ke, name='ke', label=r'$E_{kin}$ [MeV]')

        top_mct = sorted(mct.groupby('generator_name'), key=lambda x: len(x[1]), reverse=True)[:n_top]
        top_mcp = sorted(
            [(block_map[n], df) for n, df in mcp.groupby('truth_block_id')],
            key=lambda x: len(x[1]), reverse=True,
        )[:n_top]

        def build(groups):
            labels = [label for label, _ in groups]
            h = hist.Hist(hist.axis.StrCategory(labels, name='group'), ke_axis)
            for label, sub in groups:
                h.fill(group=label, ke=sub.kinetic_energy.values * 1000)
            return h

        return build(top_mct), build(top_mcp)

    def _plot_ke_panel(self, ax, h: hist.Hist, title: str, color_map: Optional[dict] = None):
        """Plot all slices of a (StrCategory × Regular) hist.Hist as step histograms."""
        for label in h.axes['group']:
            kw = dict(label=label, histtype='step', yerr=False, ax=ax)
            if color_map:
                kw['color'] = color_map.get(label)
            hep.histplot(h[label, :], **kw)

        ax.legend()
        ax.set_yscale('log')
        ax.set_title(title)
        ax.grid(visible=True)

    def plot_ke_spectra_by_pdg(self, n_bins: int = 100, figsize: tuple = (14, 6)):
        """Plot kinetic-energy spectra of MCTruths and MCParticles by PDG id."""
        h_mct, h_mcp = self.make_ke_spectra_by_pdg_hists(n_bins)

        all_labels = sorted(set(h_mct.axes['group']) | set(h_mcp.axes['group']))
        colors = plt.cm.tab10.colors
        color_map = {label: colors[i % len(colors)] for i, label in enumerate(all_labels)}

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        self._plot_ke_panel(axes[0], h_mct, 'MCTruths particles', color_map)
        self._plot_ke_panel(axes[1], h_mcp, 'MCParticles (G4)', color_map)
        fig.tight_layout()
        return fig

    def plot_ke_spectra_by_generator(self, n_top: int = 10, n_bins: int = 100, figsize: tuple = (14, 6)):
        """Plot kinetic-energy spectra of the top-N generators for MCTruths and MCParticles."""
        h_mct, h_mcp = self.make_ke_spectra_by_generator_hists(n_top, n_bins)

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        self._plot_ke_panel(axes[0], h_mct, 'MCTruths particles')
        self._plot_ke_panel(axes[1], h_mcp, 'MCParticles (G4)')
        fig.tight_layout()
        return fig

    def make_generator_counts_hist(self, query: Optional[str]=None) -> hist.Hist:
        """Return a hist.Hist of generator counts for particles of a given PDG id."""
        print(query)
        df = self.ws.mctruths
        if query:
            df = df.query(query)
        categories = sorted(df.generator_name.unique().tolist())
        h = hist.Hist(hist.axis.StrCategory(categories, name='generator', label='Generator'))
        h.fill(generator=df.generator_name.values)
        return h

    def plot_generator_activity(self, norm: str='counts', pdg_id: Optional[int]= None, figsize: tuple = (10, 10)):
        """Plot MCTruth generator counts for particles of a given PDG id."""

        match norm:
            case 'counts':
                ylabel='counts'
                title_units = 'counts'
                norm_factor = 1
                pass
            case 'rate':
                ylabel='Rates [Hz]'
                title_units = 'rates'
                norm_factor = 1/self.simulated_mc_time()
                pass
            case _:
                raise RuntimeError(f"norm ({norm}) is not one of the allowed values : 'counts', 'rate'")


        # df = self.ws.mctruths
        # if pdg_id is not None:
            # df = self.ws.mctruths.query(f'pdg == {pdg_id}')
        query = f'pdg == {pdg_id}' if pdg_id is not None else None
        h = self.make_generator_counts_hist(query)
        h *= norm_factor

        # n_bins = df.generator_name.nunique()

        fig, ax = plt.subplots(figsize=figsize)
        # df.generator_name.hist(bins=n_bins, weights=np.ones(len(df)) * norm_factor, ax=ax)
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
    
