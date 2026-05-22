
from ..workspace import TriggerPrimitivesWorkspace

import numpy as np
import matplotlib.pyplot as plt
import hist
import mplhep as hep

from typing import Literal, Optional
from rich.table import Table
from ..detector_geometry import FDVDGeometry_1x8x14
from ..analysis.histograms import compute_regaxis_specs
from matplotlib.colors import LogNorm



class TrgPrimitivesPlotter:
    
    _electronics_noise_label : str = 'DetectorElectronics'

    def __init__(
        self,
        ws: TriggerPrimitivesWorkspace,
        geo: Literal['1x8x14']='1x8x14'
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
        
        # Shortcut
        self._df = ws.tps

        # Initialize block map
        self._init_tp_origin_block_map()

        self._init_det_geo(geo)

    
    def _init_det_geo(self, geo: Literal['1x8x14']):
        # Private variables
        match geo:
            case '1x8x14':
                self._geo = FDVDGeometry_1x8x14
            case _:
                raise ValueError(f'Illegal geo value. Received {geo} expected [1x8x14]')


    def simulated_time(self) -> float:
        """Return total simulated time in seconds.

        Computed as ``2 × readout_window × num_entries × 0.5 µs``, where the
        factor of 2 accounts for pre- and post-spill readout windows.
        Always derived from ``ws.mctruths`` regardless of the active
        collection.
        """
        
        sampling_time = 0.5e-6  # Sampling time 1/2 usec
        ro_win = self._df.extra_info['readout_window']
        num_entries = self._df.extra_info['num_entries']
        return ro_win * sampling_time * num_entries

    def _init_tp_origin_block_map(self):
        """Initialize the tp origin block registry

        The origin is either a MC sample (backtracked TPs)
        or "DetectorElectronics" (non-backtracked TPs)
        """
        block_map = self.ws.mctruth_blocks_map.copy()
        block_map[-99999] = self._electronics_noise_label
        self.block_map = block_map


    def _make_int_axis(self, col_name:str, **kwargs):
        """
        Helper method to create integer axis form a dataframe column

        TODO: refactor to a helper module
        """

        df = self._df
        if df[col_name].dtype.kind not in ("i", "u"):
            raise TypeError(f"Column {col_name} is not of type integer")

        values = df[col_name].values
        lo, hi = np.min(values), np.max(values)
        return hist.axis.Integer(lo, hi+1, name=col_name, **kwargs)


    def _make_intcat_axis(self, col_name:str, **kwargs):
        """
        Helper method to create category integer axis form a dataframe

        TODO: refactor to a helper module
        """

        df = self._df
        if df[col_name].dtype.kind not in ("i", "u"):
            raise TypeError(f"Column {col_name} is not of type integer")

        return hist.axis.IntCategory(sorted(d
                                            f[col_name].unique()), name=col_name, **kwargs)
    
    def _make_regaxis(self, col_name:str, **kwargs):
        df = self._df
        if df[col_name].dtype.kind not in ("i", "u", "f"):
            raise TypeError(f"Column {col_name} is not of type numeric")
        
        return hist.axis.Regular( *compute_regaxis_specs(df[col_name], 10), name=col_name)


    def _build_histogram(self, axes):

        h = hist.Hist(*axes)

        kwa = {
            a.name: self._df[a.name]
            for a in axes
        }
        h.fill(**kwa)

        return h



class TrgPrimitivesPlotterV0:
    
    _electronics_noise_label : str = 'ElecNoise'

    def __init__(
        self,
        ws: TriggerPrimitivesWorkspace,
        geo: Literal['1x8x14']='1x8x14'
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
        self.df = ws.tps

        # Initialize block map
        self._init_block_map()

        
        # Private variables
        match geo:
            case '1x8x14':
                self._geo = FDVDGeometry_1x8x14
            case _:
                raise ValueError(f'Illegal geo value. Received {geo} expected [1x8x14]')


    def simulated_time(self) -> float:
        """Return total simulated time in seconds.

        Computed as ``2 × readout_window × num_entries × 0.5 µs``, where the
        factor of 2 accounts for pre- and post-spill readout windows.
        Always derived from ``ws.mctruths`` regardless of the active
        collection.
        """
        
        sampling_time = 0.5e-6  # Sampling time 1/2 usec
        ro_win = self.df.extra_info['readout_window']
        num_entries = self.df.extra_info['num_entries']
        return ro_win * sampling_time * num_entries

    def _init_block_map(self):
        block_map = self.ws.mctruth_blocks_map.copy()
        block_map[-99999] = self._electronics_noise_label
        self.block_map = block_map




    #---- 

    def _make_groups(self, df, 
                     by:Literal['generator_name', 'truth_block_id', 'signal_noise'],
                     n_top: Optional[int] = None) -> dict[str, object]:
        """Return a ``{label: sub-DataFrame}`` dict grouped by *by*.

        Parameters
        ----------
        by:
            Grouping mode:

            * ``'generator_name'`` — group by MCTruth generator name directly.
            * ``'truth_block_id'`` — group MCParticles by truth block id,
              label resolved via ``ws.mctruth_blocks_map``.
        n_top:
            If given, return only the *n_top* largest groups by row count.
            ``None`` keeps all groups.
        """
        match by:
            case 'generator_name':
                groups = {name if name else 'ElecNoise': sub for name, sub in df.groupby('bt_generator_name')}
            case 'truth_block_id':
                groups = {self.block_map[n]: sub for n, sub in df.groupby('bt_truth_block_id')}
            case 'signal_noise':
                groups = {n:sub for n, sub in df.groupby('bt_is_signal')}
            case _:
                raise ValueError(f"Unknown groupby: {by!r}")
        groups = dict(sorted(groups.items(), key=lambda x: len(x[1]), reverse=True))
        if n_top is not None:
            groups = dict(list(groups.items())[:n_top])
        return groups

    
    

    def make_generator_counts_hist(self, cut: Optional[str] = None) -> hist.Hist:
        """Return a 1D StrCategory hist of generator counts for the active collection.

        For ``mctruths`` the groups are keyed by ``generator_name``.  For
        ``mcparticles`` they are keyed by ``truth_block_id`` resolved via
        ``ws.mctruth_blocks_map``.

        Parameters
        ----------
        cut:
            Optional pandas cut string applied to the DataFrame before
            filling (e.g. ``'pdg == 11'`` to select electrons only).
        """
        df = self.df
        if cut:
            df = df.query(cut)
        groups = self._make_groups(df, "generator_name")
        categories = list(groups.keys())
        label_axis = hist.axis.StrCategory(categories, name='generator', label='Generator')
        rop_axis = hist.axis.IntCategory(list(range(self._geo.num_readout_planes)), name='rop', label='Readout Plane')
        h = hist.Hist(label_axis, rop_axis)
        for label, sub in groups.items():
            # for rop, sub_rop in sub.groupby('readout_plane_id'):
            h.fill(generator=label, rop=sub['readout_plane_id'])
        return h
    

    def make_generator_rates_table(self, cut: Optional[str] = None) -> Table:
        """Return a rich Table of generator names, entry counts, and activity rates in Hz.

        Always derived from ``ws.mctruths`` regardless of the active
        collection.
        """
        simu_time = self.simulated_time()

        h_counts = self.make_generator_counts_hist(cut)

        t = Table('generator name', 'entries', 'rate [Hz]', title='Backgrounds generators by activity')
        for bin_label, count in zip(h_counts.axes[0], h_counts[:,sum].values()):
            t.add_row(bin_label, f'{int(count)}', f'{count / simu_time :.2f}')

        return t
    

    def plot_generator_activity(self, 
                                norm: Literal['counts', 'rate'] = 'counts',
                                cut: Optional[str]=None,
                                figsize: tuple = (10, 10)
                                ):
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
                clabel = 'counts'
                title_units = 'counts'
                norm_factor = 1
            case 'rate':
                clabel = 'Rates [Hz]'
                title_units = 'rates'
                norm_factor = 1 / self.simulated_time()
            case _:
                raise RuntimeError(f"norm ({norm!r}) must be 'counts' or 'rate'")



        h = self.make_generator_counts_hist(cut)
        h *= norm_factor

        c_scale = 'log'
        match c_scale:
            case "log":
                c_norm = LogNorm(vmin=1, vmax=h.values().max())
            case "lin":
                c_norm = None
            case _:
                raise RuntimeError(f"Unexpected c_scale value {c_scale!r} ('lin', 'log')")

        fig, ax = plt.subplots(figsize=figsize)
        # hep.histplot(h, yerr=False, histtype='fill', ax=ax)
        artists = hep.hist2dplot(h, ax=ax, norm=c_norm)
        artists.cbar.set_label("Rate [Hz]")
        # ax.grid(visible=True)
        ax.tick_params(axis='x', rotation=90)
        ax.set_ylabel('Readout Plane')
        # ax.set_yscale('log')

        title = f'Radioactive backgrounds {title_units}'

        fig.suptitle(title)
        fig.tight_layout()
        return fig