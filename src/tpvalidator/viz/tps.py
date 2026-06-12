
from ..workspace import TriggerPrimitivesWorkspace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hist
import mplhep as hep
import copy

from typing import Literal, Optional
from rich.table import Table
from matplotlib.colors import LogNorm

from ..detector_geometry import FDVDGeometry_1x8x14
from ..analysis.histograms import compute_regaxis_specs, cumsum_hist_nd, build_histogram, make_intcat_axis, make_strcat_axis
from .textual import dataframe_to_rich_table



class TrgPrimitivesPlotter:
    
    _electronics_noise_label : str = 'DetSimElecNoise'
    _default_var_specs = {
        'adc_peak': {'name':'adc_peak', 'bin_size':10},
        'samples_over_threshold': {'name':'samples_over_threshold', 'bin_size': 2},
        'adc_integral': {'name':'adc_integral', 'bin_size': 10}
    }

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
        geo:
            <to add>

        """
        self.ws = ws
        
        # Shortcut
        self._df = ws.tps

        # Initialize block map
        self._init_tp_origin_block_map()

        # Initialize geometry
        self._init_det_geo(geo)

        # Make initialize var specs to allow tweaking
        self.var_specs = copy.deepcopy(self._default_var_specs)

    
    def _init_det_geo(self, geo: Literal['1x8x14']):
        # Private variables
        match geo:
            case '1x8x14':
                self._geo = FDVDGeometry_1x8x14
            case _:
                raise ValueError(f'Illegal geo value. Received {geo} expected [1x8x14]')

    @property
    def geo(self):
        return self._geo

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

    def _get_cat_axis_list(self, df:pd.DataFrame, categories: list[str]) -> list[hist.axis.AxisProtocol]:
       
        h_spec = []
        for cat in categories:
            match cat:
                case 'readout_plane_id':
                    rop_axis = make_intcat_axis(df, 'readout_plane_id', label='Readout Plane')
                    h_spec.append(rop_axis)

                case 'bt_is_signal':
                    bt_sig_axis = make_intcat_axis(df, 'bt_is_signal', label='Noise/Signal')
                    h_spec.append(bt_sig_axis)

                case 'bt_generator_name':
                    bt_gen_axis = make_strcat_axis(df, 'bt_generator_name', label='Generator')
                    h_spec.append(bt_gen_axis)

                case _:
                    raise ValueError(f"Category {cat} not known")
                
        return h_spec
    
    def _get_cat_axis_list(self, df:pd.DataFrame, categories: list[str]) -> list[hist.axis.AxisProtocol]:

        # Static map of known category axis maker
        cat_makers_map = {
            'readout_plane_id': lambda df: make_intcat_axis(df, 'readout_plane_id', label='Readout Plane'),
            'bt_is_signal': lambda df: make_intcat_axis(df, 'bt_is_signal', label='Noise/Signal'),
            'bt_generator_name': lambda df: make_strcat_axis(df, 'bt_generator_name', label='Generator')
        }
        h_spec = []
        for cat in categories:
            maker = cat_makers_map.get(cat, None)

            if maker is None:
                raise ValueError(f"Category {cat} not known")
            h_spec.append(maker(df))

                
        return h_spec
    
    

    def make_hist(self,
                var_spec:list[dict|str]|dict|str=[],
                categories: list[str]=['readout_plane_id'],
                weight: Optional[str]=None,
                query: Optional[str]=None,
                event_filter: Optional[dict]=None
            ):

        df = self._df

        # TODO: generalize
        if event_filter:
            evf_collection = event_filter['collection']
            evf_filter = event_filter['filter']

            coll = self.ws.get_df(evf_collection)

            # TODO: switch to using event_uid
            index_list = coll.query(evf_filter).index

            # Extract the top level
            top_level = {idx[0] for idx in index_list}

            df = df[df.index.get_level_values(0).isin({idx for idx in top_level})]


            # coll.query(evf_filter).event_uid.unique()
            

        if query:
            df = df.query(query)

        h_spec = self._get_cat_axis_list(df, categories)

        if isinstance(var_spec, dict):
            var_spec = [var_spec]

        for vs in var_spec:
            if isinstance(vs, str):
                vs = self.var_specs.get(vs)

            v_name = vs['name']
            v_bin_size = vs['bin_size']
            v_label = vs.get('label', v_name)
            
            var_axis = hist.axis.Regular( *compute_regaxis_specs(df[v_name], v_bin_size), name=v_name, label=v_label)
            h_spec.append(var_axis)

        h = build_histogram(df, h_spec, weight=weight)
        return h
        

    
    def make_var_hist(self, var:str, var_binsize: int, **kwargs):

        """Build and fill a 3-axis histogram over readout plane, backtracker signal flag, and a TP variable.

        The variable axis is a regularly-spaced ``hist.axis.Regular`` whose range and
        number of bins are derived automatically from the data via
        ``compute_regaxis_specs``.

        Args:
            var: Column name in the TP dataframe to use as the third axis.
            var_binsize: Bin width for the regular axis (same units as the column).

        Returns:
            hist.Hist: Filled histogram with axes ``[readout_plane_id, bt_is_signal, var]``.
        """

        var_spec={'name':var, 'bin_size':var_binsize, 'label':var}
        return self.make_hist(var_spec=var_spec, categories=['readout_plane_id', 'bt_is_signal'], **kwargs)
    
    

    def make_cutsequence_hist_legacy(self, var:str, cuts: list[float], weight:str=None, ):
        """Build a cumulative histogram over a variable-width cut sequence.

        Creates a 3-axis histogram (readout plane, backtracker signal flag, variable)
        using explicit bin edges defined by ``cuts``, then computes the right-to-left
        cumulative sum so each bin gives the count surviving that threshold and above.

        Args:
            var: Column name in the TP dataframe to use as the cut-sequence axis.
            cuts: Explicit bin edges for the variable axis (monotonically increasing).

        Returns:
            hist.Hist: Cumulative histogram with axes ``[readout_plane_id, bt_is_signal, var]``,
                where each bin contains the count of entries with ``var >= bin_lower_edge``.
        """

        var_axis = hist.axis.Variable(cuts, name=var)

        rop_axis = make_intcat_axis(self._df, 'readout_plane_id', label='Readout Plane')
        bt_sig_axis = make_intcat_axis(self._df, 'bt_is_signal', label='Noise/Signal')

        h_spec = [rop_axis, bt_sig_axis, var_axis]
        h = build_histogram(self._df, h_spec, weight=weight)

        # Calculate the cumulative histogram
        h_cs = cumsum_hist_nd(h, var, direction='right', flow=True)

        return h_cs
    
    def make_cutsequence_hist(self,
                            cut_var:str,
                            cuts: list[float],
                            categories:list[str]=['readout_plane_id', 'bt_is_signal'],
                            weight: Optional[str]=None,
                            query: Optional[str]=None,
                            event_filter: Optional[dict]=None
                            ):
        
        df = self._df

        # TODO: generalize
        if event_filter:
            evf_collection = event_filter['collection']
            evf_filter = event_filter['filter']

            coll = self.ws.get_df(evf_collection)
            index_list = coll.query(evf_filter).index

            # Extract the top level
            top_level = {idx[0] for idx in index_list}

            df = df[df.index.get_level_values(0).isin({idx for idx in top_level})]


        if query:
            df = df.query(query)
        
        h_spec = self._get_cat_axis_list(df, categories)
        h_spec.append(
            hist.axis.Variable(cuts, name=cut_var)
        )
        h = build_histogram(df, h_spec, weight=weight)

        # Calculate the cumulative histogram
        h_cs = cumsum_hist_nd(h, cut_var, direction='right', flow=True)

        return h_cs


    def make_generator_counts_hist(self, query: Optional[str] = None) -> hist.Hist:
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

        return self.make_hist(categories=['bt_generator_name', 'readout_plane_id'], query=query)
    

    def make_generator_activity_table(self,
                                    query: Optional[str] = None,
                                    norm: Literal['counts', 'rate'] = 'rate',
                                    geo_norm: Literal['default', 'crp', 'tpc'] = 'default'
                                    ) -> Table:
        """Return a rich Table of generator activity, ranked by count or rate.

        Parameters
        ----------
        query:
            Optional pandas query string applied before histogramming.
        norm:
            ``'counts'`` to show raw hit counts; ``'rate'`` to normalise by
            simulated time and express in Hz (default).
        geo_norm:
            Geometric normalisation applied on top of ``norm``: ``'default'``
            uses the full simulation volume (no extra factor), ``'crp'``
            divides by the number of CRPs, ``'tpc'`` divides by the number of
            TPCs.

        Returns
        -------
        rich.table.Table
            Rows are sorted by the normalised value in descending order.
            Unlabelled entries (electronics noise) are shown as
            ``'ElecNoise'``.
        """
        simu_time = self.simulated_time()

        h_counts = self.make_generator_counts_hist(query)


        # TODO: The norm and geo norm handling is general. Refactor in a separate method
        match norm:
            case 'counts':
                norm_unit = 1.
                col_name = 'counts'
                fmts = {
                    col_name:'{:.2f}'
                }
            case 'rate':
                norm_unit = 1./ simu_time
                col_name = 'rate'
                fmts = {
                    col_name:'{:.2f} HZ'
                }
            case _:
                raise ValueError(f'Invalid normalisation {norm}')
        
        match geo_norm:
            case 'default':
                # use the simulation geometry
                norm_geo = 1.
            case 'crp':
                norm_geo = 1./self.geo.num_crps                
            case 'tpc':
                norm_geo = 1./self.geo.num_tpcs
            case _:
                raise ValueError(f"Invalid 'detgeo' parameter {geo_norm}")

        h_counts *= norm_unit*norm_geo
        det_name = self.geo.name if geo_norm=='default' else geo_norm

        c_df = pd.DataFrame({
            'generator': [l if len(l) > 0 else self._electronics_noise_label for l in h_counts.axes[0]],
            col_name: h_counts[:,sum].values()
            })
        

        return dataframe_to_rich_table(c_df.sort_values(col_name, ascending=False),show_index=True, formatters=fmts, title=f'Rates per generator ({det_name})')


    def plot_var_by_generator(self,
                            var_spec:dict|str,
                            rop: int,
                            n_top: int=10,
                            norm: Literal['counts', 'rate'] = 'counts',
                            geo_norm: Literal['default', 'crp', 'tpc'] = 'default',
                            query: Optional[str] = None,
                            ax: Optional[object] = None,
                            **fig_kwargs
        ):
        """Plot a TP variable distribution broken down by backtracked generator.

        Parameters
        ----------
        var_spec:
            Variable specification dict passed to :meth:`make_hist` (keys: ``var``,
            ``bins``, ``range``, and optionally ``label``).
        rop:
            Readout-plane index to select (sliced from the 2-D histogram).
        n_top:
            Number of top generators to show, ranked by total counts (default 10).
        query:
            Optional pandas query string applied before histogramming.
        **kwargs:
            Extra keyword arguments forwarded to ``plt.subplots``.

        Returns
        -------
        matplotlib.figure.Figure
        """

        if isinstance(var_spec, str):
            var_spec = self.var_specs[var_spec]

        h_var = self.make_hist(query=query, var_spec=var_spec, categories=['bt_generator_name', 'readout_plane_id'])

        match norm:
            case 'counts':
                clabel = 'counts'
                title_units = 'counts'
                norm_unit = 1
            case 'rate':
                clabel = 'Rates [Hz]'
                title_units = 'rates'
                norm_unit = 1 / self.simulated_time()
            case _:
                raise RuntimeError(f"norm ({norm!r}) must be 'counts' or 'rate'")
            
        match geo_norm:
            case 'default':
                # use the simulation geometry
                norm_geo = 1.
            case 'crp':
                norm_geo = 1./self.geo.num_crps                
            case 'tpc':
                norm_geo = 1./self.geo.num_tpcs
            case _:
                raise ValueError(f"Invalid 'detgeo' parameter {geo_norm}")
            
        h_var *= norm_unit*norm_geo

        
        h2_var = h_var[{'readout_plane_id':rop}]
        s = h2_var.stack("bt_generator_name")

        h_top = sorted([h for h in s], key=lambda x: x.sum(), reverse=True)[:n_top]

        if 'figsize' not in fig_kwargs:
            fig_kwargs['fig_size'] = (8,5)

        create_fig = ax is None
        fig, ax = plt.subplots(**fig_kwargs) if create_fig else (ax.figure, ax)

        for h in h_top:
            hep.histplot(h, ax=ax, label=h.name if h.name else self._electronics_noise_label, yerr=False)

        # s.plot(ax=ax)
        ax.legend()
        ax.set_ylabel(clabel)
        ax.set_yscale('log')
        
        
        if create_fig:
            fig.tight_layout()
        return fig
    

    def plot_2d_var_dist(self, var_spec_x, var_spec_y, rop=2, weight=None, cmap:str=None, bt_generator_name:str=None, ev_filter:dict=None, ax=None, 
                                **fig_kwargs
                        ):
        # histogram categories
        cats = ['readout_plane_id', 'bt_is_signal', 'bt_generator_name']

        h = self.make_hist(var_spec=[var_spec_x, var_spec_y], categories=cats, weight=weight, event_filter=ev_filter)

        create_fig = ax is None
        fig, ax = plt.subplots(**fig_kwargs) if create_fig else (ax.figure, ax)

        bt_gen = bt_generator_name if bt_generator_name else sum

        artists = hep.hist2dplot(h[rop*1j,1j,bt_gen,:,:], norm=LogNorm(), cmap=cmap, ax=ax)
        artists.cbar.set_label("Counts")

        if create_fig:
            fig.tight_layout()
        return fig






























#--------------------------------------------------------------

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
                                figsize: tuple = (8, 6)
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