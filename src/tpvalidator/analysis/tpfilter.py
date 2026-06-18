import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal, Optional

from tpvalidator.viz.tps import TrgPrimitivesPlotter
from tpvalidator.utils import subplots_autogrid
from tpvalidator.viz.textual import dataframe_to_rich_table
from tpvalidator.viz.utilities import figure_manager
from tpvalidator.viz.efficiency import plot_roc


class TPFilterAnalyser:
    """Compare background and signal TP distributions to study filtering cuts.

    Wraps a pair of TrgPrimitivesPlotter instances — one for background
    (radiological) and one for signal (e-minus) — and provides plot methods
    that place both side-by-side for visual comparison.

    Attributes:
        bkg_tpp: TrgPrimitivesPlotter for the background workspace.
        sig_tpp: TrgPrimitivesPlotter for the signal workspace.
    """

    #----------
    def __init__(self, bkg_ws, sig_ws):
        """
        Args:
            bkg_ws: Background workspace (radiological sample).
            sig_ws: Signal workspace (e-minus sample).
        """
        self._bkg_ws = bkg_ws
        self._sig_ws = sig_ws

        self.bkg_tpp = TrgPrimitivesPlotter(self._bkg_ws)
        self.sig_tpp = TrgPrimitivesPlotter(self._sig_ws)


    #----------
    def _make_tpfilt_efficiency_df(self,
                            sample:Literal['sig', 'bkg'],
                            var:str,
                            var_cuts:list,
                            rop:int,
                            weight:Optional[str]=None,
                            query:Optional[str]=None,
                            generator_sel:Optional[str]=None,
                            event_filter:Optional[dict]=None
                            ):
        """Compute per-cut TP efficiency for one sample and readout plane.

        Builds a cut-sequence histogram over *var*, slices to the requested
        readout plane and signal flag, and normalises to the pre-cut count so
        each entry is the fraction of TPs surviving that minimum cut.

        Parameters
        ----------
        sample : {'sig', 'bkg'}
            Which TPPlotter to use — ``self.sig_tpp`` or ``self.bkg_tpp``.
        var : str
            TP column to cut on (e.g. ``'samples_over_threshold'``).
        var_cuts : list
            Ordered cut edges passed to ``make_cutsequence_hist``.
        rop : int
            Readout-plane ID to slice (complex-number index into the histogram).
        weight : str, optional
            TP column to use as fill weight. Plain count when None.
        query : str, optional
            pandas query string pre-filtering TPs before histogramming.
        generator_sel : str, optional
            Generator name to select (e.g. ``'Ar39GenInLAr'``). Sums over all
            generators when None.
        event_filter : dict, optional
            Passed through to ``make_cutsequence_hist`` for event-level
            filtering (e.g. by MC truth kinetic energy).

        Returns
        -------
        pd.DataFrame
            Indexed by the cut value with columns ``tp_eff`` and
            ``err_tp_eff`` (variance of the efficiency).
        """

        match sample:
            case 'sig':
                tpp = self.sig_tpp
            case 'bkg':
                tpp = self.bkg_tpp
            case _:
                raise ValueError(f'Unknow sample selector {sample}')

        # Build the cutsequence histogram for the chosen variable
        h_cs = tpp.make_cutsequence_hist(var, var_cuts, categories=['readout_plane_id', 'bt_is_signal', 'bt_generator_name'], weight=weight, query=query, event_filter=event_filter)

        # Extract the sub-histogram for the selection of rop, signal and generator label (if chosen)
        # Note: counting backtracked TPs only here
        h_cs_rop = h_cs[rop*1j,1j,generator_sel if generator_sel is not None else sum,:]

        # Calculate the distribution normalised to the first bin (no cut)
        # TODO: Now we are relying on the first bin for the normalization. Should store it in the underflow (no cut?)
        dist_rop = h_cs_rop/h_cs_rop.view(flow=True)[0].value

        cuts = dist_rop.axes[0].edges[:-1]
        eff = dist_rop.values()
        err_eff = dist_rop.variances()

        df_eff = pd.DataFrame({
            f'{var}_min': cuts,
            'tp_eff': eff,
            'err_tp_eff': err_eff
        }).set_index(f'{var}_min')

        return df_eff


    #----------
    def make_tpfilt_bkg_efficiency_df(self, var:str, var_cuts:list, rop:int=2, weight:Optional[str]=None, query:Optional[str]=None, generator_sel:Optional[str]=None):
        """TP efficiency vs cut for the background sample.

        Parameters
        ----------
        var : str
            TP column to cut on.
        var_cuts : list
            Ordered cut edges.
        rop : int, optional
            Readout-plane ID.
        weight : str, optional
            TP column used as fill weight. Plain count when None.
        query : str, optional
            pandas query string pre-filtering TPs.
        generator_sel : str, optional
            Generator name to select; sums over all generators when None.

        Returns
        -------
        pd.DataFrame
            Indexed by cut value with columns ``tp_eff`` and ``err_tp_eff``.
        """

        return self._make_tpfilt_efficiency_df('bkg', var, var_cuts, rop, weight=weight, query=query, generator_sel=generator_sel)


    def make_tpfilt_sig_efficiency_df(self, var:str, var_cuts:list, rop:int=2, weight:Optional[str]=None, query:Optional[str]=None, event_filter:dict=None):
        """TP efficiency vs cut for the signal sample.

        Parameters
        ----------
        var : str
            TP column to cut on.
        var_cuts : list
            Ordered cut edges.
        rop : int, optional
            Readout-plane ID.
        weight : str, optional
            TP column used as fill weight. Plain count when None.
        query : str, optional
            pandas query string pre-filtering TPs.
        event_filter : dict, optional
            Event-level filter passed to ``make_cutsequence_hist``
            (e.g. ``{'collection': 'mctruths', 'filter': 'kinetic_energy > 0.01'}``).

        Returns
        -------
        pd.DataFrame
            Indexed by cut value with columns ``tp_eff`` and ``err_tp_eff``.
        """

        return self._make_tpfilt_efficiency_df('sig', var, var_cuts, rop, weight=weight, query=query, event_filter=event_filter)


    #----------
    def make_tpfilt_efficiency_by_ke_df(self, var:str, var_cuts:list, ke_bins:list[int], rop:int, bkg_weight:Optional[str]=None,  sig_weight:Optional[str]=None, query:Optional[str]=None, generator_sel:Optional[str]=None):
        """Build a combined efficiency table with background and per-KE-bin signal columns.

        Computes background efficiency once, then appends one pair of signal
        efficiency columns (``tp_eff_sig_<ke_min>_to_<ke_max>`` and its error)
        for each kinetic-energy bin in *ke_bins*, merging everything on the cut
        index.

        Parameters
        ----------
        var : str
            TP column to cut on.
        var_cuts : list
            Ordered cut edges.
        ke_bins : list of (int, int)
            Kinetic-energy ranges in MeV (converted to GeV internally).
            E.g. ``[(0, 5), (5, 10), (10, 20)]``.
        rop : int
            Readout-plane ID.
        bkg_weight : str, optional
            Weight column for the background efficiency. Plain count when None.
        sig_weight : str, optional
            Weight column for the signal efficiency. Plain count when None.
        query : str, optional
            pandas query string applied to both samples.
        generator_sel : str, optional
            Background generator name to select.

        Returns
        -------
        pd.DataFrame
            Indexed by cut value with columns ``tp_bkg_eff``,
            ``err_tp_bkg_eff``, and one ``tp_eff`` / ``err_tp_eff`` pair per
            KE bin.
        """

        df_eff = self.make_tpfilt_bkg_efficiency_df(var, var_cuts, rop, generator_sel=generator_sel, query=query, weight=bkg_weight)
        df_eff.rename(columns={'tp_eff':'tp_bkg_eff', 'err_tp_eff':'err_tp_bkg_eff'}, inplace=True)

        for ke_min, ke_max in ke_bins:
            ke_evf = {'collection':'mctruths', 'filter':f'(kinetic_energy >= {ke_min}/1000) & (kinetic_energy < {ke_max}/1000)'}
            df_sig_eff = self.make_tpfilt_sig_efficiency_df(var, var_cuts, rop, event_filter=ke_evf, query=query, weight=sig_weight)

            # Explicitly rename columns before merge
            df_sig_eff.rename(columns={c:f'{c}_sig_{ke_min}_to_{ke_max}' for c in df_sig_eff.columns}, inplace=True)

            df_eff = df_eff.merge(df_sig_eff, on=f'{var}_min')

        return df_eff



    #----------
    def make_sig_evfilt_counts_vs_ke_df(self, var:str, query:Optional[str]=None, weight:Optional[str]=None, var_cuts:Optional[list]=None):
        """Build a per-event DataFrame of TP counts (or weighted sums) vs kinetic energy.

        Filters the signal TPs with *query*, then counts (or sums *weight*) per
        event with no variable cut (``no_{var}_cut`` column) and for each
        threshold in *var_cuts* (``{var}_min_{cut}`` columns).  The result is
        joined to the MC truth kinetic energy so each row represents one event.

        Parameters
        ----------
        var : str
            TP column to cut on (e.g. ``'samples_over_threshold'``).
        query : str, optional
            pandas query string applied to signal TPs before grouping
            (e.g. ``'adc_peak > 45'``).  No pre-filter when None.
        weight : str, optional
            Column of ``tps`` to sum per event (e.g. ``'adc_integral'``).
            When None the plain TP count is used instead.
        var_cuts : list of int, optional
            Thresholds to apply sequentially. Each value *N* adds a column
            ``{var}_min_{N}`` with counts after the ``{var} >= N`` selection.

        Returns
        -------
        pd.DataFrame
            One row per event with columns:
            ``event_uid``, ``kinetic_energy``, ``no_{var}_cut``,
            and one ``{var}_min_{N}`` column per entry in *var_cuts*.
        """
        if var_cuts is None:
            var_cuts = []

        ke = self._sig_ws.mctruths[['event_uid', 'kinetic_energy']].copy()

        base_filter = 'bt_is_signal == 1'
        full_filter = f'({base_filter}) & ({query})' if query is not None else base_filter
        tps = self._sig_ws.tps.query(full_filter)

        # Decorate tps with event kinetic energy ( particle gun only)
        tps_evke = tps.merge(ke, on='event_uid')

        # Group by event and plane
        ev_grps = tps_evke.groupby(['event_uid', 'readout_plane_id'], observed=False)
        counts = ev_grps[weight].sum() if weight is not None else ev_grps.size()
        # Name and reset index before merge
        counts = counts.rename(f'no_{var}_cut').reset_index()

        # add 'nocut' column
        ke_tpc = ke.merge(counts, on='event_uid', how='left')

        for sot_cut in var_cuts:
            sot_filt = f'{var} >= {sot_cut}'

            # Group by event and plane
            ev_grps = tps_evke.query(f'{sot_filt}').groupby(['event_uid', 'readout_plane_id'])
            counts = ev_grps[weight].sum() if weight is not None else ev_grps.size()

            # Name and reset index before merge
            counts = counts.rename(f'{var}_min_{sot_cut}').reset_index()

            # Add column (left-merge, )
            ke_tpc = ke_tpc.merge(counts, on=['event_uid', 'readout_plane_id'], how='left')

        ke_tpc.fillna(0, inplace=True)
        return ke_tpc


    # Plotting
    #----------
    def plot_top_vars_by_generator(self, dataset:Literal['sig', 'bkg'], rop=2, figsize=(16,4), **kwargs):
        """Plot SOT, ADC peak, and ADC integral distributions broken down by generator.

        Produces a 1×3 figure with one panel per variable
        (samples_over_threshold, adc_peak, adc_integral).

        Args:
            dataset: Which dataset to plot — 'sig' (signal) or 'bkg' (background).
            rop: Readout-plane index to select.
            figsize: Figure size passed to matplotlib.
            **kwargs: Forwarded to TrgPrimitivesPlotter.plot_var_by_generator.

        Returns:
            matplotlib Figure.
        """

        match dataset:
            case 'sig':
                tpp = self.sig_tpp
            case 'bkg':
                tpp = self.bkg_tpp
            case _:
                raise ValueError(f'Dataset {dataset} unknown')

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        tpp.plot_var_by_generator(rop=rop, var_spec='samples_over_threshold', ax=axes[0], **kwargs)
        tpp.plot_var_by_generator(rop=rop, var_spec='adc_peak', ax=axes[1], **kwargs)
        tpp.plot_var_by_generator(rop=rop, var_spec='adc_integral', ax=axes[2], **kwargs)

        fig.tight_layout()

        return fig


    #----------
    def plot_peak_vs_sot(self, weight=None, ke_evf=None, rop=2, query:str=None, cmap:str=None, generator_selection='Ar39GenInLAr', zoom:Optional[dict]=None):
        """Plot 2D ADC-peak vs samples-over-threshold for background and signal.

        Produces a 3×2 figure:
          - Row 0: background — all radiologicals (left) and a single generator
            (right, selected by ``generator_selection``).
          - Row 1: signal — all events (left) and filtered by ``ke_evf`` (right),
            with optional axis zoom applied.
          - Row 2: signal — same as row 1 but without the zoom.

        Args:
            weight: TP variable name used as histogram weight (e.g. 'adc_integral'),
                or None for unweighted counts.
            ke_evf: Event filter dict forwarded to plot_2d_var_dist as ``ev_filter``
                (e.g. ``{'collection': 'mctruths', 'filter': 'kinetic_energy < 0.02'}``).
            rop: Readout-plane index to select.
            query: Pandas query string applied to TPs before plotting.
            cmap: Matplotlib colormap name.
            generator_selection: Generator name used to select background sub-sample
                for the right panel of row 0.
            zoom: Optional dict with keys 'xmin', 'xmax', 'ymin', 'ymax' to set
                axis limits on rows 0 and 1.

        Returns:
            matplotlib Figure.
        """

        bkg_tpp = self.bkg_tpp
        sig_tpp = self.sig_tpp

        fig, axes = plt.subplots(3,2, figsize=(12,12))

        common_kwargs = {
            'rop':rop,
            'weight': weight,
            'cmap': cmap,
            'query': query
        }

        common_args = [
            'adc_peak', 'samples_over_threshold'
        ]
        row=0

        ax = axes[row][0]
        bkg_tpp.plot_2d_var_dist(*common_args, ax=ax, **common_kwargs)
        ax.set_title('radiologicals (all)')
        ax.grid()

        ax = axes[row][1]
        bkg_tpp.plot_2d_var_dist(*common_args, bt_generator_name=generator_selection, ax=ax, **common_kwargs)
        ax.set_title('Ar39')
        ax.grid()

        #-----

        row=1

        ke_evf_label = ke_evf["filter"] if ke_evf is not None else 'no ke filter'

        ax = axes[row][0]
        sig_tpp.plot_2d_var_dist(*common_args, ax=ax, **common_kwargs)
        ax.set_title('e-minus [zoom]')
        ax.grid()

        ax = axes[row][1]
        sig_tpp.plot_2d_var_dist(*common_args, ev_filter=ke_evf, ax=ax, **common_kwargs)
        ax.set_title(f'e-minus [zoom][{ke_evf_label}]')
        ax.grid()


        #----
        row=2

        ax = axes[row][0]
        sig_tpp.plot_2d_var_dist(*common_args, ax=ax, **common_kwargs)
        ax.set_title('e-minus')
        ax.grid()

        ax = axes[row][1]
        sig_tpp.plot_2d_var_dist(*common_args, ev_filter=ke_evf, ax=ax, **common_kwargs)
        ax.set_title(f'e-minus [{ke_evf_label}]')
        ax.grid()


        #----

        if zoom:

            xmin = zoom.get('xmin', None)
            xmax = zoom.get('xmax', None)
            ymin = zoom.get('ymin', None)
            ymax = zoom.get('ymax', None)

            for rax in axes[0:2]:
                for ax in rax:
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)

        fig.suptitle("Trigger Primitives multiplicity in sadc and peak - zoom on the low-E region")
        fig.tight_layout()

        return fig


class DevTPFilterAnalyser(TPFilterAnalyser):
    # TODO: define analyzer level variables for
    # - Filter variable
    # - Filter var cuts
    # - Readout plane (TBD)
    # - Weights (signal and background)
    pass


class SOTFilterAnalyser(DevTPFilterAnalyser):
    # TODO: define analyzer level variables for
    # - Filter variable
    # - Filter var cuts
    # - Readout plane (TBD)
    # - Weights (signal and background)

    #----------
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        first_sot=2
        last_sot=30

        self.var = 'samples_over_threshold'
        self.var_cuts = list(np.arange(first_sot-0.5,last_sot+0.5))
        self.tp_query = 'adc_peak > 45'
        self.readout_plane_id = 2
        self.sig_weight = 'adc_integral'
        self.bkg_weight = None
        self.bkg_generator_sel = 'Ar39GenInLAr'
        self.ke_bins = [(0, 5), (5, 10)] + [(ke, ke+10) for ke in range(10,101,10)]
        self.ke_edges = [ l for l,h in self.ke_bins]+[self.ke_bins[-1][1]]

    #------ Helper functions -----------------------
    #----------
    def make_sot_tpfilt_efficiency_by_ke_df(self, rop:int):

        return self.make_tpfilt_efficiency_by_ke_df(
            self.var,
            self.var_cuts,
            self.ke_bins,
            rop=rop,
            bkg_weight=self.bkg_weight,
            sig_weight=self.sig_weight,
            generator_sel=self.bkg_generator_sel,
            query=self.tp_query
        )


    #----------
    def make_sot_tpfilt_sig_efficiency_df(self, rop:int, event_filter:dict):

        return self.make_tpfilt_sig_efficiency_df(
            self.var,
            self.var_cuts,
            rop=rop,
            event_filter=event_filter,
            query=self.tp_query,
            weight=self.sig_weight)

    #----------
    def make_sot_tpfilt_bkg_efficiency_df(self, rop:int):

        return self.make_tpfilt_bkg_efficiency_df(
            self.var,
            self.var_cuts,
            rop=rop,
            generator_sel=self.bkg_generator_sel,
            query=self.tp_query,
            weight=self.bkg_weight
        )

    #------ Table functions -----------------------
    def make_sot_bkg_eff_table(self, rop:int):

        df_rad_eff = self.make_sot_tpfilt_bkg_efficiency_df(rop=rop)
        fmts={c:'{:.2}' for c in df_rad_eff.columns }
        t = dataframe_to_rich_table(df_rad_eff, formatters=fmts, show_index=True, index_name='sot min (>=)')
        return t

    #------ Plotting functions -----------------------
    def plot_signal_sot_tpfilt_counts_ke_matrix(self,
                                        rop:int=2,
                                        ax:object=None,
                                        **fig_kwargs
                                    ):

        with figure_manager(ax, **fig_kwargs) as (fig, ax):

            for ke_min, ke_max in self.ke_bins:
                ke_evf = {'collection':'mctruths', 'filter':f'(kinetic_energy > {ke_min}/1000) & (kinetic_energy < {ke_max}/1000)'}
                df_sig_eff = self.make_sot_tpfilt_sig_efficiency_df(rop=rop, event_filter=ke_evf)
                df_sig_eff.plot(y='tp_eff', yerr='err_tp_eff', ax=ax, label=f'{ke_min} < KE < {ke_max}', linestyle='--')

            ax.grid()

        return fig

    #----------
    def plot_sot_tpfilt_roc_matrix(self,
                                    rop:int=2,
                                    refcuts=[],
                                    figsize=(16,10)
                                ):

        df_eff = self.make_sot_tpfilt_efficiency_by_ke_df(rop=rop)

        sig_eff_cols = [c for c in df_eff.columns if c.startswith('tp_eff')]

        fig, axes = subplots_autogrid(len(sig_eff_cols), figsize=figsize)

        for i, sc in enumerate(sig_eff_cols):
            plot_roc(df_eff, 'tp_bkg_eff', sc, ax=axes[i], refcuts=refcuts, xlabel=f'{self.bkg_generator_sel if self.bkg_generator_sel else "RadBkg"} efficiency', ylabel='e-minus efficiency')
            axes[i].set_title(sc)

        fig.tight_layout()

        return fig


    #----------
    def plot_signal_sot_tpfilt_eff_by_ke(self,
                                    rop:int=2,
                                    ax:object=None,
                                    **fig_kwargs):

        with figure_manager(ax, **fig_kwargs) as (fig, ax):

            df_eff = self.make_sot_tpfilt_efficiency_by_ke_df(rop=rop)

            sig_eff_cols = [c for c in df_eff.columns if c.startswith('tp_eff')]

            for c in sig_eff_cols:
                df_eff.plot('tp_bkg_eff', c, ax=ax)

            ax.grid()
            ax.legend()

            ax.set_ylim(ymin=0.8, ymax=1.01)

            fig.suptitle("ROC as a function of the $e^{-}$ energy range")

        return fig

    #----------
    def plot_bkg_tpcounts_efficiency(self, rop:int, ax:object=None, **fig_kwargs):

        df_rad_eff = self.make_sot_tpfilt_bkg_efficiency_df(rop=rop)

        with figure_manager(ax, **fig_kwargs) as (fig, ax):
            df_rad_eff.plot(y='tp_eff', yerr='err_tp_eff', label=f'{self.bkg_generator_sel if self.bkg_generator_sel else "radbkg"} efficiency', ax=ax)
            ax.grid()
            ax.set_ylabel('efficiency')
        return fig
