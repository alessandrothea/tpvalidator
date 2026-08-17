
import hist
import mplhep as hep
import pandas as pd

from ..detgeometry import get_by_geocfg_id
from ..analysis.base import TrgWorkspaceAnalyzer

from typing import Optional
from .histograms import compute_regaxis_specs, cumsum_hist_nd, build_histogram, make_intcat_axis, make_strcat_axis

def make_wins(tps: pd.DataFrame):
    summary = (
        tps
        .groupby(['event_uid', 'TPCSetID', 'readout_plane_id','tawin_id'], observed=False, sort=False)
        .agg(
            n_tps=('tawin_id', "size"),
            sadc=("adc_integral", "sum"),
        )
        .reset_index()
    )

    return summary


class Calibrator:

    def __init__(self, m, b):
        self.m = m
        self.b = b
        
    def sadc_to_mev(self, sadc ):
        mev = sadc/self.m-self.b/self.m
        return mev

    def mev_to_sadc(self, mev ):
        sadc = self.m*mev+self.b
        return sadc



class TAWindowAnalyzer(TrgWorkspaceAnalyzer):
    
    #-----------
    def __init__(self, ws, win_len):
        super(TAWindowAnalyzer, self).__init__(ws)

        self.win_len = win_len

        # initialization
        self.tps_in_win = self.ws.tps.copy()
        self.tps_in_win['tawin_id'] = self.tps_in_win.sample_peak // self.win_len

        self.ta_wins = make_wins(self.tps_in_win)

    #-----------
    def _apply_event_filter(self, df, event_filter:dict):
        evf_collection = event_filter['collection']
        evf_filter = event_filter['filter']

        coll = self.ws.get_df(evf_collection)

        ev_uids = coll.query(evf_filter).event_uid.unique()
        df = df[df.event_uid.isin(ev_uids)]

        return df


    #-----------
    def _get_cat_axis_list(self, df:pd.DataFrame, categories: list[str]) -> list[hist.axis.AxisProtocol]:
       
        h_spec = []
        for cat in categories:
            match cat:
                case 'readout_plane_id':
                    rop_axis = make_intcat_axis(df, 'readout_plane_id', label='Readout Plane')
                    h_spec.append(rop_axis)

                # case 'bt_is_signal':
                #     bt_sig_axis = make_intcat_axis(df, 'bt_is_signal', label='Noise/Signal')
                #     h_spec.append(bt_sig_axis)

                # case 'bt_generator_name':
                #     bt_gen_axis = make_strcat_axis(df, 'bt_generator_name', label='Generator')
                #     h_spec.append(bt_gen_axis)

                case _:
                    raise ValueError(f"Category {cat} not known")
                
        return h_spec
    


    #-----------
    def make_tawin_hist(self,
                var_spec:list[dict|str]|dict|str=[],
                categories: list[str]=['readout_plane_id'],
                weight: Optional[str]=None,
                query: Optional[str]=None,
                event_filter: Optional[dict]=None
            ):
        """Build a boost-histogram from the TP dataframe.

        Args:
            var_spec: Variable(s) to histogram. Each entry is either a string key
                into ``self.var_specs`` or a dict with keys ``name``, ``bin_size``,
                and optionally ``label`` and ``type``. A single dict is accepted
                in place of a one-element list.
            categories: Column names to use as categorical axes. Defaults to
                ``['readout_plane_id']``.
            weight: Column name whose values are used as per-entry weights.
            query: Pandas query string applied to the dataframe before filling.
            event_filter: Restrict entries to events that pass a filter on a
                different collection. Dict with keys ``'collection'`` (name passed
                to ``ws.get_df()``) and ``'filter'`` (query string applied to that
                collection); only rows whose ``event_uid`` appears in the filtered
                collection are kept.

        Returns:
            boost_histogram.Histogram with one categorical axis per category and
            one regular axis per variable.
        """
        df = self.ta_wins

        # TODO: generalize
        if event_filter:

            df = self._apply_event_filter(df, event_filter)


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
            v_type = vs.get('type', 'float')
            
            n_bins, xmin, xmax = compute_regaxis_specs(df[v_name], v_bin_size, binning_type=v_type)

            var_axis = hist.axis.Regular( n_bins, xmin, xmax, name=v_name, label=v_label)

            h_spec.append(var_axis)

        h = build_histogram(df, h_spec, weight=weight)
        return h