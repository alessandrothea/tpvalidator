from rich import print
from ...workspace import TrgDataFrame
import os.path


#-----------------------
import uproot
import numpy as np
import pandas as pd
import json


from tpvalidator.algo.tafinder.trigger_algs_numba import apply_dbscan

#-----------------------

default_cfg = {
    'preselection': 'readout_view == 2 & samples_over_threshold >= 9',
    'ta_win_size': 1000,
    'ta_win_start': 0,
    'ta_inspect_sadc_min': 7500,
    'ta_inspect_sadc_max': 50000,
    'ta_inspect_sadc_cluster_threshold': 7500,
    'ta_win_sadc_dist_file': None,
    'ta_win_sadc_dist_rdm_seed': 123,
    'ta_win_sadc_add_bkg': False,
    'ta_dbscan_epsilon': 2,
    'ta_dbscan_min_neigh': 2
}


class TriggerPrimitivesProcessor:

    def __init__(self):
        pass


    def apply_preselection(self, tps_df: TrgDataFrame) -> TrgDataFrame:

        return tps_df
    
    def create_windows(self, tps_df: TrgDataFrame) -> None:
        return
    
    def process_windows(self, tsw_df: TrgDataFrame) -> None:
        return

    def process(self, tps_df: TrgDataFrame) -> None:

        return

class DFWriter:
    pass


class RootDFWriter:

    def __init__(self, rootfilename, rootfolder):
        self._filename = rootfilename
        self._folder = rootfolder

        # Force recreation
        # with uproot.recreate(self._filename) as f:
            # f.mkdir(self._folder)

        self._rootfile = uproot.recreate(self._filename) 
        self._rootfile.mkdir(self._folder)


    def write(self, df, treename):
        # Map numpy/pandas dtypes to uproot branch types (numpy dtypes are usually fine)
        branch_types = {c: np.asarray(df[c].to_numpy()).dtype for c in df.columns}
        branch_arrays = {c: np.asarray(df[c].to_numpy()) for c in df.columns}

        dtype_obj = np.dtype('O')
        branch_types = {k:(v if v != dtype_obj else np.dtype(str)) for k,v in branch_types.items()}

        # with uproot.update(self._filename) as f:
        if treename not in self._rootfile[self._folder]:
            self._rootfile[self._folder].mktree(treename, branch_types)
        self._rootfile[self._folder][treename].extend(branch_arrays)

    def writemeta(self, name:str, metadata:dict):
        # with uproot.update(self._filename) as f:
        #     f[self._folder][name] = json.dumps(metadata)
        self._rootfile[self._folder][name] = json.dumps(metadata)

from scipy.stats import rv_histogram
import uproot




class SwiftTAFinder(TriggerPrimitivesProcessor):


    def __init__(self, df_writer=None, cfg={}):
        super().__init__()

        self.writer = df_writer

        # Placeholder
        self._cfg = default_cfg.copy()
        self._cfg.update(cfg)

        print(self._cfg)
        self.entry_keys = ['event_uid', 'event', 'run', 'subrun']
        self.tawin_keys = self.entry_keys+['TPCSetID', 'ta_win_id']

        # Load bakground window sadc distribution
        ta_win_sadc_dist_file = self._cfg.get('ta_win_sadc_dist_file', None)
        self.bkg_dist=None
        if ta_win_sadc_dist_file:
            print(f"Loading '{ta_win_sadc_dist_file}'")
            with uproot.open(ta_win_sadc_dist_file) as f:
                h1 = f["h1"].to_hist()

                # extract histogram info
                counts = h1.view()
                edges  = h1.axes[0].edges
                self.bkg_dist = rv_histogram((counts, edges))


    def apply_preselection(self, tps_df: TrgDataFrame):

        print(f"Applying '{self._cfg['preselection']}'")
        return tps_df.query(self._cfg['preselection'])
    
         
    def create_windows(self, tps_df: TrgDataFrame):

        win_start = self._cfg.get('ta_win_start', None)
        win_start = win_start if win_start is not None else tps_df.sample_peak.min()
        print(f"Using 'win_start' {win_start}")
        win_size = self._cfg['ta_win_size']

        # Add ta search window identifier
        tps_in_win = tps_df.copy()
        tps_in_win['ta_win_id'] = (tps_in_win.sample_peak - win_start) // win_size
    
        return tps_in_win
    

    def add_window_bkg_sadc(self, ta_window_stats: TrgDataFrame) -> TrgDataFrame:

        print("Adding sadc background offset to tawindows")

        rnd_seed = self._cfg['ta_win_sadc_dist_rdm_seed']
        ta_window_stats['bkg_sadc'] = self.bkg_dist.rvs(size=len(ta_window_stats), random_state=rnd_seed)
        ta_window_stats['sadc'] += ta_window_stats['bkg_sadc']

        return ta_window_stats
    

    def select_windows(self, ta_window_stats: TrgDataFrame):

        sadc_min = self._cfg['ta_inspect_sadc_min']
        sadc_max = self._cfg['ta_inspect_sadc_max']

        ta_window_stats['sadc_window_thres_lo'] = ta_window_stats.sadc > sadc_min
        ta_window_stats['sadc_window_thres_hi'] = ta_window_stats.sadc > sadc_max
        return ta_window_stats
    

    def process_windows(self, ta_wins: TrgDataFrame, tps_df: TrgDataFrame):

        from tqdm import tqdm
        tqdm.pandas(desc="TA Finding")

        # add a _keep flag to TPs belonging to the selected windows
        tps_to_proc = tps_df.join(
            # from the list of selected TAs, create a temprary dataframe with a _keep flag, indexed on self.tawin_keys
            ta_wins.index
                .to_frame(index=False)
                .assign(_keep=True)
                .set_index(self.tawin_keys),
            on=self.tawin_keys)

        # Drop TPs w/o a true _keep flag
        tps_to_proc = tps_to_proc[tps_to_proc._keep == True].drop('_keep', axis=1)

        epsilon=self._cfg['ta_dbscan_epsilon']
        min_neigh=self._cfg['ta_dbscan_min_neigh']

        return (
            tps_to_proc
            .groupby(self.tawin_keys, sort=False)
            .progress_apply(apply_dbscan, epsilon=epsilon, min_samples=min_neigh, include_groups = False)
            .reset_index()
        )


    def process(self, tps_df: TrgDataFrame):
        
        # Bluntly apply the preselection to the entire dataset
        sel_tps = self.apply_preselection(tps_df)

        # Decorate TPs with window ids
        tps_in_wins = self.create_windows(sel_tps)

        # Create window stats
        ta_window_stats = (
            tps_in_wins.groupby(self.tawin_keys, sort=False)
            .agg(
                n_tps=('ta_win_id', "size"),
                sadc=("adc_integral", "sum"),
                channel_std =('channel', 'std'),
                channel_mean =('channel', 'mean'),
                sample_peak_std =('sample_peak', 'std'),
                sample_peak_mean =('sample_peak', 'mean')
            ).fillna(-1)
        )

        print(f"Window stats created ({len(ta_window_stats)} windows)")


        if self._cfg['ta_win_sadc_add_bkg']:
            ta_window_stats = self.add_window_bkg_sadc(ta_window_stats)

        # Add window selection flag (inspect/direct accept)
        ta_window_with_flags = self.select_windows(ta_window_stats)

        if self.writer:
            self.writer.write(ta_window_with_flags.reset_index(), 'ta_win_stats')


        # Select the windoes to be inspected (clustered)
        ta_inspect_windows = ta_window_with_flags.query('sadc_window_thres_lo == True & sadc_window_thres_hi == False')

        # Add cluster information and dbscan lables to windows
        ta_win_clustered = self.process_windows(ta_inspect_windows, tps_in_wins)

        # Create a sub-dataset without cluster details, onlu summary info
        ta_win_cluster_summary = (
            ta_window_with_flags
            .join(
                ta_win_clustered
                .drop(['tp_index', 'dbscan_label'], axis=1)
                .set_index(self.tawin_keys)
                )
            ).fillna(0.0)
        

        # Create an event-dataset with ta-selection counts for the 2 categories
        def count_ta_wins( g: pd.DataFrame, cluster_sadc_thres ):
            return pd.Series({
                'max_win_sadc': g.sadc.max(),
                'max_win_cluster_sadc': g.max_cluster_sadc.max(),
                'tot_n_clusters': g.n_clusters.sum(),
                'num_accept_win': (g.sadc_window_thres_hi == True).sum(),
                'num_inspect_win': ((g.sadc_window_thres_hi == False) & (g.sadc_window_thres_lo == True)).sum(),
                'num_inspect_accept_win': ((g.sadc_window_thres_hi == False) & (g.sadc_window_thres_lo == True) & (g.max_cluster_sadc > cluster_sadc_thres)).sum()
            }) 

        if self.writer:
            self.writer.write(ta_win_cluster_summary.reset_index(), 'ta_win_cluster_stats')

        # Create per_event selection flags
        cluster_sadc_thres=self._cfg['ta_inspect_sadc_cluster_threshold']
        # Create a event selection dataframe by grouping by 
        ta_event_sel = ta_win_cluster_summary.groupby(self.entry_keys, sort=False).apply(count_ta_wins, cluster_sadc_thres=cluster_sadc_thres)
        # Add a global accept flag
        ta_event_sel['accepted'] = (ta_event_sel.num_accept_win > 0 ) | (ta_event_sel.num_inspect_accept_win > 0)

        if self.writer:
            self.writer.write(ta_event_sel.reset_index(), 'ta_event_selection')

        # Create a TP collection with clustering flags (only clustered windows)
        # Select TA windows with >0 clusters and explode the tp_index and dbscan columns
        tp_ids = ta_win_clustered.query('n_clusters > 0')[['tp_index','dbscan_label']].explode(['tp_index','dbscan_label'])

        # Join the dbscan_label to the tp dataset 
        clustered_tps = tps_in_wins.join(tp_ids.set_index(pd.MultiIndex.from_tuples(tp_ids["tp_index"], names=["entry", "subentry"])).drop(columns="tp_index"))
        # clustered_tps = tps_in_wins.join(tp_ids.set_index('tp_index'))
        with pd.option_context("future.no_silent_downcasting", True):
            clustered_tps['dbscan_label'] = clustered_tps.dbscan_label.fillna(-1).astype('int16')


        if self.writer:
            self.writer.write(clustered_tps, 'tps_with_cluster_flags')

        ta_clusters = (
            clustered_tps
            .query('dbscan_label > -1')
            .groupby(self.tawin_keys+['dbscan_label'], sort=False)
            .agg(
                sadc=('adc_integral','sum'),
                num_tps=('dbscan_label', 'count'),
                sample_peak_mean=('sample_peak', 'mean'),
                sample_peak_std = ('sample_peak','std'),
                channel_mean=('channel', 'mean'),
                channel_std=('channel', 'std')
            )
        )
        if self.writer:
            self.writer.write(ta_clusters.reset_index(), 'ta_clusters')

        # import IPython
        # IPython.embed(colors='neutral')


        return ta_event_sel




        

