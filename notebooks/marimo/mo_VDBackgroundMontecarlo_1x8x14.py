import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setup
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import uproot
    import logging
    import tpvalidator.workspace as workspace
    import tpvalidator.analyzers.snn as snn

    from rich import print
    from tpvalidator.utils import temporary_log_level, pandas_backend
    from tpvalidator.viz.term import df_to_rich_table
    from tpvalidator.viz.mc import MCPlotter

    from collections import OrderedDict

    return MCPlotter, np, pd, plt, print, snn, workspace


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Code
    """)
    return


@app.cell
def _(np, pd, workspace):
    from typing import Tuple, Optional, Union, Sequence, Dict, List
    from rich.table import Table
    from tpvalidator.detector_geometry import FDVDGeometry_1x8x14

    def make_rates_table_from_workspaces(datasets: Dict[str, workspace.TriggerPrimitivesWorkspace], preselection: str='', per: str='chan', ro_win_len: int=None) -> pd.DataFrame:
        """
        Calculates the TP rates of each workspace in the dataset list.
        The rate is calculated over the chosen unit specified by the `per` argument: channel, crp, tpc or detector.
        The rates are displayed per unit and per view.

        If a preselection is specified, this is applied to all datasets before the rate calculation.
        The readout window lenght, needed for the rate estimate, is passed as argument.
        """
        sampling_period = 5e-07
        num_el_map = {'chan': lambda v: FDVDGeometry_1x8x14.crp_num_chans_by_view_sim(v) * FDVDGeometry_1x8x14.num_crps, 'crp': lambda _: FDVDGeometry_1x8x14.num_crps, 'tpc': lambda _: FDVDGeometry_1x8x14.num_tpcs, 'det': lambda _: 1}
        num_el = num_el_map[per]
        rows = []
        for s, ws in datasets.items():
            num_ev = ws.num_entries
            ro_win_len = ws._extra_info['readout_window'] if ro_win_len is None else ro_win_len
            total_time = num_ev * ro_win_len * sampling_period
            tps_sel = ws.tps.query(preselection) if preselection else ws.tps
            tp_rates = []
            for v in range(3):
                num_els = num_el(v)
                tp_rate = len(tps_sel.query(f'readout_view == {v}')) / total_time / num_els
                tp_rates = tp_rates + [tp_rate]
            rows = rows + [(s, total_time, *tp_rates)]
        _df = pd.DataFrame(np.array(rows, dtype=[('dataset', object), ('time', float), ('U', float), ('V', float), ('X', float)]), columns=['dataset', 'time', 'U', 'V', 'X'])
        return _df

    return Dict, FDVDGeometry_1x8x14, Table


@app.cell
def _(Dict, FDVDGeometry_1x8x14, Table, pd):
    def make_vd_rates_table_2(datasets: Dict[str, pd.DataFrame], num_events, preselection: str='', per: str='chan', title: str='', ro_win_len: int=None, sampling_period=5e-06) -> Table:
        num_el_map = {'chan': lambda v: FDVDGeometry_1x8x14.crp_num_chans_by_view_sim(v) * FDVDGeometry_1x8x14.num_crps, 'crp': lambda _: FDVDGeometry_1x8x14.num_crps, 'tpc': lambda _: FDVDGeometry_1x8x14.num_tpcs, 'det': lambda _: 1}
        num_el = num_el_map[per]
        _t = Table('sample', 'time', 'U', 'V', 'X', title=title)
        for s, _df in datasets.items():
            num_ev = num_events
            ro_win_len = _df.extra_info['readout_window'] if ro_win_len is None else ro_win_len
            total_time = num_ev * ro_win_len * sampling_period
            row = [s, f'{total_time:.3f} s']
            tps_sel = _df.query(preselection) if preselection else _df
            for v in range(3):
                num_els = num_el(v)
                noise_rate = len(tps_sel.query(f'readout_view == {v}')) / total_time / num_els
                row = row + [f'{noise_rate:.2f} Hz']
            _t.add_row(*row)
        return _t

    return


@app.cell
def _(FDVDGeometry_1x8x14, print):
    def _calc_vol(elem_info):
        vol = 1
        for v in ['x', 'y', 'z']:
            _vr = elem_info[f'{v}_range']
            l = _vr['max']-_vr['min']
            print(v, l)
            vol *= l/100.
        return vol


    # vol = 1
    # for v in ['x', 'y', 'z']:
    #     _vr = FDVDGeometry_1x8x14.geo()['tpcs'][0][f'{v}_range']
    #     l = _vr['max']-_vr['min']
    #     print(v, l)
    #     vol *= l/100.

    cryo_vol = _calc_vol(FDVDGeometry_1x8x14.geo()['cryostat'])
    crm_vol = _calc_vol(FDVDGeometry_1x8x14.geo()['tpcs'][0])
    det_vol = crm_vol*FDVDGeometry_1x8x14.num_tpcs
    det_cryo_ratio = det_vol/cryo_vol
    print(f"TPC vol = {crm_vol}")
    print(f"Det vol = {det_vol}")

    print(f"Cryo-vol = {cryo_vol}")

    print(f"Det/Cryo = {det_cryo_ratio}")
    return (det_cryo_ratio,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data
    """)
    return


@app.cell
def _():
    import tpvalidator.datasetloader as dsl

    dataset_name = 'radbkg'
    datasets = dsl.load('data/vd/1x8x14/old_detsim', dataset_name)
    rad_ws=datasets[dataset_name]
    return (rad_ws,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tests
    """)
    return


@app.cell
def _(rad_ws):
    (rad_ws.mctruths.t.max()-rad_ws.mctruths.t.min())*1e-9
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Rates
    """)
    return


@app.cell
def _(Table, print, rad_ws):
    _t = Table('generator name', 'id', title='Background generators')
    for gid, n in rad_ws.mctruth_blocks_map.items():
        _t.add_row(n, str(gid))
    print(_t)
    return


@app.cell
def _(rad_ws):
    rad_ws.mctruths.extra_info
    return


@app.cell
def _(Table, det_cryo_ratio, print, rad_ws):
    ro_win = rad_ws.mctruths.extra_info['readout_window']
    num_entries = rad_ws.mctruths.extra_info['num_entries']

    simulated_time = 2 * ro_win * 0.5e-6 * num_entries
    _t = Table('generator name', 'entries', 'rate [Hz]', title='Backgrounds generators by activity')
    _part_by_gen = sorted([(n, _df) for n, _df in rad_ws.mctruths.groupby('generator_name')], reverse=True, key=lambda x: len(x[1]))
    for _gen_id, _df in _part_by_gen:
        _t.add_row(_gen_id, str(len(_df)), f'{len(_df) * det_cryo_ratio / simulated_time :.2e}')
    print(_t)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MC particles spectra

    This is useful to tell the expected energy of clusters from background
    """)
    return


@app.cell
def _(MCPlotter, rad_ws):
    mcp = MCPlotter(rad_ws)

    from rich.console import Console

    mcp.make_generators_table()

    return (mcp,)


@app.cell
def _(mcp):
    mcp.make_generator_rates_table()
    return


@app.cell
def _(mcp):
    mcp.plot_distributions(bins=50)
    return


@app.cell
def _(mcp):
    mcp.plot_ke_spectra_by_pdg()
    return


@app.cell
def _(mcp):
    mcp.plot_ke_spectra_by_generator()
    return


@app.cell
def _(mcp):
    mcp.plot_generator_activity(norm='counts', pdg_id=11, figsize=(8,8))
    return


@app.cell
def _(mcp):
    mcp.plot_generator_activity(norm='rate', pdg_id=22, figsize=(8,8))
    return


@app.cell
def _(plt, rad_ws):
    _fig, _axes = plt.subplots(2, 1)
    _n_bins = rad_ws.mcparticles.query('pdg == 11').truth_block_id.nunique()
    rad_ws.mcparticles.query('pdg == 11').truth_block_id.hist(bins=_n_bins, ax=_axes[0])
    rad_ws.mcparticles.query('truth_block_id == 9').t.hist(ax=_axes[1])
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TP Spectra, by generator
    """)
    return


@app.cell
def _(rad_ws, snn):
    # Create a TP preselection - only TPs with peak within the range where backtracking works

    rad_sel = snn.TPSignalNoiseSelector(rad_ws.tps)
    return (rad_sel,)


@app.cell
def _(np, plt, rad_sel):
    # Plot rates distributions for the top backgrounds
    def plot_tpcounts_by_source(tps, ax, var='adc_integral', binsize=8):
        num_bkg = 10
        tps_by_gen = sorted([(n, _df) for n, _df in tps.groupby('bt_generator_name')], reverse=True, key=lambda x: len(x[1]))
        top_by_gen = tps_by_gen[:num_bkg]
        bin_max = max([_df[var].max() for _, _df in top_by_gen])
        _bins = np.arange(0, int(bin_max), binsize)
        for n, _df in top_by_gen:
            _ax.hist(_df[var], bins=_bins, histtype='step', label=n if n else 'noise')
        _ax.set_yscale('log')
        _ax.set_xlabel(var)
        _ax.set_ylabel('counts')
    _fig, _axes = plt.subplots(1, 3, figsize=(20, 6))
    ro_view = 2
    _ax = _axes[0]  # ax.legend()
    plot_tpcounts_by_source(rad_sel.all_by_view[ro_view], var='adc_integral', binsize=32, ax=_ax)
    _ax = _axes[1]
    plot_tpcounts_by_source(rad_sel.all_by_view[ro_view], var='adc_peak', binsize=16, ax=_ax)
    _ax = _axes[2]
    # fig, axes = plt.subplots(1,3, figsize=(20,6))
    # ro_view = 2
    # sot_thres = 7
    # ax = axes[0]
    # plot_tpcounts_by_source(rad_sel.all_by_view[ro_view].query(f'samples_over_threshold > {sot_thres}'), var='adc_integral', binsize=64, ax =ax)
    # ax = axes[1]
    # plot_tpcounts_by_source(rad_sel.all_by_view[ro_view].query(f'samples_over_threshold > {sot_thres}'), var='adc_peak', binsize=16, ax =ax)
    # ax = axes[2]
    # plot_tpcounts_by_source(rad_sel.all_by_view[ro_view].query(f'samples_over_threshold > {sot_thres}'), var='samples_over_threshold', binsize=1, ax =ax)
    # fig = plot_tpcounts_by_source(rad_sel.all_by_view[ro_view].query('samples_over_threshold > 8')f, var=var, binsize=32)
    # var = 'adc_peak'
    # fig = plot_tpcounts_by_source(rad_sel.all_by_view[ro_view].query('samples_over_threshold > 8'), var=var, binsize=16)
    # var = 'samples_over_threshold'
    # fig = plot_tpcounts_by_source(rad_sel.all_by_view[ro_view].query('samples_over_threshold > 8'), var=var, binsize=1)
    plot_tpcounts_by_source(rad_sel.all_by_view[ro_view], var='samples_over_threshold', binsize=1, ax=_ax)
    return


@app.cell
def _(mct_1, plt):
    mct_1.hist(figsize=(10, 10), bins=100)
    plt.gcf().tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Wall gammas origin checks
    """)
    return


@app.cell
def _(mct_1):
    bkg_gammas = mct_1.query('generator_name == "CavernwallGammasAtLAr1x8x6"')
    bkg_gammas[['x', 'y', 'z']].hist(bins=100, figsize=(10, 10))
    return


if __name__ == "__main__":
    app.run()
