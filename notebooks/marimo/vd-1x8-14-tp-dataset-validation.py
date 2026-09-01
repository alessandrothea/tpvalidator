import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo


    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import tpvalidator.workspace as workspace
    import tpvalidator.utils as utils
    import tpvalidator.analysis.snn as snn

    from rich import print
    from tpvalidator.viz.display import TriggerPrimitivesEventViewer
    from tpvalidator.viz.backtracker import BackTrackerPlotter


@app.cell
def _():
    import tpvalidator.datacatalogue as dsl
    datasets = dsl.load('data/vd/1x8x14/vdtf-prep/tps')
    return (datasets,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Analysis
    """)
    return


@app.function
def plot_tps_dists(ws, query:str=None):
    cls = list(ws.tps.columns)
    for c in ['event', 'run', 'subrun', 'event_uid']:
        cls.remove(c)

    tps = ws.tps if query == None else ws.tps.query(query)
    figsize=(10,8)
    bins=50
    
    panels = []
    panels.append(mo.md("## Main Variables"))
    
    _fig,_ax=plt.subplots(1,1, figsize=figsize)
    tps[[c for c in cls if not c.startswith('bt_')]].hist(ax=_ax, bins=bins)
    _fig.tight_layout()
    panels.append(mo.as_html(_fig))
    
    panels.append(mo.md("## Backtracking Variables"))
    _fig,ax=plt.subplots(1,1, figsize=figsize)
    tps[[c for c in cls if c.startswith('bt_')]].hist(ax=ax, bins=bins)
    _fig.tight_layout()
    panels.append(mo.as_html(_fig))

    return panels


@app.function
def plop_bt_origin(ws):
    fig,axes = plt.subplots(1,3, figsize=(12,4))
    ax = axes[0]
    ws.tps.query('bt_is_signal == 1').plot.scatter(x='bt_primary_z', y='bt_primary_y', s=0.2, alpha=0.1, ax=ax)
    ax = axes[1]
    ws.tps.query('bt_is_signal == 1').plot.scatter(x='bt_primary_z', y='bt_primary_x', s=0.2, alpha=0.1, ax=ax)
    ax = axes[2]
    ws.tps.query('bt_is_signal == 1').plot.scatter(x='bt_primary_y', y='bt_primary_x', s=0.2, alpha=0.1, ax=ax)
    fig.suptitle("Z-aligned muons, 10 GeV (100 )")
    fig.tight_layout()
    return fig


@app.cell
def _(datasets):
    em_ws = datasets['eminus']
    mu_ws = datasets['muminus']
    gm_ws = datasets['gamma']
    pt_ws = datasets['proton']
    nt_ws = datasets['neutron']
    return em_ws, gm_ws, mu_ws, nt_ws, pt_ws


@app.cell(hide_code=True)
def _(datasets):
    mo.md(rf"""
    # Single Electrons

    - Events {len(datasets['eminus'].event_list)}
    """)
    return


@app.cell
def _(nt_ws):
    _fig, _ax = plt.subplots(1,1, figsize=(10,10))
    nt_ws.simide_summary.hist(bins=50, ax=_ax)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(em_ws):
    plot_tps_dists(em_ws, query='bt_is_signal==True')
    return


@app.cell
def _(em_ws):
    plop_bt_origin(em_ws)
    return


@app.cell(hide_code=True)
def _(datasets):
    mo.md(rf"""
    # Single Muons

    - Events {len(datasets['muminus'].event_list)}
    """)
    return


@app.cell
def _(mu_ws):
    plot_tps_dists(mu_ws, query='bt_is_signal==True')
    return


@app.cell
def _(mu_ws):
    plop_bt_origin(mu_ws)
    return


@app.cell(hide_code=True)
def _(datasets):
    mo.md(rf"""
    # Single Gammas

    - Events {len(datasets['gamma'].event_list)}
    """)
    return


@app.cell(hide_code=True)
def _(gm_ws):
    plot_tps_dists(gm_ws, query='bt_is_signal==True')
    return


@app.cell
def _(gm_ws):
    plop_bt_origin(gm_ws)
    return


@app.cell(hide_code=True)
def _(datasets):
    mo.md(rf"""
    # Single Protons

    - Events {len(datasets['proton'].event_list)}
    """)
    return


@app.cell
def _(pt_ws):
    plot_tps_dists(pt_ws, query='bt_is_signal==True')
    return


@app.cell
def _(pt_ws):
    plop_bt_origin(pt_ws)
    return


@app.cell(hide_code=True)
def _(datasets):
    mo.md(rf"""
    # Single Neutrons

    - Events {len(datasets['neutron'].event_list)}
    """)
    return


@app.cell
def _(nt_ws):
    plot_tps_dists(nt_ws, query='bt_is_signal==True')
    return


@app.cell
def _(nt_ws):
    plop_bt_origin(nt_ws)
    return


if __name__ == "__main__":
    app.run()
