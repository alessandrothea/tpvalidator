import marimo

__generated_with = "0.23.10"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo

    from rich import print
    from tpvalidator.workspace import TriggerActivityWorkspace

    import mplhep as hep
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import mplhep as hep
    import pathlib
    import hist



@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Data
    """)
    return


@app.cell
def _():
    dataset_dir = pathlib.Path('../../data/vd/1x8x14/vdtf-prep/swift/')

    datasets = {}
    for n in ['eminus', 'muminus', 'gamma', 'proton', 'neutron']:
        datasets[n] = TriggerActivityWorkspace(dataset_dir / f'{n}.root')
    return (datasets,)


@app.cell
def _(datasets):
    em_ws = datasets['eminus']
    mu_ws = datasets['muminus']
    gm_ws = datasets['gamma']
    pt_ws = datasets['proton']
    nt_ws = datasets['neutron']

    return em_ws, gm_ws, mu_ws, nt_ws, pt_ws


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Analysis
    """)
    return


@app.function
def plot_ta_efficiencies(ws):

    ev_sum = ws.event_summary[['event_uid', 'event', 'run', 'subrun', 'tot_visible_energy_rop2']].set_index('event_uid')
    ev_sum = ev_sum.join(ws.mctruths[['event_uid', 'kinetic_energy','x', 'y', 'z']].set_index('event_uid'))
    ev_sum = ev_sum.join(ws.ta_event_selection.drop(columns=['event', 'run', 'subrun']).set_index('event_uid'))
    ev_sum['kinetic_energy'] *= 1000
    
    energy_var = 'kinetic_energy'
    label='$E_{kin}$'
    
    # energy_var = 'tot_visible_energy_rop2'
    # label='$E_{vis}$'
    
    from tpvalidator.analysis.histograms import make_regaxis
    ke_ax = make_regaxis(ev_sum, energy_var, 1, label=label)
    
    
    # ke_ax = hist.axis.Regular(99, 1, 100, name=label, flow=False)
    
    h_ke = hist.Hist(ke_ax, storage=hist.storage.Weight())
    h_ke.fill(ev_sum[energy_var])
    
    h_ke_acc = hist.Hist(ke_ax, storage=hist.storage.Weight())
    h_ke_acc.fill(ev_sum.query('accepted == True')[energy_var])
    
    h_ke_rej = hist.Hist(ke_ax, storage=hist.storage.Weight())
    h_ke_rej.fill(ev_sum.query('accepted == False')[energy_var])
    
    h_ke_dir_acc = hist.Hist(ke_ax, storage=hist.storage.Weight())
    h_ke_dir_acc.fill(ev_sum.query('num_accept_win > 0')[energy_var])
    
    h_ke_insp_acc = hist.Hist(ke_ax, storage=hist.storage.Weight())
    h_ke_insp_acc.fill(ev_sum.query('num_accept_win == 0 & num_inspect_accept_win > 0')[energy_var])
    
    h_ke_insp_rej = hist.Hist(ke_ax, storage=hist.storage.Weight())
    # h_ke_insp_rej.fill(ev_sum.query('num_accept_win == 0 & num_inspect_win > 0 & max_win_cluster_sadc < 7500')[energy_var])
    h_ke_insp_rej.fill(ev_sum.query('num_accept_win == 0 & num_inspect_win > 0 & num_inspect_accept_win == 0')[energy_var])
    
    # xmin, xmax = 0, 10
    xmin, xmax = 0, 50
    xmin, xmax = None, None
    cmap = mpl.colormaps['tab10']
    
    fig,axes = plt.subplots(3,3, figsize=(12,12))
    
    # All events
    ax=axes[0][0]
    hep.histplot(h_ke, ax=ax, color='k')
    ax.grid()
    ax.set_title('all events')
    ax.set_xlim(xmin, xmax)
    
    ax=axes[0][1]
    hep.histplot(h_ke_acc, color=cmap(2), ax=ax)
    ax.grid()
    ax.set_title('accepted - final')
    ax.set_xlim(xmin, xmax)
    
    ax=axes[0][2]
    # hep.histplot(r, ax=ax)
    hep.comp.comparison(h_ke_acc, h_ke, comparison='efficiency', ax=ax, color=cmap(0), linestyle="-")
    ax.grid()
    ax.set_ylim(0,1.1)
    ax.set_title('accepted - efficiency')
    ax.set_xlim(xmin, xmax)
    
    
    ax=axes[1,0]
    hep.histplot(h_ke_rej, color=cmap(3), ax=ax)
    ax.grid()
    ax.set_title('rejected - final')
    ax.set_xlim(xmin, xmax)
    
    ax=axes[1,1]
    hep.histplot(h_ke_dir_acc, color=cmap(2), ax=ax)
    ax.grid()
    ax.set_title('direct accept')
    ax.set_xlim(xmin, xmax)
    
    ax=axes[1,2]
    hep.comp.comparison(h_ke_dir_acc, h_ke, comparison='efficiency', ax=ax, color=cmap(0), linestyle="-")
    ax.grid()
    ax.set_ylim(0,1.1)
    ax.set_title('direct accept - efficiency')
    ax.set_xlim(xmin, xmax)
    
    ax=axes[2,0]
    hep.histplot(h_ke_insp_rej, color=cmap(3), ax=ax)
    ax.grid()
    ax.set_title('inspect reject')
    ax.set_xlim(xmin, xmax)
    
    ax=axes[2,1]
    hep.histplot(h_ke_insp_acc,color=cmap(2),  ax=ax)
    ax.grid()
    ax.set_title('inspect accept')
    ax.set_xlim(xmin, xmax)
    
    
    ax=axes[2,2]
    hep.comp.comparison(h_ke_insp_acc, h_ke, comparison='efficiency', ax=ax, color=cmap(0), linestyle="-")
    # hep.histplot(h_ke_insp_acc/h_ke, ax=ax)
    ax.grid()
    ax.set_title('inspect accept - efficiency')
    ax.set_ylim(0,1.1)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(label)
    
    fig.tight_layout()
    return fig


@app.cell(hide_code=True)
def _(datasets):
    mo.md(rf"""
    # Single Electrons

    - Events {len(datasets['eminus'].event_list)}
    """)
    return


@app.cell
def _(mu_ws):
    _fig, _ax = plt.subplots(figsize=(10,10))
    _fig.tight_layout()


    mu_ws.ta_event_selection.hist(ax=_ax, bins=50)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(em_ws):
    plot_ta_efficiencies(em_ws)
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
    plot_ta_efficiencies(mu_ws)
    return


@app.cell(hide_code=True)
def _(datasets):
    mo.md(rf"""
    # Single Gammas

    - Events {len(datasets['gamma'].event_list)}
    """)
    return


@app.cell
def _(gm_ws):
    plot_ta_efficiencies(gm_ws)
    return


@app.cell(hide_code=True)
def _(datasets):
    mo.md(rf"""
    # Single Proton

    - Events {len(datasets['proton'].event_list)}
    """)
    return


@app.cell
def _(pt_ws):
    plot_ta_efficiencies(pt_ws)
    return


@app.cell(hide_code=True)
def _(datasets):
    mo.md(rf"""
    # Single Neutron

    - Events {len(datasets['neutron'].event_list)}
    """)
    return


@app.cell
def _(nt_ws):
    plot_ta_efficiencies(nt_ws)
    return


if __name__ == "__main__":
    app.run()
