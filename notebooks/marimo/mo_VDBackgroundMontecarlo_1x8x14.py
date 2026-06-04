import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import uproot
    import logging
    import tpvalidator.workspace as workspace
    import tpvalidator.analysis.snn as snn

    from rich import print
    from tpvalidator.utils import temporary_log_level, pandas_backend
    from tpvalidator.viz.textual import dataframe_to_rich_table
    from tpvalidator.viz.mc import MCPlotter
    from tpvalidator.detector_geometry import FDVDGeometry_1x8x14


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # Geometry VD 1x8x14

    | Name | Value |
    | -- | -- |
    | Cryo volume | {FDVDGeometry_1x8x14.cryo_volume():.2f} $m^3$|
    | TPC volume | {FDVDGeometry_1x8x14.tpc_volume():.2f} $m^3$|
    | CRP volume | {4*FDVDGeometry_1x8x14.tpc_volume():.2f} $m^3$|
    | Detector volume | {FDVDGeometry_1x8x14.det_volume():.2f} $m^3$|
    | Detector/Cryostat ratio | {FDVDGeometry_1x8x14.det_volume()/FDVDGeometry_1x8x14.cryo_volume():.2f}|
    | TPC Anode surface | {FDVDGeometry_1x8x14.anode_surface():.2f} $m^2$
    | CRP surface | {4*FDVDGeometry_1x8x14.anode_surface():.2f} $m^2$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Data
    """)
    return


@app.cell(hide_code=True)
def _():
    import tpvalidator.datacatalogue as dsl

    dataset_name = 'radbkg'
    datasets = dsl.load('data/vd/1x8x14/old_detsim', dataset_name)
    rad_ws=datasets[dataset_name]
    return (rad_ws,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Rates
    """)
    return


@app.cell
def _(rad_ws):
    rad_ws.mctruths.extra_info
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # MC particles spectra

    This is useful to tell the expected energy of clusters from background
    """)
    return


@app.cell
def _(rad_ws):
    mcp_truths = MCPlotter(rad_ws, collection='mctruths')
    mcp_parts = MCPlotter(rad_ws, collection='mcparticles')
    return mcp_parts, mcp_truths


@app.cell
def _(mcp_truths):
    mcp_truths.make_generators_table()
    return


@app.cell
def _(mcp_truths):
    mo.md(f"""
    Simulated time: {mcp_truths.simulated_mc_time():.2f} sec
    """)
    return


@app.cell
def _(mcp_truths):
    mcp_truths.make_generator_rates_table()
    return


@app.cell
def _(mcp_parts):
    mcp_parts.make_generator_rates_table()
    return


@app.cell
def _(mcp_truths):
    mcp_truths.plot_distributions(bins=50)
    return


@app.cell
def _(mcp_truths):
    mcp_truths.plot_distributions('CavernwallGammasAtLAr1x8x14', bins=50)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Energy Spectra
    """)
    return


@app.cell
def _(mcp_parts, mcp_truths):
    _panels = []
    _fig = mcp_truths.plot_ke_spectra_by_pdg()
    _panels.append(mo.as_html(_fig))

    _fig = mcp_parts.plot_ke_spectra_by_pdg()
    _panels.append(mo.as_html(_fig))

    mo.vstack([
        mo.md('## $E_{kin}$ by PDG id'),
        mo.hstack(_panels)
    ])

    return


@app.cell(hide_code=True)
def _(mcp_parts, mcp_truths):
    _panels = []
    _fig = mcp_truths.plot_ke_spectra_by_generator()
    _panels.append(mo.as_html(_fig))

    _fig = mcp_parts.plot_ke_spectra_by_generator()
    _panels.append(mo.as_html(_fig))

    mo.hstack(_panels),
    return


@app.cell
def _(mcp_truths):
    mcp_truths.plot_generator_activity(norm='counts', figsize=(8,8))

    return


@app.cell(hide_code=True)
def _(mcp_truths):
    _panels = []
    _fig = mcp_truths.plot_generator_activity(norm='counts', pdg_id=11, figsize=(8,8))
    _panels.append(mo.as_html(_fig))

    _fig = mcp_truths.plot_generator_activity(norm='rate', pdg_id=22, figsize=(8,8))
    _panels.append(mo.as_html(_fig))

    mo.hstack(_panels),
    return


@app.cell
def _(mcp_truths):
    _panels = []
    for v in ['x', 'y', 'z']:
        _fig = mcp_truths.plot_generator_origin(v, bin_width_cm=10, c_scale='log')
        _panels.append(mo.as_html(_fig))
    
    mo.vstack([
        mo.md("""
            ## MC particle origin by generator

            - Origin of MC particles in x, y, z binned by generator
        """),
        mo.hstack(_panels)
    ])
    return


@app.cell
def _(mcp_truths):
    mo.vstack([
        mo.md('## Wall gammas origin in the cryostat and energy spectrum'),
        mcp_truths.plot_generator_pos_ke('CavernwallGammasAtLAr1x8x14')
    ])

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
