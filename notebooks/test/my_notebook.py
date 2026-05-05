import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    hello
    """)
    return


@app.cell
def _():
    import matplotlib

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Hello 2
    """)
    return


@app.cell
def _():

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import logging
    import tpvalidator.workspace as workspace
    # import tpvalidator.utils as utils
    import tpvalidator.analyzers.snn as snn

    from rich import print
    from tpvalidator.utils import temporary_log_level, pandas_backend

    return (plt,)


@app.cell
def _():
    import tpvalidator.datasetloader as dsl
    datasets = dsl.load('data/vd/1x8x14')
    return (datasets,)


@app.cell
def _(datasets):
    ws=datasets['ar39']
    return (ws,)


@app.cell
def _(plt, ws):
    fig, axes = plt.subplots(2,2, squeeze=False, figsize=(10,7))

    ax = axes[0,0]
    # ws.simides.query('event==1').timestamp.plot.hist(bins=1000, ax=ax)
    ws.simides.timestamp.plot.hist(bins=1000, ax=ax)
    ax.set_xlabel('time')
    ax.set_title('IDEs time')

    ax = axes[0,1]
    bins=list(range(8000, 8501, 10))
    ws.simides.query('timestamp > 8000 & timestamp < 8500').timestamp.plot.hist(bins=bins, ax=ax)
    ax.set_xlabel('time')
    ax.set_title('IDEs time - zoom around 8200 (all events)')

    early_ides = ws.simides.query('timestamp <  500 & readout_view ==2')

    ax = axes[1,0]
    early_ides.timestamp.plot.hist(bins=250, ax=ax)
    ax.set_xlabel('time')
    ax.set_title('IDEs time - time < 500 (all events)')

    ax = axes[1,1]
    early_ides.timestamp.plot.hist(bins=250, weights=early_ides.numelectrons, ax=ax)
    ax.set_xlabel('time')
    ax.set_title('IDEs time weighted by $n_{electrons}$- time < 500 (all events)')

    fig.tight_layout()

    fig
    return


if __name__ == "__main__":
    app.run()
