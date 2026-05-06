import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ar39 noise — interactive report
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Introduction
    This notebook aims at characterizing the detector response in simulation using an 1r39-only sample.

    by answering the following questions

    1. Is the rate of trigger primitives from Ar39 compatible with the expectations

    - Ar39 is uniformely generaterd inside the detector
    - Ar39 spectrum has a 1 MeV endpoint,
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Preamble
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    import functools
    import logging
    import yaml

    import matplotlib.pyplot as plt
    from pathlib import Path

    import tpvalidator
    import tpvalidator.analyzers.snn as snn
    from rich import print

    mo.md("Imports completed")
    return Path, plt, snn, tpvalidator, yaml


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## Data
    """)
    return


@app.cell
def _(mo):
    dataset_dir = mo.ui.text(
        value="data/vd/1x8x14",
        label="Dataset directory",
        full_width=True,
    )
    dataset_dir
    return (dataset_dir,)


@app.cell
def _(Path, dataset_dir, mo, tpvalidator, yaml):
    _pkg_root = Path(tpvalidator.__file__).parents[2]
    _dir = Path(dataset_dir.value)
    if not _dir.is_absolute():
        _dir = _pkg_root / _dir

    _cfg_file = _dir / "datasets.yaml"
    mo.stop(not _cfg_file.exists(), mo.md(f"*No `datasets.yaml` found in `{_dir}`.*"))

    with open(_cfg_file) as _f:
        _cfg = yaml.safe_load(_f)

    _names = list(_cfg.get("datasets_spec", {}).keys())
    _default = 'ar39_5e_00' if 'ar39_5e_00' in _names else _names[0]

    dataset_name = mo.ui.dropdown(options=_names, value=_default, label="Dataset")
    dataset_name
    return (dataset_name,)


@app.cell
def _(dataset_dir, dataset_name, mo):
    import tpvalidator.datasetloader as dsl

    mo.stop(not dataset_name.value)

    _datasets = dsl.load(dataset_dir.value, [dataset_name.value])
    ws = _datasets[dataset_name.value]
    ws.info
    return (ws,)


@app.cell
def _(mo, snn, ws):
    mo.stop(ws is None)

    all_tps  = snn.TPSignalNoiseSelector(ws.tps)
    tp_ana = snn.TPSignalNoiseAnalyzer(all_tps, signal_name='Ar39')
    return all_tps, tp_ana


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## Noise and Ar39 signal — ADC distributions
    """)
    return


@app.cell
def _(mo, snn, ws):
    mo.stop(not ws.rawdigits_hists, mo.md("*No waveform file loaded — skipping ADC distribution plot.*"))
    snn.draw_signal_and_noise_adc_distros(ws)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ADC samples distributions per plane (integrated on the full dataset)

    - Blue: ADC distribution of channels where IDE are present
    - Orange: ADC distribution of channels where IDE are absent

    The 3σ and 5σ lines calculated on noise-only waveforms (no IDEs)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## Ar39 TPs — point of origin
    """)
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_tp_sig_origin_2d_dist()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Point of origin of TPs (`bt_primary_x`) tagged as signal, i.e. matching at least one `SimIDE` object
    """)
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_tp_sig_drift_depth_dist()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Point of origin in the drift for TPs tagged as signal, i.e. matched to at least 1 IDE object.**

    All 3 distributions have a maximum in the x=(100,200) range, 1 meter from the anode.
    """)
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_tp_sig_drift_depth_dist(weight_by="adc_integral")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When weighted by `adc_integral`, the distributions acquire a linear trend. Charge capture by TPs is more efficient close to the anode and less efficient close to the cathode due to diffusion that spread the charge over multiple samples.
    The effect is lower significance of the signal peak in the waveform and loss of TPs.

    Them, the TP count plot (previous cell), can be explained in terms of lateral diffusion. An Ar39 deposit spreads over neighbouring channel generating multiple TPs.
    """)
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_tps_per_track_in_drift_grid(figsize=(12, 10), sharex=True, sharey=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The TP multiplicity per track id increases the closer the origin is to the cathode.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ==TODO==

    Count number of track ids per drift bin
    Should be lower closer to the cathode
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## Timing of TPs tagged as signal and noise
    """)
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_tp_start_sample_dist()
    return


@app.cell
def _(plt, ws):
    _fig, _ax = plt.subplots()
    ws.simides.timestamp.plot.hist(bins=1000, ax=_ax)
    _ax.set_xlabel('time')
    _ax.set_ylabel('counts')
    _ax.set_title('IDEs time of arrival at the anode')
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Top: Distribution of TP time by plane
    - Bottom: Distribution of IDEs time of arrival at the anode (CRP)
    <!-- - **NOTE**: The IDEs time distribution shows 2 issues:
        1. A spike at `time=62k`, well beyond the readout window end
        2. No IDEs beyond `time=8200` -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Example event: signal and noise TPs
    """)
    return


@app.cell
def _(mo):
    entry_number = mo.ui.number(start=0, stop=1000, value=3, label="Entry number")
    entry_number
    return (entry_number,)


@app.cell
def _(all_tps, entry_number, mo, snn):
    _panels = []
    for _thresh in (26, 36, 46, 56):
        _ana = snn.TPSignalNoiseAnalyzer(all_tps.query(f'adc_peak > {_thresh}'))
        _fig = _ana.draw_tp_event(entry_number.value)
        _panels.append(mo.as_html(_fig))

    mo.vstack([
        mo.hstack(_panels[:2]),
        mo.hstack(_panels[2:]),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Channel and `sample_peak` of TPs for the selected event
    - Incremental `adc_peak` cuts are applied (from a) to d)) to show the distribution of TPs at higher adc_peak
    <!-- - **NOTE**: A lack of signal TPs is evident at `sample_peak > 8200` for all planes. In this region noise TPs
      have a harder spectrum: a fraction survives the peakADC cut appearing very similar to signal TPs,
      suggesting they may be untagged signal TPs. -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## Basic TP distributions
    """)
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_tp_signal_noise_dist()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## TP `adc_peak` distributions across drift depth
    """)
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_variable_in_drift_grid('adc_peak', downsampling=10, sharex=True, sharey=True, figsize=(12, 10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## TP `samples_over_threshold` distributions across drift depth
    """)
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_variable_in_drift_grid('samples_over_threshold', downsampling=1, log=False, sharey=True, figsize=(12, 10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## TP `adc_integral` distributions across drift depth
    """)
    return


@app.cell
def _(mo, tp_ana):
    mo.md("Bins of bt_x (collection)")
    tp_ana.draw_variable_in_drift_grid('adc_integral', downsampling=100, sharey=True, figsize=(12, 10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Stacked TP distributions
    """)
    return


@app.cell
def _(mo, tp_ana):
    mo.hstack([
        mo.as_html(tp_ana.draw_variable_drift_stack('adc_peak',
                                     downsampling=5, n_x_bins=4, log=True, figsize=(5, 4))),
        mo.as_html(tp_ana.draw_variable_drift_stack('samples_over_threshold',
                                     downsampling=1, n_x_bins=4, log=False, figsize=(5, 4))),
        mo.as_html(tp_ana.draw_variable_drift_stack('adc_integral',
                                     downsampling=5, n_x_bins=4, log=True, figsize=(5, 4))),
    ])

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Comparison of `adc_peak`, `samples_over_threshold` and `adc_integral` in 4 regions of trueX for the collection plane.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## Effects of `adc_peak` cuts on TP distributions
    """)
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_variable_cut_sequence('adc_peak',list(range(26, 50, 5)), log=True, figsize=(15, 10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Effects of `samples_over_threshold` cuts on TP distributions
    """)
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_variable_cut_sequence('samples_over_threshold',
                      list(range(0, 10, 2)), log=True, figsize=(15, 10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Effects of `adc_integral` cuts on TP distributions
    """)
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_variable_cut_sequence('adc_integral',
                      list(range(0, 500, 100)), log=True, figsize=(15, 10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Threshold scan — noise rejection efficiency
    """)
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_threshold_scan('adc_peak', list(range(26, 120, 1)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## `adc_integral` cuts noise rejection efficiency
    """)
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_threshold_scan('samples_over_threshold', list(range(0, 10, 2)))
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_threshold_scan('adc_integral', list(range(0, 500, 100)))
    return


if __name__ == "__main__":
    app.run()
