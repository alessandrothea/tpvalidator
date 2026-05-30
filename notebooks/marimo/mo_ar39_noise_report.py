import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo


    import functools
    import logging
    import yaml

    import matplotlib.pyplot as plt
    from pathlib import Path

    import tpvalidator
    import tpvalidator.analysis.snn as snn
    from rich import print
    import tpvalidator.datacatalogue as dctlg
    from tpvalidator.viz.backtracker import BackTrackerPlotter


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Signal/Noise — interactive report
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Introduction
    This notebook aims at characterizing the detector response in simulation using background samples.

    <!-- by answering the following questions

    1. Is the rate of trigger primitives from Ar39 compatible with the expectations

    - Ar39 is uniformely generaterd inside the detector
    - Ar39 spectrum has a 1 MeV endpoint, -->
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Preamble
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ---

    ## Data
    """)
    return


@app.cell
def _():
    dataset_dir = mo.ui.text(
        value="data/vd/1x8x14/3sig",
        label="Dataset directory",
        full_width=True,
    )
    dataset_dir
    return (dataset_dir,)


@app.cell
def _(dataset_dir):


    # _pkg_root = Path(tpvalidator.__file__).parents[2]
    # _dir = Path(dataset_dir.value)
    # if not _dir.is_absolute():
    #     _dir = _pkg_root / _dir

    # _cfg_file = _dir / "datasets.yaml"
    # mo.stop(not _cfg_file.exists(), mo.md(f"*No `datasets.yaml` found in `{_dir}`.*"))

    # with open(_cfg_file) as _f:
    #     _cfg = yaml.safe_load(_f)

    # _names = list(_cfg.get("datasets_spec", {}).keys())
    print(dataset_dir)
    _names = dctlg.list_datasets(dataset_dir.value)
    _default = 'ar39_5e_00' if 'ar39_5e_00' in _names else _names[0]

    dataset_name = mo.ui.dropdown(options=_names, value=_default, label="Dataset")
    dataset_name
    return (dataset_name,)


@app.cell
def _(dataset_dir, dataset_name):

    mo.stop(not dataset_name.value)

    _datasets = dctlg.load(dataset_dir.value, [dataset_name.value])
    ws = _datasets[dataset_name.value]
    ws.info
    return (ws,)


@app.cell
def _(ws):
    mo.stop(ws is None)

    all_tps  = snn.TPSignalNoiseSelector(ws.tps)
    tp_ana = snn.TPSignalNoiseAnalyzer(all_tps, signal_name='Ar39')
    btp = BackTrackerPlotter(ws)
    return all_tps, btp, tp_ana


@app.cell
def _(ws):
    ws.tps
    return


@app.cell(hide_code=True)
def _(dataset_name):
    mo.md(f"""
    ---

    ## **Noise and '{dataset_name.value}' ADC distributions**
    """)
    return


@app.cell
def _(ws):
    mo.stop(not ws.rawdigits_hists, mo.md("*No waveform file loaded — skipping ADC distribution plot.*"))
    snn.draw_signal_and_noise_adc_distros(ws)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ADC samples distributions per plane (integrated on the full dataset)

    - Blue: ADC distribution of channels where IDE are present
    - Orange: ADC distribution of channels where IDE are absent

    The 3σ and 5σ lines calculated on noise-only waveforms (no IDEs)
    """)
    return


@app.cell
def _(btp, ws):
    if len(ws.rawdigis_events) != 0:
        _ev_uid = ws.rawdigits_tree.event_list().iloc[0].to_dict()
        _some_collection_tps = (
            ws.tps
            .query('(event=={event}) & (run=={run}) & (subrun=={subrun})'.format(**_ev_uid))
            .query('readout_plane_id==2 & samples_over_threshold<4 & bt_is_signal==1')
            .iloc[0:6]
        )
        btp.plot_tps_vs_ides(tps=_some_collection_tps)
    return


@app.cell(hide_code=True)
def _(dataset_name):
    mo.md(f"""
    ---

    ## **'{dataset_name.value}' TPs — point of origin**
    """)
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_tp_sig_origin_2d_dist()
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    Point of origin of TPs (`bt_primary_x`) tagged as signal, i.e. matching at least one `SimIDE` object
    """)
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_tp_sig_drift_depth_dist()
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### **Point of origin in the drift for TPs tagged as signal, i.e. matched to at least 1 IDE object.**

    All 3 distributions have a maximum in the x=(100,200) range, 1 meter from the anode.
    """)
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_tp_sig_drift_depth_dist(weight_by="adc_integral")
    return


@app.cell(hide_code=True)
def _():
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
def _():
    mo.md(r"""
    The TP multiplicity per track id increases the closer the origin is to the cathode.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ==TODO==

    Count number of track ids per drift bin
    Should be lower closer to the cathode
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ---
    # Signal / Noise TPs
    Validation of detector simulation and backtracking
    ## **Time distributions**
    """)
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_tp_start_sample_dist()
    return


@app.cell
def _(ws):
    _fig, _ax = plt.subplots()
    ws.simides.timestamp.plot.hist(bins=1000, ax=_ax)
    _ax.set_xlabel('time')
    _ax.set_ylabel('counts')
    _ax.set_title('IDEs time of arrival at the anode')
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Top: Distribution of TP time by plane
    - Bottom: Distribution of IDEs time of arrival at the anode (CRP)
    <!-- - **NOTE**: The IDEs time distribution shows 2 issues:
        1. A spike at `time=62k`, well beyond the readout window end
        2. No IDEs beyond `time=8200` -->
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## **TP Distribution in channel and time**
    """)
    return


@app.cell
def _():
    entry_number = mo.ui.number(start=0, stop=1000, value=3, label="Entry number")
    entry_number
    return (entry_number,)


@app.cell
def _(all_tps, entry_number):
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
def _():
    mo.md(r"""
    - Channel and `sample_peak` of TPs for the selected event
    - Incremental `adc_peak` cuts are applied (from a) to d)) to show the distribution of TPs at higher adc_peak
    <!-- - **NOTE**: A lack of signal TPs is evident at `sample_peak > 8200` for all planes. In this region noise TPs
      have a harder spectrum: a fraction survives the peakADC cut appearing very similar to signal TPs,
      suggesting they may be untagged signal TPs. -->
    """)
    return


@app.cell
def _(btp):
    btp.draw_nel_eff_by_plane()
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ---

    ## **Distributions of main TP variables**
    """)
    return


@app.cell
def _(tp_ana):
    mo.vstack([
        mo.md("### **ROP 0 (Induction U)**"),
        tp_ana.draw_tp_signal_noise_dist(roview=0),
        mo.md("### **ROP 1 (Induction V)**"),
        tp_ana.draw_tp_signal_noise_dist(roview=1),
        mo.md("### **ROP 2 (Collection)**"),
        tp_ana.draw_tp_signal_noise_dist(roview=2),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ---

    ## **TP `adc_peak` distributions across drift depth**
    """)
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_variable_in_drift_grid('adc_peak', bin_size=10, sharex=True, sharey=True, figsize=(12, 10))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## **TP `samples_over_threshold` distributions across drift depth**
    """)
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_variable_in_drift_grid('samples_over_threshold', bin_size=1, log=False, sharey=True, figsize=(12, 10))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## **TP `adc_integral` distributions across drift depth**
    """)
    return


@app.cell
def _(tp_ana):
    mo.md("Bins of bt_x (collection)")
    tp_ana.draw_variable_in_drift_grid('adc_integral', bin_size=100, sharey=True, figsize=(12, 10))
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Stacked TP distributions
    """)
    return


@app.cell(hide_code=True)
def _(tp_ana):
    mo.vstack([
        mo.hstack([
            mo.as_html(tp_ana.draw_variable_drift_stack('adc_peak', roview=_roview,
                                         bin_size=5, n_x_bins=4, log=True, figsize=(5, 4))),
            mo.as_html(tp_ana.draw_variable_drift_stack('samples_over_threshold', roview=_roview,
                                         bin_size=1, n_x_bins=4, log=False, figsize=(5, 4))),
            mo.as_html(tp_ana.draw_variable_drift_stack('adc_integral', roview=_roview,
                                         bin_size=5, n_x_bins=4, log=True, figsize=(5, 4))),
        ]) for _roview in range(3)
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    Comparison of `adc_peak`, `samples_over_threshold` and `adc_integral` in 4 regions of `bt_primary_x` for the collection plane.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ---

    ## Effects of `adc_peak` cuts on TP distributions
    """)
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_variable_cut_sequence('adc_peak',list(range(26, 70, 5)), log=True, figsize=(15, 10))
    return


@app.cell(hide_code=True)
def _():
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
def _():
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
def _():
    mo.md("""
    ## Threshold scan — noise rejection efficiency
    """)
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_threshold_scan('adc_peak', plane_id=2, thresholds=list(range(26, 120, 1)))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## `adc_integral` cuts noise rejection efficiency
    """)
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_threshold_scan('samples_over_threshold', plane_id=2, thresholds=list(range(0, 10, 2)))
    return


@app.cell
def _(tp_ana):
    tp_ana.draw_threshold_scan('adc_integral', plane_id=2, thresholds=list(range(0, 500, 100)))
    return


if __name__ == "__main__":
    app.run()
