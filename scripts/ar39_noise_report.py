#!/usr/bin/env python
"""Generate a PDF validation report for Ar39 noise studies."""

import json
# import functools
import logging

import matplotlib.pyplot as plt

from fpdf import FontFace, TextStyle, Align
from pathlib import Path
from rich import print
from rich.logging import RichHandler

import click

import tpvalidator.workspace as workspace
import tpvalidator.analyzers.snn as snn
from tpvalidator.utils import temporary_log_level
from tpvalidator.report.portfolio import Portfolio
from tpvalidator.report.pdf import ReportPDF, load_report_fonts


FORMAT = "%(message)s"
logging.basicConfig(
    level="WARN", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger('ar39_report')

# ---------------------------------------------------------------------------
# Report theme
# ---------------------------------------------------------------------------

_COLOR_DEEP_BLUE   = "#122a5dff"
_COLOR_ORANGE = "#f06000"
_COLOR_TEAL = "#3779AF"

_TAG_STYLES = {
    "h1":   TextStyle(color=_COLOR_DEEP_BLUE,   font_size_pt=28, font_family='Raleway'),
    "h2":   TextStyle(color=_COLOR_ORANGE, font_size_pt=24, font_family='Raleway'),
    "h3":   TextStyle(color=_COLOR_TEAL, font_size_pt=20, font_family='Raleway'),
    "code": FontFace(family='SourceCodePro'),
}

# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def prepare_figures(ws: workspace.TriggerPrimitivesWorkspace, output_dir: Path) -> Portfolio:
    print("Generate figures")

    tps = snn.TPSignalNoiseSelector(ws.tps)
    tp_ana = snn.TPSignalNoiseAnalyzer(tps, signal_name='Ar39')

    # tps = snn.TPSignalNoiseSelector(ws.tps[(ws.tps.sample_start > 100) & (ws.tps.sample_start < 8100)])
    # tp_ana = snn.TPSignalNoiseAnalyzer(tps, signal_name='Ar39')

    pf = Portfolio(output_dir, 'ar39_5e_00')

    if ws.rawdigits_hists:
        pf.add_figure('adc_dist', snn.draw_signal_and_noise_adc_distros, tpws=ws)

    pf.add_figure('xyz_pos_dist_all_tps', tp_ana.draw_tp_sig_origin_2d_dist, fmt='png')
    pf.add_figure('x_pos_dist_all_tps', tp_ana.draw_tp_sig_drift_depth_dist)
    pf.add_figure('x_pos_weighted_dist_all_tps', tp_ana.draw_tp_sig_drift_depth_dist, weight_by='adc_integral')
    pf.add_figure('x_pos_tp_mult_by_tracked', tp_ana.draw_tps_per_track_in_drift_grid, sharex=True, sharey=True, figsize=(12, 10))
    pf.add_figure('start_time_dist_all_tps', tp_ana.draw_tp_start_sample_dist)

    for threshold in (26, 36, 46, 56):
        x = snn.TPSignalNoiseAnalyzer(tps.query(f'adc_peak > {threshold}'))
        pf.add_figure(f'event_10_peak{threshold}_all_tps', x.draw_tp_event, fmt='png', entry=3)

    def plot_ides_time():
        fig, ax = plt.subplots()
        ws.simides.timestamp.plot.hist(bins=1000, ax=ax)
        ax.set_xlabel('time')
        ax.set_ylabel('counts')
        return fig

    pf.add_figure('ides_time_dist_all_tps', plot_ides_time)
    # pf.add_figure('start_time_dist', tp_ana.draw_tp_start_sample_dist)
    pf.add_figure('vs_elnoise_var_dist', tp_ana.draw_tp_signal_noise_dist)

    pf.add_figure('peakadc_dist_in_drift_bins', tp_ana.draw_variable_in_drift_grid, var='adc_peak', downsampling=10, sharex=True, sharey=True, figsize=(12, 10))
    pf.add_figure('tot_dist_in_drift_bins', tp_ana.draw_variable_in_drift_grid, var='samples_over_threshold', downsampling=1, log=False, sharex=True, sharey=True, figsize=(12, 10))
    pf.add_figure('sadc_dist_in_drift_bins', tp_ana.draw_variable_in_drift_grid, var='adc_integral', downsampling=100, sharex=True, sharey=True, figsize=(12, 10))

    pf.add_figure('peak_dist_stack_in_drift_bins', tp_ana.draw_variable_drift_stack, var='adc_peak', downsampling=5, n_x_bins=5, log=True, figsize=(5, 4))
    pf.add_figure('tot_dist_stack_in_drift_bins', tp_ana.draw_variable_drift_stack, var='samples_over_threshold', downsampling=1, n_x_bins=5, log=False, figsize=(5, 4))
    pf.add_figure('sadc_dist_stack_in_drift_bins', tp_ana.draw_variable_drift_stack, var='adc_integral', downsampling=5, n_x_bins=5, log=True, figsize=(5, 4))

    cuts = list(range(26, 50, 5))
    pf.add_figure('dists_with_peakadc_cuts', tp_ana.draw_variable_cut_sequence, var='adc_peak', thresholds=cuts, log=True, figsize=(15, 10))
    cuts = list(range(0, 10, 2))
    pf.add_figure('dists_with_tot_cuts', tp_ana.draw_variable_cut_sequence, var='samples_over_threshold', thresholds=cuts, log=True, figsize=(15, 10))
    cuts = list(range(0, 500, 100))
    pf.add_figure('dists_with_sadcs_cuts', tp_ana.draw_variable_cut_sequence, var='adc_integral', thresholds=cuts, log=True, figsize=(15, 10))

    thresholds = list(range(26, 120, 1))
    pf.add_figure('peak_thresh_scan_perf', tp_ana.draw_threshold_scan, var='adc_peak', thresholds=thresholds)
    thresholds = list(range(0, 10, 2))
    pf.add_figure('tot_thresh_scan_perf', tp_ana.draw_threshold_scan, var='samples_over_threshold', thresholds=thresholds)
    thresholds = list(range(0, 500, 100))
    pf.add_figure('sadc_thresh_scan_perf', tp_ana.draw_threshold_scan, var='adc_integral', thresholds=thresholds)

    return pf


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def write_report(pf: Portfolio, notes: dict, report_file: Path) -> None:
    
    # Default settings
    md_defaults = {'tag_styles': _TAG_STYLES, 'li_prefix_color': _COLOR_ORANGE}

    pdf = ReportPDF(md_defaults=md_defaults, orientation="landscape", format="A4")
    load_report_fonts(pdf)
    pdf.set_font("Lato", size=15)

    ts = _TAG_STYLES
    li_color = _COLOR_ORANGE

    pdf.add_slide("Characterisation of the VD detector response to Ar39")
    pdf.write_markdown(f"""
    This report characterizes the detector response in simulation to Ar39 .
    * Ar39 is uniformely generaterd inside the detector
    * Ar39 spectrum has a 1 MeV endpoin
    * Ar39 is the closest physics process to the TP threshold

    Questions addressed in the report
    1. What is the noise/Ar39 ADC distribution
    1. Are the Ar39 that produce TPs uniformely distributed in the detector?
    1. What are the effects of diffusion on Ar39? (Relevant since Ar39 is close to TP threshold)
    1. What is the backtracking efficienct for the Ar39 sample
    1. Is the rate of trigger primitives from Ar39 compatible with expectations?  
    """)    
    # Page 1 — title & data provenance
    pdf.add_slide("Data sample details")
    pdf.write_markdown(f"""
        * Datasets folder: **{notes['dataset_dir_path']}**
        * Dataset name: **{notes['dataset_name']}**
        * Detector geometry: **{notes['ws_info']['geo']['detector']}**
        * MC generator(s): **{', '.join(notes['mc_generator_labels'])}**
        * TP generation details:
            * Algorithm: **{notes['tpg_info']['tool']}**
            * Threshold U: **{notes['tpg_info']['threshold_tpg_plane0']}**
            * Threshold V: **{notes['tpg_info']['threshold_tpg_plane1']}**
            * Threshold X/Z: **{notes['tpg_info']['threshold_tpg_plane2']}**
        * Events: **{notes['event_begin']}-{notes['event_end']}**
        """,
    )

    # pdf.add_page()
    # pdf.write_markdown("# Introduction", tag_styles=ts)

    # pdf.write_markdown(f"""
    # This notebook characterizes the detector response in simulation to Ar39-only .

    # 1. Is the rate of trigger primitives from Ar39 compatible with the expectations

    #     * Ar39 is uniformely generaterd inside the detector
    #     * Ar39 spectrum has a 1 MeV endpoint,
                       
    # """)


    # Page 2 — ADC distributions
    pdf.add_slide("Noise and AR39 signal distribution")
    pdf.image(pf.path('adc_dist'), w=pdf.epw)
    pdf.write_markdown(r"""
        ADC samples distributions per plane (integrated on the full dataset)

        * Blue: ADC distribution of channels where IDE are present
        * Orange: ADC distribution of channels where IDE are absent

        The 3σ and 5σ lines calculated on noise-only waveforms (no IDEs)
    """, tag_styles=ts, li_prefix_color=li_color)

    # Page 3 — TP origin
    pdf.add_page()
    pdf.write_markdown("# **Ar39 TPs origin**", tag_styles=ts)
    pdf.image(pf.path('xyz_pos_dist_all_tps', 'png'), h=pdf.eph * 0.9, x=Align.L)
    pdf.set_xy(pdf.eph + 30, 30)
    pdf.write_markdown("Point of origin of TPs (`bt_primary_x`) tagged as signal, i.e. matching an IDE", tag_styles=ts)

    # Page 4 — example event with incremental peak-ADC cuts
    pdf.add_page()
    pdf.write_markdown("# **Example event: signal and noise TPs**", tag_styles=ts)
    pdf.move_cursor(dy=5)
    img_w = pdf.epw // 2
    with pdf.local_context(font_size_pt=10):
        pdf.cell(w=img_w, text="a) adc_peak>26", markdown=True, align=Align.C)
        pdf.cell(w=img_w, text="b) adc_peak>36", markdown=True, align=Align.C)
    pdf.ln()

    x0, y0 = pdf.get_x(), pdf.get_y()
    pdf.image(pf.path('event_10_peak26_all_tps', 'png'), w=img_w, x=Align.L)
    pdf.set_xy(x0 + img_w, y0)
    ii = pdf.image(pf.path('event_10_peak36_all_tps', 'png'), w=img_w)
    pdf.set_xy(x0, y0 + ii.rendered_height)
    x0, y0 = pdf.get_x(), pdf.get_y()
    pdf.image(pf.path('event_10_peak46_all_tps', 'png'), w=img_w)
    pdf.set_xy(x0 + img_w, y0)
    ii = pdf.image(pf.path('event_10_peak56_all_tps', 'png'), w=img_w)
    pdf.set_xy(x0, y0 + ii.rendered_height)
    with pdf.local_context(font_size_pt=10):
        pdf.cell(w=img_w, text="c) adc_peak>46", markdown=True, align=Align.C)
        pdf.cell(w=img_w, text="d) adc_peak>56", markdown=True, align=Align.C)
    pdf.ln()
    pdf.write_markdown(f"""
        * Channel and `sample_peak` of TPs in event 10
        * Incremental adc_peak cuts are applied (from a) to d)) to show the distribution of TPs at higher adc_peak.
        """, tag_styles=ts, li_prefix_color=li_color)
        # * NOTE: A lack of signal TPs is evident at `sample_peak > 8200` for all planes. In this region noise TPs have a harder spectrum:
        #     A fraction survives the peakADC cut appearing very similar to signal TPs, suggesting that they may be untagged signal TPs.

    # Page 5 — TP origin vs drift depth
    pdf.add_page()
    pdf.write_markdown("# **TP point of origin vs drift depth for signal TPs**", tag_styles=ts)
    pdf.image(pf.path('x_pos_dist_all_tps'), w=pdf.epw, x=Align.C)
    pdf.write_markdown("""
                        Point of origin in the drift for TPs tagged as signal, i.e. matched to at least 1 IDE object.
                        All distributions have a maximum around x=(100,200), 1 meter from the anode.
                        One possible explanation could be lateral diffusion. At ~1m, the point-like Ar39 energy deposit 
                        diffuses over the strips neighbouring the main one.
        """,
        tag_styles=ts)

    # Page 6 — weighted adc_integral vs drift
    pdf.add_page()
    pdf.write_markdown("# **Ar39 - total adc_integral sum vs drift depth for signal TPs**", tag_styles=ts)
    pdf.image(pf.path('x_pos_weighted_dist_all_tps'), w=pdf.epw, x=Align.C)
    pdf.write_markdown("""
        When weighted by `adc_integral`, the distributions acquire a linear trend. Charge capture by TPs is more efficient close to the anode and less efficient close to the cathode due to diffusion that spread the charge over multiple samples.
        The effect is lower significance of the signal peak in the waveform and loss of TPs.
        Them, the TP count plot (previous cell), can be explained in terms of lateral diffusion. An Ar39 deposit spreads over neighbouring channel generating multiple TPs.
        """)
    
    # Page 7:
    pdf.add_page()
    pdf.write_markdown("# TP multiplicity per track id vs drift depth", tag_styles=ts)
    pdf.image(pf.path('x_pos_tp_mult_by_tracked'), w=0.95*pdf.eph, x=Align.C)
    pdf.write_markdown("""
                       Ar39 deposits are expected to span one or two channels. This is visible in the x-bin closest to CRPs.
                       The further the origin is from the CRP, the higher the number of TPs, indicating that lateral
                       diffusion pushes charge on neighbouring channels.
        """,
        tag_styles=ts)

    # Page 7 — TP timing
    pdf.add_page()
    pdf.write_markdown("# **Timing of TPs tagged as signal and noise**", tag_styles=ts)
    pdf.image(pf.path('start_time_dist_all_tps'), w=pdf.epw * 0.9, x=Align.L)
    ii = pdf.image(pf.path('ides_time_dist_all_tps'), h=pdf.eph * 0.3, x=Align.L)
    pdf.move_cursor(dx=ii.rendered_width, dy=-ii.rendered_height + 10)
    pdf.write_markdown("""
        * Top figures: Distribution of TP time by plane
        * Bottom figure: Distribution of IDEs time of arrival at the anode (CRP)
    """, tag_styles=ts)
        # * OLD NOTE: The IDEs time distribution shows 2 issues:
        #     1. A spike at `time=62k`, well beyond the readout window end,
        #     2. No IDEs beyond `time=8200`.



    # # Page 8 — TP timing (clean)
    # pdf.add_page()
    # pdf.write_markdown("# **Timing of TPs tagged as signal and noise (clean)**", tag_styles=ts)
    # pdf.image(pf.path('start_time_dist'), w=pdf.epw * 0.9, x=Align.L)
    # pdf.write_markdown("""
    #     * Distribution of TP startTime after applying cleanup
    #     * `sample_start > 100 && sample_start < 8200`
    # """, tag_styles=ts)

    # Page 8 — basic TP distributions
    pdf.add_page()
    pdf.write_markdown("# **Basic TP distribution**", tag_styles=ts)
    pdf.image(pf.path('vs_elnoise_var_dist'), w=pdf.epw, x=Align.C)

    # Page 13 — stacked distributions
    pdf.add_slide('Stacked TP distributions', subtitle='Bins of `bt_primary_x` - Plane 2 (X, collection)')
    pdf.set_y(pdf.eph // 3)
    x, y = pdf.get_x(), pdf.get_y()
    pdf.image(pf.path('peak_dist_stack_in_drift_bins'), w=pdf.epw // 3)
    pdf.set_y(y)
    pdf.image(pf.path('tot_dist_stack_in_drift_bins'), w=pdf.epw // 3, x=x + pdf.epw // 3)
    pdf.set_y(y)
    pdf.image(pf.path('sadc_dist_stack_in_drift_bins'), w=pdf.epw // 3, x=x + 2 * pdf.epw // 3)
    pdf.write_markdown(
        "Comparison of adc_peak, samples_over_threshold and adc_integral in 5 regions of 'bt_primary_x` for the collection plane.",
        tag_styles=ts)

    # Pages 10-12 — distributions in drift bins
    for name, title, subtitle in [
        ('peakadc_dist_in_drift_bins', 'Distribution of TP adc_peak across the drift','Bins of `bt_primary_x` - Plane 2 (X, collection)'),
        ('tot_dist_in_drift_bins',     'Distribution of TP samples_over_threshold across the drift','Bins of `bt_primary_x` - Plane 2 (X, collection)'),
        ('sadc_dist_in_drift_bins',    'Distribution of TP adc_integral across the drift', 'Bins of `bt_primary_x` - Plane 2 (X, collection)'),
    ]:
        pdf.add_slide(title, subtitle=subtitle)
        pdf.image(pf.path(name), w=pdf.eph, x=Align.C)



    # Pages 14-16 — cut sequence effects
    for name, title in [
        ('dists_with_peakadc_cuts', 'Effects of adc_peak cuts on distributions'),
        ('dists_with_tot_cuts',     'Effects of samples_over_threshold cuts on distributions'),
        ('dists_with_sadcs_cuts',   'Effects of SADCs cuts on distributions'),
    ]:
        pdf.add_slide(f"{title}","Plane 2 (X, collection)")
        pdf.image(pf.path(name), h=pdf.eph * 0.85, x=Align.L)
        pdf.set_xy(pdf.eph + 30, 30)

    # Pages 17-19 — threshold scan performance
    for name, title in [
        ('peak_thresh_scan_perf', 'PeakADC cut noise rejection efficiency'),
        ('tot_thresh_scan_perf',  'samples_over_threshold cut noise rejection efficiency'),
        ('sadc_thresh_scan_perf', 'adc_integral cut noise rejection efficiency'),
    ]:
        pdf.add_slide(f"{title}")
        pdf.image(pf.path(name), w=pdf.epw * 0.95, x=Align.C)
        pdf.set_xy(pdf.eph + 30, 30)

    pdf.output(str(report_file))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("dataset_dir_path", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option('-d', '--dataset-name', 'dataset_name', type=str, default='ar39_5e_00',
              help="Path to the raw-waveform ROOT file.")
@click.option('-i', '--interactive', is_flag=True, default=False,
              help="Drop into an IPython shell after loading data.")
@click.option('--figs/--no-figs', 'make_figures', default=True, show_default=True,
              help="Generate figures (disable to re-use a previous run's figures).")
@click.option('--report/--no-report', 'make_report', default=True, show_default=True,
              help="Assemble and write the PDF report.")
@click.option('-o', 'output_dir', type=click.Path(file_okay=False), default='./reports/ar39/',
              show_default=True)
def cli(dataset_dir_path, dataset_name, output_dir,
        make_figures, make_report, interactive):

    tp_algorith = 'tpmakerTPCSimpleThreshold::TriggerPrimitiveMaker'

    report_dir = Path(output_dir)
    figures_dir = report_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    notes_file = figures_dir / 'notes.json'

    pf = Portfolio(figures_dir, dataset_name)

    if make_figures:
        import tpvalidator.datasetloader as dsl
        datasets = dsl.load(dataset_dir_path, [dataset_name])
        ws=datasets[dataset_name]

        events = ws.event_summary.event.unique()
        notes = {
            'dataset_dir_path': dataset_dir_path,
            'dataset_name': dataset_name,
            'ws_info': ws.info,
            'mc_generator_labels': list(ws.mctruth_blocks_map.values()),
            'tpg_info': ws.info['tpg'][tp_algorith],
            'event_begin': int(min(events)),
            'event_end': int(max(events)),
        }
        with open(notes_file, 'w') as fp:
            json.dump(notes, fp, indent=4)

        if interactive:
            import IPython
            IPython.embed(colors='neutral')

        with temporary_log_level(log, logging.INFO):
            pf = prepare_figures(ws, figures_dir)

    if make_report:
        with open(notes_file, 'r') as fp:
            notes = json.load(fp)
        write_report(pf, notes, report_dir / "ar39_report.pdf")


if __name__ == '__main__':
    cli()
