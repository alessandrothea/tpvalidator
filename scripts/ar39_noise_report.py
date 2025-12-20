#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import mistletoe as mt
import uproot
import textwrap
import logging
import tpvalidator.mcprod.workspace as workspace
import tpvalidator.utilities as utils
import tpvalidator.mcprod.analyzers.snn as snn

from rich import print
from rich.logging import RichHandler
from tpvalidator.utilities import temporary_log_level, subplot_autogrid
from tpvalidator.histograms import uproot_hist_mean_std
from io import BytesIO


from fpdf import FPDF, HTML2FPDF, FontFace, TextStyle, Align
from pathlib import Path


from tpvalidator.portfolio import ReportPDF, Portfolio


FORMAT = "%(message)s"
logging.basicConfig(
    level="WARN", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)


import functools

log_prep = logging.getLogger('prepare_figures')
def prepare_figures(ws: workspace.TriggerPrimitivesWorkspace, output_dir: Path) -> dict:
    print("Generate figures")

    # Analyze the entire dataset
    all_tps = snn.TPSignalNoiseSelector(ws.tps)
    alltp_ana = snn.TPSignalNoiseAnalyzer(all_tps, sig_label='Ar39')

    # create analyzers
    tps = snn.TPSignalNoiseSelector(ws.tps[(ws.tps.sample_start >100) & (ws.tps.sample_start <8100)])
    tp_ana = snn.TPSignalNoiseAnalyzer(tps, sig_label='Ar39')

    pf = Portfolio(output_dir, 'ar39')

    if ws.rawdigis_events:
        pf.add_figure('adc_dist', functools.partial(snn.draw_signal_and_noise_adc_distros, ws))


    # 2 Ar39 point of origin on the xy, xz and yz planes
    pf.add_figure('xyz_pos_dist_all_tps', alltp_ana.draw_tp_sig_origin_2d_dist, fmt='png')

    # 3 Ar39 point of origin on the xy, xz and yz planes
    pf.add_figure('x_pos_dist_all_tps', alltp_ana.draw_tp_sig_drift_depth_dist)
    pf.add_figure('x_pos_weighted_dist_all_tps', functools.partial(alltp_ana.draw_tp_sig_drift_depth_dist, weight_by="adc_integral"))

    # Distribution of signat tps time in the drift direction
    pf.add_figure('start_time_dist_all_tps', alltp_ana.draw_tp_start_sample_dist)


    x = snn.TPSignalNoiseAnalyzer(all_tps.query('adc_peak > 26'))
    pf.add_figure('event_10_peak26_all_tps', functools.partial(x.draw_tp_event, 10), fmt='png')
    x = snn.TPSignalNoiseAnalyzer(all_tps.query('adc_peak > 36'))
    pf.add_figure('event_10_peak36_all_tps', functools.partial(x.draw_tp_event, 10), fmt='png')
    x = snn.TPSignalNoiseAnalyzer(all_tps.query('adc_peak > 46'))
    pf.add_figure('event_10_peak46_all_tps', functools.partial(x.draw_tp_event, 10), fmt='png')
    x = snn.TPSignalNoiseAnalyzer(all_tps.query('adc_peak > 56'))
    pf.add_figure('event_10_peak56_all_tps', functools.partial(x.draw_tp_event, 10), fmt='png')

    # Plot ides time distribution
    def plot_ides_time():
        fig, ax = plt.subplots()
        ws.ides.timestamp.plot.hist(bins=1000, ax=ax)
        ax.set_xlabel('time')
        ax.set_ylabel('counts')
        return fig

    pf.add_figure('ides_time_dist_all_tps', plot_ides_time)

    # Draw TP start time distribution after cleaning
    pf.add_figure('start_time_dist', tp_ana.draw_tp_start_sample_dist)

    # Draw signal and noise distributions
    pf.add_figure('vs_elnoise_var_dist', tp_ana.draw_tp_signal_noise_dist)

    # Draw grid of tp adc_peak, samples_over_threshold and adc_integral dist in bins of depth
    pf.add_figure('peakadc_dist_in_drift_bins', functools.partial(tp_ana.draw_variable_in_drift_grid, 'adc_peak', downsampling=10, sharex=True, sharey=True, figsize=(12,10)))
    pf.add_figure('tot_dist_in_drift_bins', functools.partial(tp_ana.draw_variable_in_drift_grid, 'samples_over_threshold', downsampling=1, log=False, sharey=True, figsize=(12,10)))
    pf.add_figure('sadc_dist_in_drift_bins', functools.partial(tp_ana.draw_variable_in_drift_grid, 'adc_integral', downsampling=100, sharey=True, figsize=(12,10)))

    # Draw grid of tp adc_peak, samples_over_threshold and adc_integral dist in bins of depth
    pf.add_figure('peak_dist_stack_in_drift_bins', functools.partial(tp_ana.draw_variable_drift_stack, 'adc_peak', downsampling=5, n_x_bins=4, log=True, figsize=(5,4)))
    pf.add_figure('tot_dist_stack_in_drift_bins', functools.partial(tp_ana.draw_variable_drift_stack, 'samples_over_threshold', downsampling=1, n_x_bins=4, log=False, figsize=(5,4)))
    pf.add_figure('sadc_dist_stack_in_drift_bins', functools.partial(tp_ana.draw_variable_drift_stack, 'adc_integral', downsampling=5, n_x_bins=4, log=True, figsize=(5,4)))

    # Draw the impact of cuts on TP distributions
    cuts = [t for t in range(26, 50, 5)]
    pf.add_figure('dists_with_peakadc_cuts', functools.partial( tp_ana.draw_variable_cut_sequence, 'adc_peak', cuts, log=True, figsize=(15, 10)))
    cuts = [t for t in range(0,10,2)]
    pf.add_figure('dists_with_tot_cuts', functools.partial( tp_ana.draw_variable_cut_sequence, 'samples_over_threshold', cuts, log=True, figsize=(15, 10)))
    cuts = [t for t in range(0, 500, 100)]
    pf.add_figure('dists_with_sadcs_cuts', functools.partial( tp_ana.draw_variable_cut_sequence, 'adc_integral', cuts, log=True, figsize=(15, 10)))


    # Draw the impact of cuts on TP distributions
    thresholds = [t for t in range(26, 120, 1)]
    pf.add_figure('peak_thresh_scan_perf', functools.partial( tp_ana.draw_threshold_scan, 'adc_peak', thresholds))
    thresholds = [t for t in range(0,10,2)]
    pf.add_figure('tot_thresh_scan_perf', functools.partial( tp_ana.draw_threshold_scan, 'samples_over_threshold', thresholds))
    thresholds = [t for t in range(0, 500, 100)]
    pf.add_figure('sadc_thresh_scan_perf', functools.partial( tp_ana.draw_threshold_scan, 'adc_integral', thresholds))

    return




def write_report(figures_dir, report_file):
    # Creater report file
    pdf = ReportPDF(orientation="landscape", format="A4")

    pdf.add_font('Raleway', '', '/Users/ale/Library/Fonts/Raleway-Regular.ttf')
    pdf.add_font('Raleway', 'B', '/Users/ale/Library/Fonts/Raleway-Bold.ttf')
    pdf.add_font('OpenSans', '', '/Library/Fonts/OpenSans_Regular.ttf')
    pdf.add_font('OpenSans', 'B', '/Library/Fonts/OpenSans_Bold.ttf')
    pdf.add_font('Roboto', '', '/Library/Fonts/Roboto_Regular.ttf')
    pdf.add_font('Roboto', 'B', '/Library/Fonts/Roboto_Regular.ttf')
    pdf.add_font('Lato', '', '/Users/ale/Library/Fonts/Lato-Regular.ttf')
    pdf.add_font('Lato', 'B', '/Users/ale/Library/Fonts/Lato-Bold.ttf')
    pdf.add_font('Consolas', '', '/Users/ale/Library/Fonts/Consolas.ttf')
    pdf.add_font('Consolas', 'B', '/Users/ale/Library/Fonts/Consolas Bold.ttf')
    pdf.add_font('SourceCodePro', '', '/Users/ale/Library/Fonts/SourceCodePro-Medium.ttf')
    pdf.add_font('SourceCodePro', 'B', '/Users/ale/Library/Fonts/SourceCodePro-Semibold.ttf')
    
    pdf.set_font("Lato", size=15)

    color_blue = "#093fb5ff"
    color_orange = "#f06000"

    tag_styles={
        "h1": FontFace(color=color_blue, size_pt=28, family='Raleway'),
        "h2": FontFace(color=color_orange, size_pt=24, family='Raleway'),
        "code": FontFace(family='Consolas'),

    }

    with open(figures_dir / 'notes.json', 'r') as fp:
        notes = json.load(fp)



    # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **A first look at Ar39 in VD simulation**", tag_styles=tag_styles)
    pdf.write_markdown(f"""
                        ## Data sample details

                        * TP data file: **{notes['tp_file_path']}**
                        * Waveforms data: **{notes['wf_file_path']}**
                        * Detector geometry: **{notes['ws_info']['geo']['detector']}**
                        * MC generator(s): **{', '.join(notes['mc_generator_labels'])}**
                        * TP generation details:
                            * Algorithm: **{notes['tpg_info']['tool']}**
                            * Threshold U: **{notes['tpg_info']['threshold_tpg_plane0']}**
                            * Threshold V: **{notes['tpg_info']['threshold_tpg_plane1']}**
                            * Threshold X/Z: **{notes['tpg_info']['threshold_tpg_plane2']}**
                        * Events: **{notes['event_begin']}-{notes['event_end']}**
                        """, tag_styles=tag_styles, li_prefix_color=color_orange)
    
    # ---------------------------------------------------------------------
    # TODO: add TOC
    # pdf.add_page()
    # pdf.write_markdown("# **Table of content**", tag_styles=tag_styles)
    # pdf.write_html("<toc>")

    # Page 2
    # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **Noise and AR39 signal distribution**", tag_styles=tag_styles, li_prefix_color=color_orange)

    pdf.image( figures_dir / 'ar39_adc_dist.svg', w=pdf.epw)
    # with pdf.text_columns() as cols:

    pdf.write_markdown(r"""
                ADC samples distributions per plane (integrated on the full dataset)
                        
                * Blue: ADC distribution of channels where IDE are present
                * Orange: ADC distribution of channels where IDE are absent
                        
                The 3σ and 5σ lines calculated on noise-only waveforms (no IDEs)
    """, tag_styles=tag_styles, li_prefix_color=color_orange)

    # Page 3
    # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **Ar39 TPs origin**", tag_styles=tag_styles)

    pdf.image( figures_dir / 'ar39_xyz_pos_dist_all_tps.png', h=pdf.eph*0.9, x=Align.L)

    pdf.set_xy(pdf.eph+30, 30)
    pdf.write_markdown("""
        Point of origin of TPs (trueX) tagged as signal, i.e. matching an IDE
    """, tag_styles=tag_styles)


    # Page 4
    # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **Example event: signal and noise TPs**", tag_styles=tag_styles)
    # pdf.write_markdown("**Event 10**", tag_styles=tag_styles)
    
    # pdf.debug_grid()
    # pdf.draw_margins()

    # pdf.mark_cursor(text="A")
    pdf.move_cursor(dy=5)
    img_w = pdf.epw//2
    with pdf.local_context(font_size_pt = 10):
        pdf.cell(w=img_w, text="a) adc_peak>26", markdown=True, align=Align.C)
        pdf.cell(w=img_w, text="b) adc_peak>36", markdown=True, align=Align.C)

    # pdf.mark_cursor(text="C")
    pdf.ln()
    # pdf.mark_cursor(text="D")


    x0, y0 = pdf.get_x(), pdf.get_y()
    ii = pdf.image( figures_dir / 'ar39_event_10_peak26_all_tps.png', w=img_w, x=Align.L,)
    # pdf.mark_cursor(text="I1")

    pdf.set_xy(x0+img_w, y0)
    # pdf.move_cursor(dx=img_w)

    # pdf.mark_cursor(text="I2")

    ii = pdf.image( figures_dir / 'ar39_event_10_peak36_all_tps.png', w=img_w)
    # pdf.mark_cursor(text="I3")

    pdf.set_xy(x0, y0+ii.rendered_height)
    x0, y0 = pdf.get_x(), pdf.get_y()

    ii = pdf.image( figures_dir / 'ar39_event_10_peak46_all_tps.png', w=img_w)
    # pdf.mark_cursor(text="I4")
    pdf.set_xy(x0+img_w, y0)
    ii = pdf.image( figures_dir / 'ar39_event_10_peak56_all_tps.png', w=img_w)
    pdf.set_xy(x0, y0+ii.rendered_height)

    with pdf.local_context(font_size_pt = 10):

        pdf.cell(w=img_w, text="c) adc_peak>46", markdown=True, align=Align.C)
        pdf.cell(w=img_w, text="d) adc_peak>56", markdown=True, align=Align.C)

    pdf.ln()
    # pdf.mark_cursor(text="I5")

    pdf.write_markdown(f"""
                * Channel and `sample_peak` of TPs in event 10
                * Incremental adc_peak cuts are applied (from a) to d)) to show the distribution of TPs at higher adc_peak.
                * NOTE: A lack of signal TPs is evident at `sample_peak > 8200` for all planes. In this region noise TPs have a harder spectrum:
                    A fraction survives the prakADC cut appearing very similar to signal TPs, suggesting that they may be untagged signal TPs.
                """, tag_styles=tag_styles, li_prefix_color=color_orange)




    # ---------------------------------------------------------------------
    pdf.add_page()

    pdf.write_markdown("# **TP point of origin vs drift depth for signal TPs**", tag_styles=tag_styles)

    pdf.image( figures_dir / 'ar39_x_pos_dist_all_tps.svg', w=pdf.epw, x=Align.C)
    pdf.write_markdown(
        """Point of origin in the drift for TPs tagged as signal, i.e. matched to at least 1 IDE object.
            All 3 distributions have a maximum in the x=(100,200) range, 1 meter from the anode.
        """, tag_styles=tag_styles)
    

    # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **Ar39  - total adc_integral sum vs drift depth for signal TPs**", tag_styles=tag_styles)

    pdf.image( figures_dir / 'ar39_x_pos_weighted_dist_all_tps.svg', w=pdf.epw, x=Align.C)



    # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **Timing of TPs tagged as signal and noise**", tag_styles=tag_styles)
    ii = pdf.image( figures_dir / 'ar39_start_time_dist_all_tps.svg', w=pdf.epw*0.9, x=Align.L)
    ii = pdf.image( figures_dir / 'ar39_ides_time_dist_all_tps.svg', h=pdf.eph*0.3, x=Align.L)
    pdf.move_cursor(dx=ii.rendered_width, dy=-ii.rendered_height+10)
    # pdf.set_xy(pdf.epw//3+30, 2*pdf.eph//3+30)
    pdf.write_markdown("""
                * Top figures: Distribution of TP time by plane
                * Bottom figure: Distribution of IDEs time of arrival at the anode (CRP)
                * NOTE: The IDEs time distribution shows 2 issues:
                    1. A spike at `time=62k`, well beyond the readout window end,
                    2. No IDEs beyond `time=8200`.
    """, tag_styles=tag_styles)


    # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **Timing of TPs tagged as signal and noise (clean)**", tag_styles=tag_styles)
    pdf.image( figures_dir / 'ar39_start_time_dist.svg', w=pdf.epw*0.9, x=Align.L)
    pdf.write_markdown("""
                * Distribution of TP startTime after applying cleanup
                * `sample_start > 100 && sample_start < 8200`
    """, tag_styles=tag_styles)

    # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **Basic TP distribution**", tag_styles=tag_styles)
    pdf.image( figures_dir / 'ar39_vs_elnoise_var_dist.svg', w=pdf.epw, x=Align.C)
    # pdf.write_html("""
    #                <ul>
    #                <li> Distribution of TP startTime after applying cleanup
    #                <li> startT > 100 && startT < 8200
    #                </ul>

    # """, tag_styles=tag_styles)

    # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **Distribution of TP adc_peak across the drift**\nBins of trueX - plane 2 (collection)", tag_styles=tag_styles)
    pdf.image( figures_dir / 'ar39_peakadc_dist_in_drift_bins.svg', w=pdf.eph, x=Align.C)


    # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **Distribution of TP samples_over_threshold across the drift**\nBins of trueX - plane 2 (collection)", tag_styles=tag_styles)
    pdf.image( figures_dir / 'ar39_tot_dist_in_drift_bins.svg', w=pdf.eph, x=Align.C)

    # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **Distribution of TP adc_integral across the drift**\nBins of trueX - plane 2 (collection)", tag_styles=tag_styles)
    pdf.image( figures_dir / 'ar39_sadc_dist_in_drift_bins.svg', w=pdf.eph, x=Align.C)


    # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **Stacked TP distributions**\nBins of trueX - plane 2 (collection)", tag_styles=tag_styles)

    pdf.set_y(pdf.eph//3)
    x = pdf.get_x()
    y = pdf.get_y()
    pdf.image( figures_dir / 'ar39_peak_dist_stack_in_drift_bins.svg', w=pdf.epw//3)
    pdf.set_y(y)
    pdf.image( figures_dir / 'ar39_tot_dist_stack_in_drift_bins.svg', w=pdf.epw//3, x=x+pdf.epw//3)
    pdf.set_y(y)
    pdf.image( figures_dir / 'ar39_sadc_dist_stack_in_drift_bins.svg', w=pdf.epw//3, x=x+2*pdf.epw//3)
    pdf.write_markdown("""
                Comparison of adc_peak, samples_over_threshold and adc_integral for in 4 regions of trueX for the collection plane.
    """, tag_styles=tag_styles)
    # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **Effects of adc_peak cuts on distributions**", tag_styles=tag_styles)

    pdf.image( figures_dir / 'ar39_dists_with_peakadc_cuts.svg', h=pdf.eph*0.9, x=Align.L)

    pdf.set_xy(pdf.eph+30, 30)

    # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **Effects of samples_over_threshold cuts on distributions**", tag_styles=tag_styles)

    pdf.image( figures_dir / 'ar39_dists_with_tot_cuts.svg', h=pdf.eph*0.9, x=Align.L)

    pdf.set_xy(pdf.eph+30, 30)


    # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **Effects of SADCs cuts on distributions**", tag_styles=tag_styles)

    pdf.image( figures_dir / 'ar39_dists_with_sadcs_cuts.svg', h=pdf.eph*0.9, x=Align.L)

    pdf.set_xy(pdf.eph+30, 30)

    # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **PeakADC cuts noise rejection efficiency**", tag_styles=tag_styles)

    pdf.image( figures_dir / 'ar39_peak_thresh_scan_perf.svg', w=pdf.epw*0.95, x=Align.C)

    pdf.set_xy(pdf.eph+30, 30)

    # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **samples_over_threshold cuts noise rejection efficiency**", tag_styles=tag_styles)

    pdf.image( figures_dir / 'ar39_tot_thresh_scan_perf.svg', w=pdf.epw*0.95, x=Align.C)

    pdf.set_xy(pdf.eph+30, 30)

        # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **adc_integral cuts noise rejection efficiency**", tag_styles=tag_styles)

    pdf.image( figures_dir / 'ar39_sadc_thresh_scan_perf.svg', w=pdf.epw*0.95, x=Align.C)

    pdf.set_xy(pdf.eph+30, 30)

    pdf.output(report_file)

import click

@click.command()
@click.argument("tp_file_path", type=click.Path(dir_okay=False))
@click.argument("wf_file_path",type=click.Path(dir_okay=False))
@click.option('-f', '--first_entry', type=int, default=0)
@click.option('-l', '--last_entry', type=int, default=None)
@click.option('-i', 'interactive', is_flag=True, default=False)
@click.option('--figs/--no-figs', 'make_figures', default=True)
@click.option('-o', 'output_dir', type=click.Path(file_okay=False, exists=True), default='./reports/ar39/')
def cli(tp_file_path, wf_file_path, output_dir, first_entry, last_entry, make_figures, interactive):

    # report_dir = Path('./reports/ar39/')
    report_dir = Path(output_dir)
    figures_dir = report_dir / 'figures'

    make_report = True
    if make_figures:

        with temporary_log_level(workspace.TriggerPrimitivesWorkspace._log, logging.INFO):
            # entry_begin, entry_end = entry_range if not entry_range is None else (0, None)
            ws = workspace.TriggerPrimitivesWorkspace(tp_file_path, first_entry, last_entry)

            print(ws.info)
            if wf_file_path:
                ws.add_rawdigits(wf_file_path)

        events = ws.event_summary.event.unique()

        notes = {
            'tp_file_path': tp_file_path,
            'wf_file_path': wf_file_path,
            'ws_info': ws.info,
            'mc_generator_labels': list(ws.mctruth_blocks_map.values()),
            'tpg_info': ws.info['tpg'][ws.tp_maker_name],
            'event_begin': int(min(events)),
            'event_end': int(max(events))
        }

        with open(figures_dir / 'notes.json', 'w') as fp:
            json.dump(notes, fp, indent=4)

        if interactive:
            import IPython
            IPython.embed(colors='neutral')

        with temporary_log_level(log_prep, logging.INFO):

            images = prepare_figures(ws, figures_dir)
            print(images)

    # ------
    if make_report:
        write_report(figures_dir, report_dir / "ar39_report.pdf")

if __name__ == '__main__':
    # tp_tree_file = 'data/vd/ar39/100events/trigtree_tpg_vd_ar39.root'
    # wf_file = 'data/vd/ar39/100events/trigger_digits_waves_detsim_vd_ar39.root'

    # cli(tp_tree_file, wf_file, entry_range=(0, 99), interactive=False)
    # main(tp_tree_file, None, entry_range=(0,10), interactive=False)
    cli()
