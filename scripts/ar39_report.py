#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import uproot
import logging
import tpvalidator.workspace as workspace
import tpvalidator.utilities as utils
import tpvalidator.analyzers.snn as snn

from rich import print
from rich.logging import RichHandler
from tpvalidator.utilities import temporary_log_level, subplot_autogrid
from tpvalidator.histograms import uproot_hist_mean_std
from io import BytesIO


from fpdf import FPDF, FontFace, Align
from pathlib import Path


class MyPDF(FPDF):
    # def header(self):
    #     # Rendering logo:
    #     self.image("../docs/fpdf2-logo.png", 10, 8, 33)
    #     # Setting font: helvetica bold 15
    #     self.set_font("helvetica", style="B", size=15)
    #     # Moving cursor to the right:
    #     self.cell(80)
    #     # Printing title:
    #     self.cell(30, 10, "Title", border=1, align="C")
    #     # Performing a line break:
    #     self.ln(20)

    def footer(self):
        # Position cursor at 1.5 cm from bottom:
        self.set_y(-15)
        # Setting font: helvetica italic 8
        self.set_font("Raleway", size=8)
        # Printing page number:
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


FORMAT = "%(message)s"
logging.basicConfig(
    level="WARN", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

#
# EXPERIMENTAL
#
# class MyLittleReport:
#     pass
#     log = logging.getLogger('PlotMaker')


# class FigurePage:

#     def __init__(self, name, fmt='svg'):
#         self.name = name
#         self.fmt = fmt
#         self.fig = None

#     @property
#     def log(self):
#         return self._maker.log

#     def make_plot(self, maker: PlotMaker) -> plt.Figure:
#         # 2 Ar39 point of origin on the xy, xz and yz planes
#         fig = maker.alltp_ana.draw_tp_origin_2d_dist()
#             self.fig = fig

#     def save_fig(self, output_dir: str) -> str:
#         if self.fig is None:
#             return
        
#         self.log.info(f'Generating {self.name}')
#         img_path = output_dir / (self.name + self.fmt)
#         self.fig.save_fig(img_path)
#         return img_path

#     def make_page(self, pdf):
#         pdf.write_html("<h1>Noise and AR39 signal distribution</h1>")

#         pdf.image('./tmp/ar39_adc_dist.svg', w=pdf.epw)
#         # with pdf.text_columns() as cols:

#         pdf.write_html("""
#             ADC samples distributions per plane (integrated on all events)
#             <ul>
#                     <li> Blue: ADC distribution on channels where IDE are present
#                     <li> Orange: ADC distribution on channels where IDE are absent
#             </ul>
#         """
#         )



#
# Experimental
#

class Portfolio:
    _log = logging.getLogger('TriggerPrimitivesWorkspace')

    def __init__(self, image_folder: str):
        """A porfolio of named images
        """
        
        self.folder = Path(image_folder)
        self.figures = {}

        self.folder.mkdir(parents=True, exist_ok=True)

    def add_figure(self, img_name: str, fig, fmt: str='svg') -> Path:
        """Add a figure to the portfolio

        Args:
            fig (_type_): _description_
            img_name (str): name of the image
            fmt (str, optional): image format. Defaults to 'svg'.

        Returns:
            _type_: _description_
        """
        img_path = self.folder / (f'{img_name}.{fmt}')
        self.figures[img_name] = img_path
        # if fig is a function, generate the figure
        if callable(fig):
            self._log.info(f"Generating {img_name}")

            fig = fig()
        self.figures[img_name] = (fig, img_path)

        # TODO: decouple?
        self._log.info(f"Saving {img_name} as {img_path}")
        fig.savefig(img_path)

        return img_path
    

class LazyFigure:

    def __init__(self, analyzer_name, analyzer_method, *args, **kwargs):
        self.ana_name = analyzer_name
        self.ana_method = analyzer_method
        self.args = args
        self.kwarg = kwargs

    def __call__(self, ws) -> mpl.figure.Figure:

        ana = getattr(ws.analyzers, self.ana_name)
        method = getattr(ana, self.ana_method)
        fig = method(*self.args, **self.kwargs)


def save_fig(fig, img_name, fmt='svg', out_dir=Path('./tmp')):
    logging.info(f'Generating {img_name}')
    img_path = out_dir / (f'{img_name}.{fmt}')
    fig.savefig(img_path)
    return {img_name: img_path}



import functools

log_prep = logging.getLogger('prepare_figures')
def prepare_figures(ws: workspace.TriggerPrimitivesWorkspace, output_dir: Path) -> dict:

    # Analyze the entire dataset
    all_tps = snn.TPSignalNoisePreSelection(ws.tps)
    alltp_ana = snn.TPSignalNoiseAnalyzer(all_tps)

    # create analyzers
    tps = snn.TPSignalNoisePreSelection(ws.tps[(ws.tps.TP_startT >100) & (ws.tps.TP_startT <8100)])
    tp_ana = snn.TPSignalNoiseAnalyzer(tps)


    pf = Portfolio(output_dir)

    pf.add_figure('ar39_adc_dist', functools.partial(snn.draw_signal_and_noise_adc_distros, ws))


    # 2 Ar39 point of origin on the xy, xz and yz planes
    pf.add_figure('ar39_xyz_pos_dist_all_tps', alltp_ana.draw_tp_origin_2d_dist, fmt='png')

    # 3 Ar39 point of origin on the xy, xz and yz planes
    pf.add_figure('ar39_x_pos_dist_all_tps', alltp_ana.draw_tp_drift_depth_dist)

    # Distribution of signat tps time in the drift direction
    pf.add_figure('ar39_start_time_dist_all_tps', alltp_ana.draw_tp_start_time_dist)

    # Plot ides time distribution
    def plot_ides_time():
        fig, ax = plt.subplots()
        ws.ides.time.plot.hist(bins=1000, ax=ax)
        ax.set_xlabel('time')
        ax.set_ylabel('counts')
        return fig

    pf.add_figure('ar39_ides_time_dist_all_tps', plot_ides_time)

    # Draw TP start time distribution after cleaning
    pf.add_figure('ar39_start_time_dist', tp_ana.draw_tp_start_time_dist)

    # Draw signal and noise distributions
    pf.add_figure('ar39_vs_elnoise_var_dist', tp_ana.draw_tp_signal_noise_dist)

    # Draw grid of tp peakADC, TOT and SADC dist in bins of depth
    pf.add_figure('ar39_peakadc_dist_in_drift_bins', functools.partial(tp_ana.draw_variable_in_drift_grid, 'peakADC', downsampling=10, sharex=True, sharey=True, figsize=(12,10)))
    pf.add_figure('ar39_tot_dist_in_drift_bins', functools.partial(tp_ana.draw_variable_in_drift_grid, 'TOT', downsampling=1, log=False, sharey=True, figsize=(12,10)))
    pf.add_figure('ar39_sadc_dist_in_drift_bins', functools.partial(tp_ana.draw_variable_in_drift_grid, 'SADC', downsampling=100, sharey=True, figsize=(12,10)))

    # Draw grid of tp peakADC, TOT and SADC dist in bins of depth
    pf.add_figure('ar39_peak_dist_stack_in_drift_bins', functools.partial(tp_ana.draw_variable_drift_stack, 'peakADC', downsampling=5, n_x_bins=4, log=True, figsize=(5,4)))
    pf.add_figure('ar39_tot_dist_stack_in_drift_bins', functools.partial(tp_ana.draw_variable_drift_stack, 'TOT', downsampling=1, n_x_bins=4, log=False, figsize=(5,4)))
    pf.add_figure('ar39_sadc_dist_stack_in_drift_bins', functools.partial(tp_ana.draw_variable_drift_stack, 'SADC', downsampling=5, n_x_bins=4, log=True, figsize=(5,4)))

    # Draw the impact of cuts on TP distributions
    cuts = [t for t in range(26, 50, 5)]
    pf.add_figure('ar39_dists_with_peakadc_cuts', functools.partial( tp_ana.draw_variable_cut_sequence, 'peakADC', cuts, log=True, figsize=(10, 10)))
    cuts = [t for t in range(0,10,2)]
    pf.add_figure('ar39_dists_with_tot_cuts', functools.partial( tp_ana.draw_variable_cut_sequence, 'TOT', cuts, log=True, figsize=(10, 10)))
    cuts = [t for t in range(0, 500, 100)]
    pf.add_figure('ar39_dists_with_sadcs_cuts', functools.partial( tp_ana.draw_variable_cut_sequence, 'SADC', cuts, log=True, figsize=(10, 10)))


    # Draw the impact of cuts on TP distributions
    thresholds = [t for t in range(26, 120, 1)]
    pf.add_figure('ar39_peak_thresh_scan_perf', functools.partial( tp_ana.draw_threshold_scan, 'peakADC', thresholds))
    thresholds = [t for t in range(0,10,2)]
    pf.add_figure('ar39_tot_thresh_scan_perf', functools.partial( tp_ana.draw_threshold_scan, 'TOT', thresholds))
    thresholds = [t for t in range(0, 500, 100)]
    pf.add_figure('ar39_sadc_thresh_scan_perf', functools.partial( tp_ana.draw_threshold_scan, 'SADC', thresholds))

    return


def main(tp_file_path : str, wf_file_path: str, event_range=None, interactive: bool=False):

    report_dir = Path('./reports/ar39/')
    figures_dir = report_dir / 'figures'

    prepare_figs = False
    make_report = True
    if prepare_figs:

        with temporary_log_level(workspace.TriggerPrimitivesWorkspace._log, logging.INFO):
            event_begin, event_end = event_range if not event_range is None else (0, None)
            ws = workspace.TriggerPrimitivesWorkspace(tp_file_path, event_begin, event_end)

            print(ws.info)
            if wf_file:
                ws.add_rawdigits(wf_file)

        notes = {
            'tp_file_path': tp_file_path,
            'wf_file_path': wf_file_path,
            'ws_info': ws.info,
            'event_begin': event_begin,
            'event_end': event_end
        }

        with open(figures_dir / 'notes.json', 'w') as fp:
            json.dump(notes, fp)

        if interactive:
            import IPython
            IPython.embed(colors='neutral')

        with temporary_log_level(log_prep, logging.INFO):

            images = prepare_figures(ws, figures_dir)
            print(images)




    # ------
    if make_report:
        # Creater report file
        pdf = MyPDF(orientation="landscape", format="A4")

        pdf.add_font('Raleway', '', '/Users/ale/Library/Fonts/Raleway-Regular.ttf')
        pdf.add_font('Raleway', 'B', '/Users/ale/Library/Fonts/Raleway-Bold.ttf')
        pdf.set_font("Raleway", size=16)

        tag_styles={
            "h1": FontFace(color="#093fb5ff", size_pt=28),
            "h2": FontFace(color="#f06000", size_pt=24),
        }

        with open(figures_dir / 'notes.json', 'r') as fp:
            notes = json.load(fp)


        # This requires access to the WS
        pdf.add_page()
        pdf.write_html("<h1>A first look at Ar39 in VD simulation</h1>", tag_styles=tag_styles)
        pdf.write_html(f"""<h2>Data sample details</h2>
        <ul>
            <li>TP data file: <b>{notes['tp_file_path']}</b></li>
            <li>Waveforms data: <b>{notes['wf_file_path']}</b></li>
            <li>Detector geometry: <b>{notes['ws_info']['geo']['detector']}</b></li>
            <li>MC generator(s): <b>{', '.join(notes['ws_info']['mc_generator_labels'])}</b></li>
            <li>TPG settings:
            <ul>
                <li> Algorithm: <b>{notes['ws_info']['tpg']['tool']}</b>
                <li> Threshold U: <b>{notes['ws_info']['tpg']['threshold_tpg_plane0']}</b>
                <li> Threshold V: <b>{notes['ws_info']['tpg']['threshold_tpg_plane1']}</b>
                <li> Threshold X/Z: <b>{notes['ws_info']['tpg']['threshold_tpg_plane2']}</b>
            </ul>
            <li>Event: <b>{notes['event_begin']}-{notes['event_end']}</b>
            </li>
        </ul>""", tag_styles=tag_styles)


        # pdf.output(report_dir / "ar39_report.pdf")
        # return
        # Page 2
        # ---------------------------------------------------------------------
        pdf.add_page()
        pdf.write_html("<h1>Noise and AR39 signal distribution</h1>", tag_styles=tag_styles)

        pdf.image( figures_dir / 'ar39_adc_dist.svg', w=pdf.epw)
        # with pdf.text_columns() as cols:

        pdf.write_html("""
            ADC samples distributions per plane (integrated on all events)
            <ul>
                    <li> Blue: ADC distribution on channels where IDE are present
                    <li> Orange: ADC distribution on channels where IDE are absent
            </ul>
        """, tag_styles=tag_styles)

        # Page 3
        # ---------------------------------------------------------------------
        pdf.add_page()
        pdf.write_html("<h1>Ar39 TPs origin</h1>", tag_styles=tag_styles)

        pdf.image( figures_dir / 'ar39_xyz_pos_dist_all_tps.png', h=pdf.eph*0.9, x=Align.L)

        pdf.set_xy(pdf.eph+30, 30)
        pdf.write_html("""
            Point of origin of TPs (trueX) tagged as signal, i.e. matching an IDE
        """, tag_styles=tag_styles)

        # ---------------------------------------------------------------------
        pdf.add_page()

        pdf.write_html("<h1>Ar39 TPs depth origin in the drift</h1>", tag_styles=tag_styles)
        pdf.image( figures_dir / 'ar39_x_pos_dist_all_tps.svg', w=pdf.epw*0.9, x=Align.C)
        pdf.write_html("Distribution of trueX for TPs tagged as signal in the 3 planes", tag_styles=tag_styles)


        # ---------------------------------------------------------------------
        pdf.add_page()
        pdf.write_html("<h1>Timing of TPs tagged as signal and noise</h1>", tag_styles=tag_styles)
        pdf.image( figures_dir / 'ar39_start_time_dist_all_tps.svg', w=pdf.epw*0.8, x=Align.L)
        pdf.image( figures_dir / 'ar39_ides_time_dist_all_tps.svg', h=pdf.eph*0.4, x=Align.L)
        pdf.set_xy(pdf.epw//3+30, 2*pdf.eph//3+30)
        pdf.write_html("""
                    <ul>
                    <li> top 3 figures: Distribution of TP time by plane
                    <li> bottom figure: Distribution of IDEs time of arrival at the anode (CRP)
                    </ul>
        """, tag_styles=tag_styles)


        # ---------------------------------------------------------------------
        pdf.add_page()
        pdf.write_html("<h1>Timing of TPs tagged as signal and noise (clean)</h1>", tag_styles=tag_styles)
        pdf.image( figures_dir / 'ar39_start_time_dist.svg', w=pdf.epw*0.8, x=Align.L)
        pdf.write_html("""
                    <ul>
                    <li> Distribution of TP startTime after applying cleanup
                    <li> startT > 100 && startT < 8200
                    </ul>

        """, tag_styles=tag_styles)


        # ---------------------------------------------------------------------
        pdf.add_page()
        pdf.write_html("<h1>Basic TP distribution</h1>", tag_styles=tag_styles)
        pdf.image( figures_dir / 'ar39_vs_elnoise_var_dist.svg', w=pdf.epw, x=Align.C)
        # pdf.write_html("""
        #                <ul>
        #                <li> Distribution of TP startTime after applying cleanup
        #                <li> startT > 100 && startT < 8200
        #                </ul>

        # """, tag_styles=tag_styles)

        # ---------------------------------------------------------------------
        pdf.add_page()
        pdf.write_html("<h1>Distribution of TP peakADC across the drift</h1>(Bins of trueX)", tag_styles=tag_styles)
        pdf.image( figures_dir / 'ar39_peakadc_dist_in_drift_bins.svg', w=pdf.eph, x=Align.C)


        # ---------------------------------------------------------------------
        pdf.add_page()
        pdf.write_html("<h1>Distribution of TP TOT across the drift</h1>(Bins of trueX)", tag_styles=tag_styles)
        pdf.image( figures_dir / 'ar39_tot_dist_in_drift_bins.svg', w=pdf.eph, x=Align.C)

        # ---------------------------------------------------------------------
        pdf.add_page()
        pdf.write_html("<h1>Distribution of TP SADC across the drift</h1>(Bins of trueX)", tag_styles=tag_styles)
        pdf.image( figures_dir / 'ar39_sadc_dist_in_drift_bins.svg', w=pdf.eph, x=Align.C)


        # ---------------------------------------------------------------------
        pdf.add_page()
        pdf.write_html("<h1>Stacked TP distributions</h1>", tag_styles=tag_styles)

        pdf.set_y(pdf.eph//3)
        x = pdf.get_x()
        y = pdf.get_y()
        pdf.image( figures_dir / 'ar39_peak_dist_stack_in_drift_bins.svg', w=pdf.epw//3)
        pdf.set_y(y)
        pdf.image( figures_dir / 'ar39_tot_dist_stack_in_drift_bins.svg', w=pdf.epw//3, x=x+pdf.epw//3)
        pdf.set_y(y)
        pdf.image( figures_dir / 'ar39_sadc_dist_stack_in_drift_bins.svg', w=pdf.epw//3, x=x+2*pdf.epw//3)

        # ---------------------------------------------------------------------
        pdf.add_page()
        pdf.write_html("<h1>Effects of peakADC cuts on distributions</h1>", tag_styles=tag_styles)

        pdf.image( figures_dir / 'ar39_dists_with_peakadc_cuts.svg', h=pdf.eph*0.9, x=Align.L)

        pdf.set_xy(pdf.eph+30, 30)
        # pdf.write_html("""
        #     Point of origin of TPs (trueX) tagged as signal, i.e. matching an IDE
        # """, tag_styles=tag_styles)

        # ---------------------------------------------------------------------
        pdf.add_page()
        pdf.write_html("<h1>Effects of TOT cuts on distributions</h1>", tag_styles=tag_styles)

        pdf.image( figures_dir / 'ar39_dists_with_tot_cuts.svg', h=pdf.eph*0.9, x=Align.L)

        pdf.set_xy(pdf.eph+30, 30)
        # pdf.write_html("""
        #     Point of origin of TPs (trueX) tagged as signal, i.e. matching an IDE
        # """, tag_styles=tag_styles)


        # ---------------------------------------------------------------------
        pdf.add_page()
        pdf.write_html("<h1>Effects of SADCs cuts on distributions</h1>", tag_styles=tag_styles)

        pdf.image( figures_dir / 'ar39_dists_with_sadcs_cuts.svg', h=pdf.eph*0.9, x=Align.L)

        pdf.set_xy(pdf.eph+30, 30)
        # pdf.write_html("""
        #     Point of origin of TPs (trueX) tagged as signal, i.e. matching an IDE
        # """, tag_styles=tag_styles)

        # ---------------------------------------------------------------------
        pdf.add_page()
        pdf.write_html("<h1>Ar39 TPs origin</h1>", tag_styles=tag_styles)

        pdf.image( figures_dir / 'ar39_peak_thresh_scan_perf.svg', w=pdf.epw*0.95, x=Align.C)

        pdf.set_xy(pdf.eph+30, 30)

        # ---------------------------------------------------------------------
        pdf.add_page()
        pdf.write_html("<h1>Ar39 TPs origin</h1>", tag_styles=tag_styles)

        pdf.image( figures_dir / 'ar39_tot_thresh_scan_perf.svg', w=pdf.epw*0.95, x=Align.C)

        pdf.set_xy(pdf.eph+30, 30)

            # ---------------------------------------------------------------------
        pdf.add_page()
        pdf.write_html("<h1>Ar39 TPs origin</h1>", tag_styles=tag_styles)

        pdf.image( figures_dir / 'ar39_sadc_thresh_scan_perf.svg', w=pdf.epw*0.95, x=Align.C)

        pdf.set_xy(pdf.eph+30, 30)

        pdf.output(report_dir / "ar39_report.pdf")

if __name__ == '__main__':
    tp_tree_file = 'data/vd/ar39/100events/tptree_st_tpg_vd_ar39.root'
    wf_file = 'data/vd/ar39/100events/trigger_digits_waves_detsim_vd_ar39.root'

    main(tp_tree_file, wf_file, event_range=(0, 100), interactive=False)

    # import tempfile

    # with tempfile.TemporaryDirectory() as tmpdir:
        # print(tmpdir)