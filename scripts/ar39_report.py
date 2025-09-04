#!/usr/bin/env python


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


from fpdf import FPDF, FontFace
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

    def __init__(self, image_folder: str):
        """A porfolio of named images
        """
        
        self.folder = Path(image_folder)
        self.figures = {}

    def add_figure(self, fig, img_name: str, fmt: str='svg', *args, **kwrgs):
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
            fig = fig(*args, **kwrgs)
        fig.savefig(img_path)

        return img_path



def save_fig(fig, img_name, fmt='svg', out_dir=Path('./tmp')):
    logging.info(f'Generating {img_name}')
    img_path = out_dir / (f'{img_name}.{fmt}')
    fig.savefig(img_path)
    return {img_name: img_path}



log_prep = logging.getLogger('prepare_figures')

def prepare_figures(ws: workspace.TriggerPrimitivesWorkspace, output_dir: Path) -> dict:

    pf = Portfolio('./tmp')


    log = log_prep

    images = {}


    # 1
    fig = snn.draw_signal_and_noise_adc_distros(ws)
    images.update(save_fig(fig, 'ar39_adc_dist'))

    # Dataset with all TPs - no cleanup
    all_tps = snn.TPSignalNoisePreSelection(ws.tps)
    alltp_ana = snn.TPSignalNoiseAnalyzer(all_tps)

    # 2 Ar39 point of origin on the xy, xz and yz planes
    fig = alltp_ana.draw_tp_origin_2d_dist()
    images.update(save_fig(fig, 'ar39_xyz_pos_dist_all_tps', fmt='png'))

    # 3 Ar39 point of origin on the xy, xz and yz planes
    fig = alltp_ana.draw_tp_drift_depth_dist()
    images.update(save_fig(fig, 'ar39_x_pos_dist_all_tps'))

    # ---
    fig = alltp_ana.draw_tp_start_time_dist()
    images.update(save_fig(fig, 'ar39_start_time_dist_all_tps'))

    # ---
    fig, ax = plt.subplots()
    ws.ides.time.plot.hist(bins=1000, ax=ax)
    images.update(save_fig(fig, 'ar39_ides_time_dist_all_tps'))

    fig, ax = plt.subplots()
    ws.ides.time.plot.hist(bins=1000)
    images.update(save_fig(fig, 'ar39_ides_time_dist_all_tps'))


    tps = snn.TPSignalNoisePreSelection(ws.tps[(ws.tps.TP_startT >100) & (ws.tps.TP_startT <8100)])
    tp_ana = snn.TPSignalNoiseAnalyzer(tps)
    
    # ---
    fig = tp_ana.draw_tp_start_time_dist()
    images.update(save_fig(fig, 'ar39_start_time_dist'))

    # ---
    fig = tp_ana.draw_tp_signal_noise_dist()
    images.update(save_fig(fig, 'ar39_vs_elnoise_var_dist'))

    fig = tp_ana.draw_variable_in_drift_grid('peakADC', downsampling=10, sharex=True, sharey=True, figsize=(12,10))
    images.update(save_fig(fig, 'ar39_peakadc_dist_in_drift_bins'))

    fig = tp_ana.draw_variable_in_drift_grid('TOT', downsampling=1, log=False, sharey=True, figsize=(12,10))
    images.update(save_fig(fig, 'ar39_tot_dist_in_drift_bins'))

    fig = tp_ana.draw_variable_in_drift_grid('SADC', downsampling=100, sharey=True, figsize=(12,10))
    images.update(save_fig(fig, 'ar39_sadc_dist_in_drift_bins'))


    fig = tp_ana.draw_variable_drift_stack('peakADC', downsampling=5, n_x_bins=4, log=True, figsize=(5,4))
    images.update(save_fig(fig, 'ar39_peak_dist_stack_in_drift_bins'))

    fig = tp_ana.draw_variable_drift_stack('TOT', downsampling=1, n_x_bins=4, log=False, figsize=(5,4))
    images.update(save_fig(fig, 'ar39_tot_dist_stack_in_drift_bins'))

    fig = tp_ana.draw_variable_drift_stack('SADC', downsampling=5, n_x_bins=4, log=True, figsize=(5,4))
    images.update(save_fig(fig, 'ar39_sadc_dist_stack_in_drift_bins'))

    return images


def main(tp_file_path : str, wf_file_path: str, event_range=None, interactive: bool=False):

    with temporary_log_level(workspace.TriggerPrimitivesWorkspace._log, logging.INFO):
        event_begin, event_end = event_range if not event_range is None else (0, None)
        ws = workspace.TriggerPrimitivesWorkspace(tp_file_path, event_begin, event_end)

        print(ws.info)
        if wf_file:
            ws.add_rawdigits(wf_file)

    if interactive:
        import IPython
        IPython.embed(colors='neutral')

    with temporary_log_level(log_prep, logging.INFO):

        images = prepare_figures(ws, Path('./tmp/'))
        print(images)

    return

    # Creater report file
    pdf = MyPDF(orientation="landscape", format="A4")

    pdf.add_font('Raleway', '', '/Users/ale/Library/Fonts/Raleway-Regular.ttf')
    pdf.add_font('Raleway', 'B', '/Users/ale/Library/Fonts/Raleway-Bold.ttf')
    pdf.set_font("Raleway", size=16)

    tag_styles={
        "h1": FontFace(color="#0069f2e7", size_pt=28),
        "h2": FontFace(color="#f06000", size_pt=24),
    }
    # pdf.add_page()
    # pdf.write_html("<h1>A first look at Ar39 in VD simulation</h1>")
    # pdf.write_html(f"""<h2>Data sample details</h2>
    # <ul>
    #     <li>TP data file: <b>{tp_file_path}</b></li>
    #     <li>Waveforms data: <b>{wf_file_path}</b></li>
    #     <li>Detector geometry: <b>{ws.info['geo']['detector']}</b></li>
    #     <li>MC generator(s): <b>{', '.join(ws.info['mc_generator_labels'])}</b></li>
    #     <li>TPG settings:
    #     <ul>
    #         <li> Algorithm: <b>{ws.info['tpg']['tool']}</b>
    #         <li> Threshold U: <b>{ws.info['tpg']['threshold_tpg_plane0']}</b>
    #         <li> Threshold V: <b>{ws.info['tpg']['threshold_tpg_plane1']}</b>
    #         <li> Threshold X/Z: <b>{ws.info['tpg']['threshold_tpg_plane2']}</b>
    #     </ul>
    #     <li>Event: <b>{ws.tps.event.min()}-{ws.tps.event.max()}</b>
    #     </li>
    # </ul>""")



    # Page 2
    pdf.add_page()
    pdf.write_html("<h1>Noise and AR39 signal distribution</h1>", tag_styles=tag_styles)

    pdf.image('./tmp/ar39_adc_dist.svg', w=pdf.epw)
    # with pdf.text_columns() as cols:

    pdf.write_html("""
        ADC samples distributions per plane (integrated on all events)
        <ul>
                <li> Blue: ADC distribution on channels where IDE are present
                <li> Orange: ADC distribution on channels where IDE are absent
        </ul>
    """, tag_styles=tag_styles)

    # Page 3
    pdf.add_page()
    pdf.write_html("<h1>Ar39 TPs generation origin</h1>", tag_styles=tag_styles)

    pdf.image('./tmp/ar39_xyz_pos_dist_all_tps.png', h=pdf.eph*0.9)

    pdf.set_xy(pdf.eph+30, 30)
    pdf.write_html("""
        Point of origin of TPs tagged as signal, i.e. matching an IDE
    """, tag_styles=tag_styles)


    pdf.output("ar39_report.pdf")

if __name__ == '__main__':
    tp_tree_file = 'data/vd/ar39/100events/tptree_st_tpg_vd_ar39.root'
    wf_file = 'data/vd/ar39/100events/trigger_digits_waves_detsim_vd_ar39.root'

    main(tp_tree_file, wf_file, event_range=(0, 1), interactive=False)

    # import tempfile

    # with tempfile.TemporaryDirectory() as tmpdir:
        # print(tmpdir)