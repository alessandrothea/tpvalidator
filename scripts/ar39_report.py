#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import mistletoe as mt
import uproot
import textwrap
import logging
import tpvalidator.workspace as workspace
import tpvalidator.utilities as utils
import tpvalidator.analyzers.snn as snn

from rich import print
from rich.logging import RichHandler
from tpvalidator.utilities import temporary_log_level, subplot_autogrid
from tpvalidator.histograms import uproot_hist_mean_std
from io import BytesIO


from fpdf import FPDF, HTML2FPDF, FontFace, TextStyle, Align
from pathlib import Path


from tpvalidator.portfolio import ReportPDF, Portfolio


# class MyHTML2FPDF(HTML2FPDF):
#     def __init__(
#         self,
#         pdf,
#         image_map=None,
#         li_tag_indent=None,
#         dd_tag_indent=None,
#         table_line_separators=False,
#         ul_bullet_char="disc",
#         li_prefix_color=(190, 0, 0),
#         heading_sizes=None,
#         pre_code_font=None,
#         warn_on_tags_not_matching=True,
#         tag_indents=None,
#         tag_styles=None,
#         font_family="times",
#         render_title_tag=False,
#     ):
        
#         super().__init__(
#             pdf,
#             image_map=image_map,
#             li_tag_indent=li_tag_indent,
#             dd_tag_indent=dd_tag_indent,
#             table_line_separators=table_line_separators,
#             ul_bullet_char=ul_bullet_char,
#             li_prefix_color=li_prefix_color,
#             heading_sizes=heading_sizes,
#             pre_code_font=pre_code_font,
#             warn_on_tags_not_matching=warn_on_tags_not_matching,
#             tag_indents=tag_indents,
#             tag_styles=tag_styles,
#             font_family=font_family,
#             render_title_tag=render_title_tag,
#         )


# class ReportPDF(FPDF):
#     HTML2FPDF_CLASS = MyHTML2FPDF


#     # def header(self):
#     #     # Rendering logo:
#     #     self.image("../docs/fpdf2-logo.png", 10, 8, 33)
#     #     # Setting font: helvetica bold 15
#     #     self.set_font("helvetica", style="B", size=15)
#     #     # Moving cursor to the right:
#     #     self.cell(80)
#     #     # Printing title:
#     #     self.cell(30, 10, "Title", border=1, align="C")
#     #     # Performing a line break:
#     #     self.ln(20)

#     def footer(self):
#         # Position cursor at 1.5 cm from bottom:
#         self.set_y(-15)
#         # Setting font: helvetica italic 8
#         self.set_font("Raleway", size=8)
#         # Printing page number:
#         self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

#     def write_markdown(self, text, *args, **kwargs):
#         text = textwrap.dedent(text)
#         # print(text)
#         self.write_html(mt.markdown(text), *args, **kwargs)

#     def move_cursor(self, dx=0, dy=0):
#         """Move the cursor in x and y

#         Args:
#             dx (int, optional): cursor displacement along x. Defaults to 0.
#             dy (int, optional): cursor displacement aling y. Defaults to 0.
#         """
#         self.set_xy(self.get_x() + dx, self.get_y() + dy)

#     def mark_cursor(self, size=3, text=None):
#         """Draw a crosshair centered at the current cursor posistion

#         Args:
#             size (int, optional): Crosshair size. Defaults to 3.
#         """
#         x, y = self.get_x(), self.get_y()
#         # small cross centered at (x, y)
#         self.set_draw_color(200, 0, 0)
#         self.line(x - size, y, x + size, y)      # horizontal
#         self.line(x, y - size, x, y + size)  
#         s = text or f"({x:.1f},{y:.1f})"
#         # self.set_text_color(0, 0, 180)
#         # self.set_font("Helvetica", size=8)
#         self.text(x + 2, y - 2, s)  # tiny label near the cross

#     def outline_next_cell(self, w, h):
#         x, y = self.get_x(), self.get_y()
#         self.set_draw_color(0, 100, 200)
#         self.rect(x, y, w, h)   # outline

#     def label_cursor(self, text=None):
#         x, y = self.get_x(), self.get_y()
#         s = text or f"({x:.1f},{y:.1f})"
#         # self.set_text_color(0, 0, 180)
#         # self.set_font("Helvetica", size=8)
#         self.text(x + 2, y - 2, s)  # tiny label near the cross
        
#     def debug_grid(self, step=10):
#         w, h = self.w, self.h
#         self.set_draw_color(220, 220, 220)
#         for gx in range(0, int(w), step):
#             self.line(gx, 0, gx, h)
#         for gy in range(0, int(h), step):
#             self.line(0, gy, w, gy)

#     def draw_margins(self):
#         """Box showing the margin-bounded content area."""
#         x = self.l_margin
#         y = self.t_margin
#         w = self.w - self.l_margin - self.r_margin
#         h = self.h - self.t_margin - self.b_margin
#         self.set_draw_color(0, 120, 200)
#         self.set_line_width(0.3)
#         self.rect(x, y, w, h)


#     def image_with_caption(
#         self,
#         img_path: str,
#         *,
#         x: float = None,
#         y: float = None,
#         w: float = None,
#         h: float = None,
#         caption: str = "",
#         gap: float = 2.0,          # space between image and caption (in units)
#         line_h: float = 5.0,       # caption line height
#         center_on_page: bool = False,
#     ):
#         """
#         Draw an image and a caption centered under the image.
#         - Provide either w or h (or both); if only one is given, FPDF keeps aspect ratio.
#         - If center_on_page=True and w is given, image is horizontally centered in the content area.
#         """
#         # Decide X position
#         if x is None:
#             x = self.get_x()
#         if y is None:
#             y = self.get_y()

#         # If we want page-centering and a known width, compute X from margins
#         if center_on_page and w is not None:
#             content_w = self.w - self.l_margin - self.r_margin
#             x = self.l_margin + (content_w - w) / 2

#         # Draw image
#         self.image(img_path, x=x, y=y, w=w, h=h)

#         # Compute the final drawn image height (needed for caption Y).
#         # If h not provided but w is, we can approximate via intrinsic size if available,
#         # otherwise rely on FPDF to preserve aspect ratio and just place caption after a gap.
#         # Easiest robust approach: ask current Y after placing image:
#         # (FPDF doesn't move the cursor for image(), so we compute using our inputs.)
#         img_h = h
#         if img_h is None and w is not None:
#             # If you need exact height from the file’s aspect ratio, set h yourself,
#             # or use PIL to read size. Here we’ll just place caption immediately after y + (h or 0).
#             pass

#         # Caption Y position
#         cap_y = (y + (img_h or 0)) + gap

#         # Draw caption within the image width, centered
#         # (If you didn’t set w, choose a reasonable width for the caption.)
#         cap_w = w if w is not None else (self.w - self.l_margin - self.r_margin)

#         # Save original X to keep layout predictable after multi_cell
#         old_x = self.get_x()
#         self.set_xy(x, cap_y)
#         self.multi_cell(cap_w, line_h, caption, align="C")
#         # After multi_cell, X resets to left margin; restore a sensible X just below caption:
#         self.set_x(old_x)

#         # Return bottom Y so caller can continue below
#         return self.get_y()


FORMAT = "%(message)s"
logging.basicConfig(
    level="WARN", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)


# class Portfolio:
#     _log = logging.getLogger('TriggerPrimitivesWorkspace')

#     def __init__(self, image_folder: str):
#         """A porfolio of named images
#         """
        
#         self.folder = Path(image_folder)
#         self.figures = {}

#         self.folder.mkdir(parents=True, exist_ok=True)

#     def add_figure(self, img_name: str, fig, fmt: str='svg') -> Path:
#         """Add a figure to the portfolio

#         Args:
#             fig (_type_): _description_
#             img_name (str): name of the image
#             fmt (str, optional): image format. Defaults to 'svg'.

#         Returns:
#             _type_: _description_
#         """
#         img_path = self.folder / (f'{img_name}.{fmt}')
#         self.figures[img_name] = img_path
#         # if fig is a function, generate the figure
#         if callable(fig):
#             self._log.info(f"Generating {img_name}")

#             fig = fig()
#         self.figures[img_name] = (fig, img_path)

#         # TODO: decouple?
#         self._log.info(f"Saving {img_name} as {img_path}")
#         fig.savefig(img_path)
#         plt.close(fig)

#         return img_path
    

# class LazyFigure:
#     """Wrapper 
#     """

#     def __init__(self, analyzer_name, analyzer_method, *args, **kwargs):
#         self.ana_name = analyzer_name
#         self.ana_method = analyzer_method
#         self.args = args
#         self.kwarg = kwargs

#     def __call__(self, ws) -> mpl.figure.Figure:

#         ana = getattr(ws.analyzers, self.ana_name)
#         method = getattr(ana, self.ana_method)
#         fig = method(*self.args, **self.kwargs)



import functools

log_prep = logging.getLogger('prepare_figures')
def prepare_figures(ws: workspace.TriggerPrimitivesWorkspace, output_dir: Path) -> dict:

    # Analyze the entire dataset
    all_tps = snn.TPSignalNoisePreSelection(ws.tps)
    alltp_ana = snn.TPSignalNoiseAnalyzer(all_tps)

    # create analyzers
    tps = snn.TPSignalNoisePreSelection(ws.tps[(ws.tps.TP_startT >100) & (ws.tps.TP_startT <8100)])
    tp_ana = snn.TPSignalNoiseAnalyzer(tps)


    pf = Portfolio(output_dir, 'ar39')

    pf.add_figure('adc_dist', functools.partial(snn.draw_signal_and_noise_adc_distros, ws))


    # 2 Ar39 point of origin on the xy, xz and yz planes
    pf.add_figure('xyz_pos_dist_all_tps', alltp_ana.draw_tp_sig_origin_2d_dist, fmt='png')

    # 3 Ar39 point of origin on the xy, xz and yz planes
    pf.add_figure('x_pos_dist_all_tps', alltp_ana.draw_tp_sig_drift_depth_dist)
    pf.add_figure('x_pos_weighted_dist_all_tps', functools.partial(alltp_ana.draw_tp_sig_drift_depth_dist, weight_by="SADC"))

    # Distribution of signat tps time in the drift direction
    pf.add_figure('start_time_dist_all_tps', alltp_ana.draw_tp_start_time_dist)


    x = snn.TPSignalNoiseAnalyzer(all_tps.query('TP_peakADC > 26'))
    pf.add_figure('event_10_peak26_all_tps', functools.partial(x.draw_tp_event, 10), fmt='png')
    x = snn.TPSignalNoiseAnalyzer(all_tps.query('TP_peakADC > 36'))
    pf.add_figure('event_10_peak36_all_tps', functools.partial(x.draw_tp_event, 10), fmt='png')
    x = snn.TPSignalNoiseAnalyzer(all_tps.query('TP_peakADC > 46'))
    pf.add_figure('event_10_peak46_all_tps', functools.partial(x.draw_tp_event, 10), fmt='png')
    x = snn.TPSignalNoiseAnalyzer(all_tps.query('TP_peakADC > 56'))
    pf.add_figure('event_10_peak56_all_tps', functools.partial(x.draw_tp_event, 10), fmt='png')

    # Plot ides time distribution
    def plot_ides_time():
        fig, ax = plt.subplots()
        ws.ides.time.plot.hist(bins=1000, ax=ax)
        ax.set_xlabel('time')
        ax.set_ylabel('counts')
        return fig

    pf.add_figure('ides_time_dist_all_tps', plot_ides_time)

    # Draw TP start time distribution after cleaning
    pf.add_figure('start_time_dist', tp_ana.draw_tp_start_time_dist)

    # Draw signal and noise distributions
    pf.add_figure('vs_elnoise_var_dist', tp_ana.draw_tp_signal_noise_dist)

    # Draw grid of tp peakADC, TOT and SADC dist in bins of depth
    pf.add_figure('peakadc_dist_in_drift_bins', functools.partial(tp_ana.draw_variable_in_drift_grid, 'peakADC', downsampling=10, sharex=True, sharey=True, figsize=(12,10)))
    pf.add_figure('tot_dist_in_drift_bins', functools.partial(tp_ana.draw_variable_in_drift_grid, 'TOT', downsampling=1, log=False, sharey=True, figsize=(12,10)))
    pf.add_figure('sadc_dist_in_drift_bins', functools.partial(tp_ana.draw_variable_in_drift_grid, 'SADC', downsampling=100, sharey=True, figsize=(12,10)))

    # Draw grid of tp peakADC, TOT and SADC dist in bins of depth
    pf.add_figure('peak_dist_stack_in_drift_bins', functools.partial(tp_ana.draw_variable_drift_stack, 'peakADC', downsampling=5, n_x_bins=4, log=True, figsize=(5,4)))
    pf.add_figure('tot_dist_stack_in_drift_bins', functools.partial(tp_ana.draw_variable_drift_stack, 'TOT', downsampling=1, n_x_bins=4, log=False, figsize=(5,4)))
    pf.add_figure('sadc_dist_stack_in_drift_bins', functools.partial(tp_ana.draw_variable_drift_stack, 'SADC', downsampling=5, n_x_bins=4, log=True, figsize=(5,4)))

    # Draw the impact of cuts on TP distributions
    cuts = [t for t in range(26, 50, 5)]
    pf.add_figure('dists_with_peakadc_cuts', functools.partial( tp_ana.draw_variable_cut_sequence, 'peakADC', cuts, log=True, figsize=(15, 10)))
    cuts = [t for t in range(0,10,2)]
    pf.add_figure('dists_with_tot_cuts', functools.partial( tp_ana.draw_variable_cut_sequence, 'TOT', cuts, log=True, figsize=(15, 10)))
    cuts = [t for t in range(0, 500, 100)]
    pf.add_figure('dists_with_sadcs_cuts', functools.partial( tp_ana.draw_variable_cut_sequence, 'SADC', cuts, log=True, figsize=(15, 10)))


    # Draw the impact of cuts on TP distributions
    thresholds = [t for t in range(26, 120, 1)]
    pf.add_figure('peak_thresh_scan_perf', functools.partial( tp_ana.draw_threshold_scan, 'peakADC', thresholds))
    thresholds = [t for t in range(0,10,2)]
    pf.add_figure('tot_thresh_scan_perf', functools.partial( tp_ana.draw_threshold_scan, 'TOT', thresholds))
    thresholds = [t for t in range(0, 500, 100)]
    pf.add_figure('sadc_thresh_scan_perf', functools.partial( tp_ana.draw_threshold_scan, 'SADC', thresholds))

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

    pdf.set_font("OpenSans", size=15)

    color_blue = "#093fb5ff"
    color_orange = "#f06000"

    tag_styles={
        "h1": FontFace(color=color_blue, size_pt=28, family='Raleway'),
        "h2": FontFace(color=color_orange, size_pt=24, family='Raleway'),

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
                        * MC generator(s): **{', '.join(notes['ws_info']['mc_generator_labels'])}**
                            * Algorithm: **{notes['ws_info']['tpg']['tool']}**
                            * Threshold U: **{notes['ws_info']['tpg']['threshold_tpg_plane0']}**
                            * Threshold V: **{notes['ws_info']['tpg']['threshold_tpg_plane1']}**
                            * Threshold X/Z: **{notes['ws_info']['tpg']['threshold_tpg_plane2']}**
                        * Event: **{notes['event_begin']}-{notes['event_end']}**
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


    # Page 3
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
        pdf.cell(w=img_w, text="a) peakADC>26", markdown=True, align=Align.C)
        pdf.cell(w=img_w, text="b) peakADC>36", markdown=True, align=Align.C)

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

        pdf.cell(w=img_w, text="c) peakADC>46", markdown=True, align=Align.C)
        pdf.cell(w=img_w, text="d) peakADC>56", markdown=True, align=Align.C)

    pdf.ln()
    # pdf.mark_cursor(text="I5")

    pdf.write_markdown(f"""
                * Channel and peakT of TPs in event 10
                * Incremental peakADC cuts are applied (from a) to d)) to show the distribution of TPs at higher peakADC.
                * NOTE: A lack of signal TPs is evident at peakT > 8200 for all planes. In this region noise TPs have a harder spectrum:
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
    pdf.write_markdown("# **Ar39  - total SADC sum vs drift depth for signal TPs**", tag_styles=tag_styles)

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
                * startT > 100 && startT < 8200
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
    pdf.write_markdown("# **Distribution of TP peakADC across the drift**\nBins of trueX - plane 2 (collection)", tag_styles=tag_styles)
    pdf.image( figures_dir / 'ar39_peakadc_dist_in_drift_bins.svg', w=pdf.eph, x=Align.C)


    # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **Distribution of TP TOT across the drift**\nBins of trueX - plane 2 (collection)", tag_styles=tag_styles)
    pdf.image( figures_dir / 'ar39_tot_dist_in_drift_bins.svg', w=pdf.eph, x=Align.C)

    # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **Distribution of TP SADC across the drift**\nBins of trueX - plane 2 (collection)", tag_styles=tag_styles)
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
                Comparison of peakADC, TOT and SADC for in 4 regions of trueX for the collection plane.
    """, tag_styles=tag_styles)
    # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **Effects of peakADC cuts on distributions**", tag_styles=tag_styles)

    pdf.image( figures_dir / 'ar39_dists_with_peakadc_cuts.svg', h=pdf.eph*0.9, x=Align.L)

    pdf.set_xy(pdf.eph+30, 30)

    # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **Effects of TOT cuts on distributions**", tag_styles=tag_styles)

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
    pdf.write_markdown("# **TOT cuts noise rejection efficiency**", tag_styles=tag_styles)

    pdf.image( figures_dir / 'ar39_tot_thresh_scan_perf.svg', w=pdf.epw*0.95, x=Align.C)

    pdf.set_xy(pdf.eph+30, 30)

        # ---------------------------------------------------------------------
    pdf.add_page()
    pdf.write_markdown("# **SADC cuts noise rejection efficiency**", tag_styles=tag_styles)

    pdf.image( figures_dir / 'ar39_sadc_thresh_scan_perf.svg', w=pdf.epw*0.95, x=Align.C)

    pdf.set_xy(pdf.eph+30, 30)

    pdf.output(report_file)


def main(tp_file_path : str, wf_file_path: str, event_range=None, make_figures:bool=True, interactive: bool=False):

    report_dir = Path('./reports/ar39/')
    figures_dir = report_dir / 'figures'

    make_report = True
    if make_figures:

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
        write_report(figures_dir, report_dir / "ar39_report.pdf")

if __name__ == '__main__':
    tp_tree_file = 'data/vd/ar39/100events/tptree_st_tpg_vd_ar39.root'
    wf_file = 'data/vd/ar39/100events/trigger_digits_waves_detsim_vd_ar39.root'

    main(tp_tree_file, wf_file, event_range=(0, 10), interactive=False)
