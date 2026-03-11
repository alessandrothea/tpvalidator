import logging
import textwrap
import mistletoe as mt

import matplotlib as mpl
import matplotlib.pyplot as plt

from fpdf import FPDF, HTML2FPDF
from pathlib import Path

class ReportHTML2FPDF(HTML2FPDF):
    def __init__(
        self,
        pdf,
        image_map=None,
        li_tag_indent=None,
        dd_tag_indent=None,
        table_line_separators=False,
        ul_bullet_char="disc",
        li_prefix_color=(190, 0, 0),
        heading_sizes=None,
        pre_code_font=None,
        warn_on_tags_not_matching=True,
        tag_indents=None,
        tag_styles=None,
        font_family="times",
        render_title_tag=False,
    ):
        
        super().__init__(
            pdf,
            image_map=image_map,
            li_tag_indent=li_tag_indent,
            dd_tag_indent=dd_tag_indent,
            table_line_separators=table_line_separators,
            ul_bullet_char=ul_bullet_char,
            li_prefix_color=li_prefix_color,
            heading_sizes=heading_sizes,
            pre_code_font=pre_code_font,
            warn_on_tags_not_matching=warn_on_tags_not_matching,
            tag_indents=tag_indents,
            tag_styles=tag_styles,
            font_family=font_family,
            render_title_tag=render_title_tag,
        )


class ReportPDF(FPDF):
    HTML2FPDF_CLASS = ReportHTML2FPDF


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

    def write_markdown(self, text, *args, **kwargs):
        """Write text in the document with markdown syntax
        """
        text = textwrap.dedent(text)
        # print(text)
        self.write_html(mt.markdown(text), *args, **kwargs)

    def move_cursor(self, dx=0, dy=0):
        """Move the cursor in x and y

        Args:
            dx (int, optional): cursor displacement along x. Defaults to 0.
            dy (int, optional): cursor displacement aling y. Defaults to 0.
        """
        self.set_xy(self.get_x() + dx, self.get_y() + dy)

    def mark_cursor(self, size=3, text=None):
        """Draw a crosshair centered at the current cursor posistion

        Args:
            size (int, optional): Crosshair size. Defaults to 3.
        """
        x, y = self.get_x(), self.get_y()
        # small cross centered at (x, y)
        self.set_draw_color(200, 0, 0)
        self.line(x - size, y, x + size, y)      # horizontal
        self.line(x, y - size, x, y + size)  
        s = text or f"({x:.1f},{y:.1f})"
        # self.set_text_color(0, 0, 180)
        # self.set_font("Helvetica", size=8)
        self.text(x + 2, y - 2, s)  # tiny label near the cross

    def outline_next_cell(self, w, h):
        x, y = self.get_x(), self.get_y()
        self.set_draw_color(0, 100, 200)
        self.rect(x, y, w, h)   # outline

    def label_cursor(self, text=None):
        x, y = self.get_x(), self.get_y()
        s = text or f"({x:.1f},{y:.1f})"
        # self.set_text_color(0, 0, 180)
        # self.set_font("Helvetica", size=8)
        self.text(x + 2, y - 2, s)  # tiny label near the cross
        
    def debug_grid(self, step=10):
        w, h = self.w, self.h
        self.set_draw_color(220, 220, 220)
        for gx in range(0, int(w), step):
            self.line(gx, 0, gx, h)
        for gy in range(0, int(h), step):
            self.line(0, gy, w, gy)

    def draw_margins(self):
        """Box showing the margin-bounded content area."""
        x = self.l_margin
        y = self.t_margin
        w = self.w - self.l_margin - self.r_margin
        h = self.h - self.t_margin - self.b_margin
        self.set_draw_color(0, 120, 200)
        self.set_line_width(0.3)
        self.rect(x, y, w, h)


    def image_with_caption(
        self,
        img_path: str,
        *,
        x: float = None,
        y: float = None,
        w: float = None,
        h: float = None,
        caption: str = "",
        gap: float = 2.0,          # space between image and caption (in units)
        line_h: float = 5.0,       # caption line height
        center_on_page: bool = False,
    ):
        """
        Draw an image and a caption centered under the image.
        - Provide either w or h (or both); if only one is given, FPDF keeps aspect ratio.
        - If center_on_page=True and w is given, image is horizontally centered in the content area.
        """
        # Decide X position
        if x is None:
            x = self.get_x()
        if y is None:
            y = self.get_y()

        # If we want page-centering and a known width, compute X from margins
        if center_on_page and w is not None:
            content_w = self.w - self.l_margin - self.r_margin
            x = self.l_margin + (content_w - w) / 2

        # Draw image
        self.image(img_path, x=x, y=y, w=w, h=h)

        # Compute the final drawn image height (needed for caption Y).
        # If h not provided but w is, we can approximate via intrinsic size if available,
        # otherwise rely on FPDF to preserve aspect ratio and just place caption after a gap.
        # Easiest robust approach: ask current Y after placing image:
        # (FPDF doesn't move the cursor for image(), so we compute using our inputs.)
        img_h = h
        if img_h is None and w is not None:
            # If you need exact height from the file’s aspect ratio, set h yourself,
            # or use PIL to read size. Here we’ll just place caption immediately after y + (h or 0).
            pass

        # Caption Y position
        cap_y = (y + (img_h or 0)) + gap

        # Draw caption within the image width, centered
        # (If you didn’t set w, choose a reasonable width for the caption.)
        cap_w = w if w is not None else (self.w - self.l_margin - self.r_margin)

        # Save original X to keep layout predictable after multi_cell
        old_x = self.get_x()
        self.set_xy(x, cap_y)
        self.multi_cell(cap_w, line_h, caption, align="C")
        # After multi_cell, X resets to left margin; restore a sensible X just below caption:
        self.set_x(old_x)

        # Return bottom Y so caller can continue below
        return self.get_y()
    

class LazyFigure:
    """Wrapper 
    """

    def __init__(self, analyzer_name, analyzer_method, *args, **kwargs):
        self.ana_name = analyzer_name
        self.ana_method = analyzer_method
        self.args = args
        self.kwarg = kwargs

    def __call__(self, ws) -> mpl.figure.Figure:

        ana = getattr(ws.analyzers, self.ana_name)
        method = getattr(ana, self.ana_method)
        fig = method(*self.args, **self.kwargs)


class Portfolio:
    _log = logging.getLogger('TriggerPrimitivesWorkspace')

    def __init__(self, img_folder: str, img_prefix: str=''):
        """A porfolio of named images
        OK, maybe the name is not quite right.
        """
        
        self.folder = Path(img_folder)
        self.prefix = img_prefix
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
        img_fname = (f'{self.prefix}_' if self.prefix else '')+f'{img_name}.{fmt}'
        img_path = self.folder / img_fname
        self.figures[img_name] = img_path
        # if fig is a function, generate the figure
        if callable(fig):
            self._log.info(f"Generating {img_name}")

            fig = fig()
        self.figures[img_name] = (fig, img_path)

        # TODO: decouple?
        self._log.info(f"Saving {img_name} as {img_path}")
        fig.savefig(img_path)
        plt.close(fig)

        return img_path