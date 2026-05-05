import textwrap

import mistletoe as mt

from fpdf import FPDF, HTML2FPDF
from pathlib import Path
from typing import Dict

_FONTS_DIR = Path(__file__).parent / "fonts"

_FONT_REGISTRY = [
    ("Raleway",       "",  "Raleway-Regular.ttf"),
    ("Raleway",       "B", "Raleway-Bold.ttf"),
    ("Lato",          "",  "Lato-Regular.ttf"),
    ("Lato",          "B", "Lato-Bold.ttf"),
    ("OpenSans",      "",  "OpenSans-Regular.ttf"),
    ("OpenSans",      "B", "OpenSans-Bold.ttf"),
    ("Roboto",        "",  "Roboto-Regular.ttf"),
    ("SourceCodePro", "",  "SourceCodePro-Medium.ttf"),
    ("SourceCodePro", "B", "SourceCodePro-Semibold.ttf"),
]


def load_report_fonts(pdf: FPDF) -> None:
    """Register bundled fonts with an FPDF instance.

    Silently skips any font file not found in the package fonts directory.
    """
    for family, style, filename in _FONT_REGISTRY:
        path = _FONTS_DIR / filename
        if path.exists():
            pdf.add_font(family, style, str(path))


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

    # def __init__(self, orientation = PageOrientation.PORTRAIT, unit = "mm", format = "A4", font_cache_dir = "DEPRECATED", *, enforce_compliance = None):
    #     super().__init__(orientation, unit, format, font_cache_dir, enforce_compliance=enforce_compliance)

    def __init__(self, md_defaults: Dict, **kwargs):
        self.md_defaults = md_defaults
        super().__init__(**kwargs)

    def footer(self):
        self.set_y(-15)
        self.set_font("Raleway", size=8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def add_slide(self, title, subtitle=None, **kwargs):
        self.add_page(**kwargs)
        md = [f"# **{title}**"]
        if subtitle:
            md += [f"### {subtitle}"]
        print(md)
        self.write_markdown('\n'.join(md))

    def write_markdown(self, text, **kwargs):
        """Write text in the document with markdown syntax."""
        text = textwrap.dedent(text)
        write_kwargs = self.md_defaults.copy()
        write_kwargs.update(kwargs)
        self.write_html(mt.markdown(text), **write_kwargs)

    def move_cursor(self, dx=0, dy=0):
        """Move the cursor by (dx, dy) relative to its current position.

        Args:
            dx (int, optional): cursor displacement along x. Defaults to 0.
            dy (int, optional): cursor displacement along y. Defaults to 0.
        """
        self.set_xy(self.get_x() + dx, self.get_y() + dy)

    def mark_cursor(self, size=3, text=None):
        """Draw a crosshair centered at the current cursor position.

        Args:
            size (int, optional): Crosshair size. Defaults to 3.
        """
        x, y = self.get_x(), self.get_y()
        self.set_draw_color(200, 0, 0)
        self.line(x - size, y, x + size, y)
        self.line(x, y - size, x, y + size)
        s = text or f"({x:.1f},{y:.1f})"
        self.text(x + 2, y - 2, s)

    def outline_next_cell(self, w, h):
        x, y = self.get_x(), self.get_y()
        self.set_draw_color(0, 100, 200)
        self.rect(x, y, w, h)

    def label_cursor(self, text=None):
        x, y = self.get_x(), self.get_y()
        s = text or f"({x:.1f},{y:.1f})"
        self.text(x + 2, y - 2, s)

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
        gap: float = 2.0,
        line_h: float = 5.0,
        center_on_page: bool = False,
    ):
        """Draw an image and a caption centered under it.

        Provide either w or h (or both); if only one is given, FPDF keeps the
        aspect ratio.  If center_on_page=True and w is given, the image is
        horizontally centered in the content area.

        Returns:
            Bottom Y position after the caption, so the caller can continue below.
        """
        if x is None:
            x = self.get_x()
        if y is None:
            y = self.get_y()

        if center_on_page and w is not None:
            content_w = self.w - self.l_margin - self.r_margin
            x = self.l_margin + (content_w - w) / 2

        self.image(img_path, x=x, y=y, w=w, h=h)

        cap_y = (y + (h or 0)) + gap
        cap_w = w if w is not None else (self.w - self.l_margin - self.r_margin)

        old_x = self.get_x()
        self.set_xy(x, cap_y)
        self.multi_cell(cap_w, line_h, caption, align="C")
        self.set_x(old_x)

        return self.get_y()
