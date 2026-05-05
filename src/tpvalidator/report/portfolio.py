import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class LazyFigure:
    """Deferred figure generator — resolves an analyzer method call at render time."""

    def __init__(self, analyzer_name, analyzer_method, *args, **kwargs):
        self.ana_name = analyzer_name
        self.ana_method = analyzer_method
        self.args = args
        self.kwargs = kwargs

    def __call__(self, ws) -> Figure:
        ana = getattr(ws.analyzers, self.ana_name)
        method = getattr(ana, self.ana_method)
        return method(*self.args, **self.kwargs)


class Portfolio:
    """Registry of named matplotlib figures saved to disk for report generation.

    Figures are stored under a single folder with a consistent naming scheme:
    ``{prefix}_{name}.{fmt}`` (or ``{name}.{fmt}`` when no prefix is set).

    Typical workflow::

        pf = Portfolio("output/figs", img_prefix="ar39")
        pf.add_figure("adc_dist", fig)                  # saves ar39_adc_dist.svg
        pf.add_figure("xyz_pos", draw_fn, fmt="png")    # lazy: calls draw_fn()
        pdf.image(pf.path("adc_dist"))                  # retrieve path for PDF
    """

    _log = logging.getLogger('Portfolio')

    def __init__(self, img_folder: str, img_prefix: str = ''):
        """
        Args:
            img_folder: Directory where figure files are written.
                Created (including parents) if it does not already exist.
            img_prefix: Optional prefix prepended to every filename, e.g.
                ``"ar39"`` produces ``ar39_adc_dist.svg``.
                Leave empty to omit the prefix.
        """
        self.folder = Path(img_folder)
        self.prefix = img_prefix
        self.figures = {}

        self.folder.mkdir(parents=True, exist_ok=True)

    def add_figure(self, img_name: str, fig: Figure | Callable[[], Figure], fmt: str = 'svg', *fig_args, **fig_kwargs) -> Path:
        """Save a figure and register it under *img_name*.

        Args:
            img_name: Logical name used as the registry key and embedded in the
                filename: ``{prefix}_{img_name}.{fmt}``.
            fig: A ``matplotlib.figure.Figure``, or any callable that takes no
                arguments and returns one.  Callables are invoked lazily here.
            fmt: File format passed to ``Figure.savefig`` (e.g. ``'svg'``,
                ``'png'``).  Defaults to ``'svg'``.

        Returns:
            Path to the saved figure file.
        """
        img_fname = (f'{self.prefix}_' if self.prefix else '') + f'{img_name}.{fmt}'
        img_path = self.folder / img_fname

        if callable(fig):
            # self._log.info(f"Generating '{img_name}' with arguments {fig_args}")
            print(f"Generating '{img_name}' with arguments {fig_args} {fig_kwargs}")
            fig = fig(*fig_args, **fig_kwargs)

        self.figures[img_name] = (fig, img_path)
        self._log.info(f"Saving {img_name} as {img_path}")
        fig.savefig(img_path)
        plt.close(fig)

        return img_path

    def path(self, name: str, fmt: str = 'svg') -> Path:
        """Return the saved path for a named figure.

        If the figure was registered in this session, returns its stored path
        (preserving the original format). Falls back to computing the path from
        the naming convention using ``fmt`` — useful when the portfolio is
        reconstructed from a previous run with ``--no-figs``.
        """
        if name in self.figures:
            entry = self.figures[name]
            return entry[1] if isinstance(entry, tuple) else entry
        img_fname = (f'{self.prefix}_' if self.prefix else '') + f'{name}.{fmt}'
        return self.folder / img_fname
