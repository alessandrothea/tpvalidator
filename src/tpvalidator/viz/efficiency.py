import matplotlib.pyplot as plt
import numpy as np


def plot_roc(df_eff, x: str, y: str, ax=None, label=None, refcuts=None,
             xlabel=None, ylabel=None, title=None, **fig_kwargs):
    """Plot a ROC curve from a cut-efficiency dataframe.

    Parameters
    ----------
    df_eff : pd.DataFrame
        DataFrame indexed by cut value; columns must include *x* and *y*.
    x : str
        Column name for the background efficiency (horizontal axis).
    y : str
        Column name for the signal efficiency (vertical axis).
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure is created when None.
    label : str, optional
        Legend label for the ROC curve.
    refcuts : list, optional
        Cut values (must be valid index labels in *df_eff*) to mark with
        triangle markers. Defaults to no markers.
    xlabel, ylabel : str, optional
        Axis labels. Default to the column names *x* and *y*.
    title : str, optional
        Figure suptitle. Applied only when a new figure is created.
    **fig_kwargs
        Forwarded to ``plt.subplots`` when creating a new figure
        (e.g. ``figsize=(6, 5)``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if refcuts is None:
        refcuts = []

    array_x = df_eff[x].values
    array_y = df_eff[y].values

    order = np.argsort(array_x)
    array_x = array_x[order]
    array_y = array_y[order]

    aoc = np.trapezoid(array_y, array_x)

    create_fig = ax is None
    fig, ax = plt.subplots(**fig_kwargs) if create_fig else (ax.figure, ax)
    ax.plot(array_x, array_y, lw=2, label=label)

    if refcuts:
        cmap = plt.get_cmap('Dark2')
        for i, refcut in enumerate(refcuts):
            xp, yp = df_eff.loc[refcut][x], df_eff.loc[refcut][y]
            ax.plot(xp, yp, 'v', markersize=10, label=f'$\\bf SOT \\geq {refcut}$\nAr39={xp:.2e}\ne-={yp:.2e}', c=cmap(i))

    ax.set_ylabel(ylabel if ylabel is not None else y)
    ax.set_xlabel(xlabel if xlabel is not None else x)

    aoc_box_props = dict(facecolor='white', alpha=1)
    ax.text(0.9, 0.1, f"AOC = {aoc:.3f}", transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right', bbox=aoc_box_props)

    ax.legend()

    if create_fig:
        ax.grid(True)
        if title is not None:
            fig.suptitle(title)
        fig.tight_layout()

    return fig