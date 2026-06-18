import matplotlib.pyplot as plt
from contextlib import contextmanager

@contextmanager
def figure_manager(ax:object=None, **kwargs):
    create_fig = ax is None
    fig, ax = plt.subplots(**kwargs) if create_fig else (ax.figure, ax)
    try:
        yield fig, ax
    finally:
        # Code to release resource, e.g.:
        if create_fig:
            fig.tight_layout()