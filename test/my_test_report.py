#!/usr/bin/env python


import matplotlib as mpl
import matplotlib.pyplot as plt



def make_my_puny_plot():

    # Add figsize?
    def make_figure():
        fig, ax = plt.subplots()
        ax.plot([1,2,3],[6,5,4],antialiased=True,linewidth=2,color='red',label='a curve')
        ax.set_title(r'$\omega$')
        fig.tight_layout()

    title = "A test plot"
    body = """This is a plot that talks about hwo beautiful the Ar39 distributions are in The VD detector
    <ul>
        <li> this is a bullet
        <li> and this is another
    </ul>
    """

    return make_figure, title, body


class REport:

    def __init__(self, fig_dir: str='tmp'):

        self.fig_dir = fig_dir

    def make_figures(self):
        pass

    def make_document():
        pass

    

rep = Report()

rep.make_figures()
rep.make_document()