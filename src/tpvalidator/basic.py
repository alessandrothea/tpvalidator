import click
from pathlib import Path
from rich import print
import math

from .utilities import load_data, load_info, calculate_angles, calculate_more_angles, compute_histogram_ratio
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


# name of branches in the TTree to read into the pandas df
MC_BRANCHES = ["event", "Eng", "Ekin", "startX", "startY", "startZ",  "Px", "Py", "Pz", "P"] + ['totQ_X', 'totQ_U', 'totQ_V', 'detQ_X', 'detQ_U', 'detQ_V']
TP_BRANCHES = ["event", "n_TPs", "TP_channel", "TP_startT", "TP_peakT", "TP_peakADC", 
               "TP_SADC", "TP_TOT", "TP_plane", "TP_TPC", "TP_trueX", "TP_trueY", 'TP_trueZ', 'TP_signal']

a4_landscape = (11.69,8.27)
a5_landscape = (8.27,5.85)


def equalize_ranges(df) -> pd.DataFrame:
    x = df.agg(['min', 'max'])
    med = (x.loc['max']+x.loc['min'])/2
    rng = (x.loc['max']-x.loc['min'])

    max_rng = rng.max()

    upper = med+max_rng/2
    lower = med-max_rng/2
    return pd.concat([upper, lower], axis=1).T


class BasicTPData:

    def __init__(self, data_path: Path):


        self.data_path = data_path

        # FIXME: The same ROOT file is opened 5 times, and the same tree is accessed 4 times.
        # Maybe we can do something better
        self.info = load_info(data_path)

        self.events = load_data(data_path, tree_name='triggerana/tree', branch_names=['event'])

        self.tps = load_data(data_path, tree_name='triggerana/tree', branch_names=TP_BRANCHES)
        self.mc = load_data(data_path, tree_name='triggerana/tree', branch_names=MC_BRANCHES)
        self.all = load_data(data_path, tree_name='triggerana/tree')

        self.ides = load_data(data_path, 'triggerana/qtree')
        self.waveforms = {}

        self._init_angles()

    def _init_angles(self) -> None:
       
        match self.detector_name():
            case 'dunevd10kt_3view_30deg_v5_refactored_1x8x6ref':
                det_type = 'vd'
            case _ as det_name:
                raise ValueError(f'detector type of detector {det_name} unknown')
            

        self.angles = self.mc[['event']].copy(deep=True)

        theta_y, theta_y_U, theta_y_V, theta_xz, theta_xz_U, theta_xz_V = calculate_angles(self.mc.Px, self.mc.Py, self.mc.Pz, self.mc.P, det_type)
        self.angles['theta_y'] = theta_y
        self.angles['theta_yU'] = theta_y_U
        self.angles['theta_yV'] = theta_y_V
        self.angles['theta_xz'] = theta_xz
        self.angles['theta_xzU'] = theta_xz_U
        self.angles['theta_xzV'] = theta_xz_V

        theta_drift, theta_beam, theta_coll, theta_u, theta_v, phi_coll, phi_ind_u, phi_ind_v = calculate_more_angles(self.mc.Px, self.mc.Py, self.mc.Pz, self.mc.P)
        self.angles['theta_drift'] = theta_drift
        self.angles['theta_beam'] = theta_beam
        self.angles['theta_coll'] = theta_coll
        self.angles['theta_u'] = theta_u
        self.angles['theta_v'] = theta_v
        self.angles['phi_coll'] = phi_coll
        self.angles['phi_ind_u'] = phi_ind_u
        self.angles['phi_ind_v'] = phi_ind_v

    def detector_name(self):
        print(self.info)
        if 'geo' in self.info:
            return self.info['geo']['detector']
        elif 'detector' in self.info:
            return self.info['detector']
        else:
            raise KeyError(f"Unable to find detector name in tpg processing info")
        
    def tp_algorithm(self):
        return self.info['tpg']['tool']
    

    def tp_threshold(self, plane: int):
        if plane in [0,1,2]:
            return self.info['tpg'][f'threshold_tpg_plane{plane}']
        else:
            return ValueError(f"Invalid plane id: {plane}")
        
    def tp_backtracker_offset(self, plane: int):
        plane_map = {
            0: 'U_window_offset',
            1: 'V_window_offset',
            2: 'X_window_offset',
        }

        if not plane in plane_map:
            return KeyError(f"Plane '{plane}' not known")
        
        return self.info['tptree'][plane_map[plane]]


### Helper functions

def find_axis_aligned_events(tp_data):
    """Returns the 3 events with the highest combination of 
    energy and alignment to the x, y and z detector axis.

    Returns:
        _type_: _description_
    """
    return [
        int(tp_data.mc[(s / tp_data.mc.P).abs() > 0.95].sort_values('P', ascending=False).iloc[0]['event'])
        for s in (tp_data.mc.Px, tp_data.mc.Py, tp_data.mc.Pz)
    ]


## Report plots

def plot_data_summary(tp_data: BasicTPData, title_size=None, figsize=a4_landscape, **kwargs) -> Figure:
    """Plot the histogram for all variables in the TP tree

    Args:
        tp_data (BasicTPData): _description_
        title_size (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    tp_data.all.hist(bins=50, figsize=figsize, **kwargs)
    fig = plt.gcf()
    fig.suptitle("Tree variables overview", fontsize=title_size)
    return fig


def plot_tp_summary(tp_data: BasicTPData, title_size=None, figsize=a4_landscape, **kwargs) -> Figure:
    tp_data.tps.hist(bins=50, figsize=figsize, color='xkcd:dusk blue', **kwargs)
    fig = plt.gcf()
    fig.suptitle("TP distributions", fontsize=title_size)
    return fig


def plot_mc_summary(tp_data: BasicTPData, title_size=None, figsize=a4_landscape, **kwargs) -> Figure:
    tp_data.mc.hist(bins=50, figsize=figsize, color='k', **kwargs)
    fig = plt.gcf()
    fig.suptitle("MC distributions", fontsize=title_size)
    return fig


def plot_angles_summary(tp_data: BasicTPData, title_size=None, figsize=a4_landscape, **kwargs) -> Figure:
    """_summary_

    Args:
        tp_data (BasicTPData): _description_
        title_size (_type_, optional): _description_. Defaults to None.
        figsize (_type_, optional): _description_. Defaults to a4_landscape.

    Returns:
        Figure: _description_
    """
    tp_data.angles.hist(bins=50, figsize=figsize, color='xkcd:algae', **kwargs)
    fig = plt.gcf()
    fig.suptitle("Angle distributions", fontsize=title_size)
    return fig


def draw_tps_point_of_origin(ax : plt.Axes, tp_data: BasicTPData, ev_num: int, ev_label: str = "", is_signal: bool = None) -> None:
    """Draw the trigger primitives point of origin, based on backtracked informations

    Args:
        ax (plt.Axes): _description_
        tp_data (BasicTPData): _description_
        ev_num (int): _description_
        ev_label (str, optional): _description_. Defaults to "".
        is_signal (bool, optional): _description_. Defaults to None.
    """

    vmax = tp_data.tps[(tp_data.tps.event == ev_num) & (tp_data.tps.TP_signal == 1)].TP_SADC.max()/5
    vmin = tp_data.tps[(tp_data.tps.event == ev_num) & (tp_data.tps.TP_signal == 1)].TP_SADC.min()


    tps_selection = (tp_data.tps.event == ev_num)
    if not is_signal is None:
        tps_selection &= tp_data.tps.TP_signal == is_signal
    tps=tp_data.tps[tps_selection]
    # tps = tp_data.tps[(tp_data.tps.event == ev_num)]
    # if not is_signal is None:
        # tps = tps[tp_data.tps.TP_signal == is_signal]

    # equalize the range
    ranges = equalize_ranges(tps[['TP_trueX', 'TP_trueY', 'TP_trueZ']])
    
    # Draw 3D points
    # ax.scatter(tps.TP_trueY, tps.TP_trueZ, tps.TP_trueX)
    ax.scatter(tps.TP_trueY, tps.TP_trueZ, tps.TP_trueX, s=tps.TP_TOT/2, c=tps.TP_SADC, vmin=vmin, vmax=vmax)

    # # Add projections on the YZ plane (CRP) and XY plane (collection/drift)
    ax.scatter(np.full_like(tps.TP_trueY, ranges.TP_trueY[0]), tps.TP_trueZ, tps.TP_trueX, s=tps.TP_TOT/2, c='gray')
    ax.scatter(tps.TP_trueY, tps.TP_trueZ, np.full_like(tps.TP_trueX, ranges.TP_trueX[0]), s=tps.TP_TOT/2, c='gray')
    ax.scatter(tps.TP_trueY, np.full_like(tps.TP_trueZ, ranges.TP_trueZ[1]), tps.TP_trueX, s=tps.TP_TOT/2, c='gray')

    ax.set_xlim3d(*list(ranges.TP_trueY))
    ax.set_ylim3d(*list(ranges.TP_trueZ))
    ax.set_zlim3d(*list(ranges.TP_trueX))

    ax.set_xlabel("y (collection)")
    ax.set_ylabel("z (beam)")
    ax.set_zlabel("x (drift)")
    ax.set_title(ev_label)


def plot_3dev_points_of_origin(tp_data: BasicTPData, figsize=a4_landscape, title_size=None) -> Figure:
    """
    """
    events=find_axis_aligned_events(tp_data)
    # Event display 3d
    vmax = tp_data.tps[(tp_data.tps.event.isin(events)) & (tp_data.tps.TP_signal == 1)].TP_SADC.max()/5
    vmin = tp_data.tps[(tp_data.tps.event.isin(events)) & (tp_data.tps.TP_signal == 1)].TP_SADC.min()

    # Row labels on the left side
    # FIXME: duplicated in the next method
    ev_labels = []
    ev_infos = []
    for e, l in zip(events, ['x', 'y', 'z']):
        ev_row = tp_data.mc[tp_data.mc.event==e]
        Pax = ev_row[f'P{l}']
        ev_labels.append(f"Event {int(e)}")
        ev_infos.append(f"""
    Particle direction parallel to $\\hat{{{l}}}$
    $E_{{kin}}={float(ev_row['Ekin'].values):.2f}$ MeV
    $P_{l}$={float(Pax.values)*1000:.2f} MeV
    Particle origin: ({float(ev_row['startX'].values):.1f}, {float(ev_row['startY'].values):.1f}, {float(ev_row['startZ'].values):.1f})
    """)


    # fig, axes = plt.subplot_mosaic([[0,1,2],[3,'.','.']], subplot_kw={'projection': '3d'}, figsize=figsize)
    fig, axes = plt.subplots(2,3, subplot_kw={'projection': '3d'}, gridspec_kw={'height_ratios': [3, 1]}, figsize=figsize)

    
    fig.suptitle(f"TP point of origin for highest energy events along X, Y and Z)", fontsize=title_size)

    for e, en_num in enumerate(events):

        # TODO: switch to using 'draw_tps_point_of_origin'
        ax = axes[0][e]
        ev_label = ev_labels[e]
        # mcf = tp_data.mc[tp_data.mc.event==en_num]
        tps = tp_data.tps[(tp_data.tps.event == en_num) & (tp_data.tps.TP_signal == 1)]

        # equalize the range
        ranges = equalize_ranges(tps[['TP_trueX', 'TP_trueY', 'TP_trueZ']])
        
        # Draw 3D points
        ax.scatter(tps.TP_trueY, tps.TP_trueZ, tps.TP_trueX, s=tps.TP_TOT/2, c=tps.TP_SADC, vmin=vmin, vmax=vmax)

        # Add projections on the YZ plane (CRP) and XY plane (collection/drift)
        ax.scatter(np.full_like(tps.TP_trueY, ranges.TP_trueY[0]), tps.TP_trueZ, tps.TP_trueX, s=tps.TP_TOT/2, c='gray')
        ax.scatter(tps.TP_trueY, tps.TP_trueZ, np.full_like(tps.TP_trueX, ranges.TP_trueX[0]), s=tps.TP_TOT/2, c='gray')
        ax.scatter(tps.TP_trueY, np.full_like(tps.TP_trueZ, ranges.TP_trueZ[1]), tps.TP_trueX, s=tps.TP_TOT/2, c='gray')

        ax.set_xlim3d(*list(ranges.TP_trueY))
        ax.set_ylim3d(*list(ranges.TP_trueZ))
        ax.set_zlim3d(*list(ranges.TP_trueX))

        ax.set_xlabel("y (collection)")
        ax.set_ylabel("z (beam)")
        ax.set_zlabel("x (drift)")
        ax.set_title(ev_label)

        ax = axes[1,e]

        ax.axis('off') # off
        ax.annotate(ev_infos[e], (0.0, 0.7), xycoords='axes fraction', va='center')


    return fig


def plot_3dev_plane_view(tp_data: BasicTPData, figsize=a4_landscape, title_size=None) -> Figure:
    #-----
    # Event display

    events=find_axis_aligned_events(tp_data)
    title=['U','V', "X"]
    # TODO: move to BasicTPDataClass
    vmax = tp_data.tps[(tp_data.tps.event.isin(events)) & (tp_data.tps.TP_signal == 1)].TP_SADC.max()/5
    vmin = tp_data.tps[(tp_data.tps.event.isin(events)) & (tp_data.tps.TP_signal == 1)].TP_SADC.min()

    ev_labels = []
    for e, l in zip(events, ['x', 'y', 'z']):
        ev_row = tp_data.mc[tp_data.mc.event==e]
        Pax = ev_row[f'P{l}']
        ev_labels.append(f"Event {int(e)},\n[$\\vec{{P}} \\parallel \\hat{{{l}}}$, $P_{l}$={float(Pax.values)*1000:.2f} MeV]")

    fig, ax = plt.subplots(len(events),3, figsize=figsize)
    fig.suptitle(f"TP plane/time for highest energy events along X, Y and Z", fontsize=title_size)
        
    for e, ev_num in enumerate(events):
        mcf = tp_data.mc[tp_data.mc.event==ev_num]
        for plane in [0,1,2]:
            x = tp_data.tps[(tp_data.tps.event == ev_num) & (tp_data.tps.TP_plane == plane) & (tp_data.tps.TP_signal == 1)]
            # x = df_tps[(df_tps.event == event) & (df_tps.TP_plane == plane)]
            s = ax[e][plane].scatter(x.TP_channel, x.TP_peakT,s=x.TP_TOT/2, c=x.TP_SADC, vmin=vmin, vmax=vmax, label = f"{int(mcf.Ekin.values)} MeV electron")
            if plane == 2:
                plt.colorbar(s, ax=ax[e][plane], label="SADC")

        for i in range(0,3):
            ax[e][i].grid(alpha=0)
            ax[e][i].set_title(title[i])
            ax[e][i].set_xlabel("channel ID")
            ax[e][i].set_ylabel("time")

            ax[e][i].legend()

    for i, label in enumerate(ev_labels):
        # Place text to the left of the row
        fig.text(0.02, 0.8 - i * 0.3, label, va='center', ha='left', fontsize=12, rotation=90)

    return fig



def plot_tp_plane(tp_data: BasicTPData, figsize=a4_landscape, title_size=None,  **kwargs) -> Figure:
    # Plane distribution
    fig, axes = plt.subplots(1,3,figsize=figsize, **kwargs)
    ax = axes[0]
    ax.grid()
    ax.hist([tp_data.tps[tp_data.tps.TP_signal == 1].TP_plane, tp_data.tps[tp_data.tps.TP_signal == 0].TP_plane], histtype='bar', stacked=True)
    ax.set_title("All TPs")

    ax = axes[1]
    ax.grid()
    ax.hist((tp_data.tps[tp_data.tps.TP_signal == 1].TP_plane, []), histtype='bar', stacked=True)
    ax.set_title("Signal TPs")

    ax = axes[2]
    ax.grid()
    ax.hist(([], tp_data.tps[tp_data.tps.TP_signal == 0].TP_plane), histtype='bar', stacked=True)
    ax.set_title("Noise TPs")


    fig.suptitle("TP distribution by plane", fontsize=title_size)

    return fig


def plot_charge(tp_data: BasicTPData, title_size=None,  **kwargs) -> Figure:
    #-----
    # Charge distributions and ratios

    x = tp_data.mc[['totQ_X', 'totQ_U', 'totQ_V', 'detQ_X', 'detQ_U', 'detQ_V']].copy()
    x['rQ_X'] = x.detQ_X/x.totQ_X
    x['rQ_U'] = x.detQ_U/x.totQ_U
    x['rQ_V'] = x.detQ_V/x.totQ_V

    x.hist(bins=100, figsize=a4_landscape, layout=(3,3), color='xkcd:dark salmon', **kwargs)
    fig = plt.gcf()
    fig.suptitle("Charge distributions", fontsize=title_size)
    return fig


def plot_charge2(tp_data: BasicTPData, title_size=None, figsize=a4_landscape, **kwargs) -> Figure:
    #-----
    # Charge distributions and ratios

    x = tp_data.mc[['totQ_X', 'totQ_U', 'totQ_V', 'detQ_X', 'detQ_U', 'detQ_V']].copy()
    x['rQ_X'] = x.detQ_X/x.totQ_X
    x['rQ_U'] = x.detQ_U/x.totQ_U
    x['rQ_V'] = x.detQ_V/x.totQ_V

    fig,ax = plt.subplots(3,3,figsize=figsize)
    
    ax[0, 0].hist(x.totQ_X, bins=100, color='xkcd:dark salmon', **kwargs)
    ax[0, 0].grid()
    ax[0, 0].set_title("totQ_X")

    ax[0, 1].hist(x.totQ_U, bins=100, color='xkcd:dark salmon', **kwargs)
    ax[0, 1].grid()
    ax[0, 1].set_title("totQ_U")

    ax[0, 2].hist(x.totQ_V, bins=100, color='xkcd:dark salmon', **kwargs)
    ax[0, 2].grid()
    ax[0, 2].set_title("totQ_V")

    ax[1, 0].hist(x.detQ_X, bins=100, color='xkcd:dark salmon', **kwargs)
    ax[1, 0].grid()
    ax[1, 0].set_title("detQ_X")

    ax[1, 1].hist(x.detQ_U, bins=100, color='xkcd:dark salmon', **kwargs)
    ax[1, 1].grid()
    ax[1, 1].set_title("detQ_U")

    ax[1, 2].hist(x.detQ_V, bins=100, color='xkcd:dark salmon', **kwargs)
    ax[1, 2].grid()
    ax[1, 2].set_title("detQ_V")

    bin_centers, ratio, ratio_err, bins = compute_histogram_ratio(x.detQ_X, x.totQ_X, bins=100)
    ax[2, 0].errorbar(
        bin_centers, ratio, yerr=ratio_err,
        fmt='o', color='xkcd:dark salmon', markersize=4, capsize=2, ecolor='xkcd:dark salmon'
    )
    ax[2, 0].grid()
    ax[2, 0].set_title("detQ_X/totQ_X")

    bin_centers, ratio, ratio_err, bins = compute_histogram_ratio(x.detQ_U, x.totQ_U, bins=100)
    ax[2, 1].errorbar(
        bin_centers, ratio, yerr=ratio_err,
        fmt='o', color='xkcd:dark salmon', markersize=4, capsize=2, ecolor='xkcd:dark salmon'
    )
    ax[2, 1].grid()
    ax[2, 1].set_title("detQ_U/totQ_U")

    bin_centers, ratio, ratio_err, bins = compute_histogram_ratio(x.detQ_V, x.totQ_V, bins=100)
    ax[2, 2].errorbar(
        bin_centers, ratio, yerr=ratio_err,
        fmt='o', color='xkcd:dark salmon', markersize=4, capsize=2, ecolor='xkcd:dark salmon'
    )
    ax[2, 2].grid()
    ax[2, 2].set_title("detQ_V/totQ_V")

    fig.suptitle("Charge distributions", fontsize=title_size)
    return fig


def plot_sadc_vs_ekin(tp_data: BasicTPData, title_size=None, figsize=a4_landscape, **kwargs) -> Figure:
    # Correlation of SADC (collection) and Kinetic energy
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    fig.suptitle("SADC vs Ekin", fontsize=title_size)

    tp_coll = tp_data.tps[tp_data.tps.TP_plane==2]
    sadc_event = tp_coll.groupby('event').TP_SADC.sum()

    s = ax.scatter(tp_data.mc.Ekin, sadc_event, c=tp_data.tps[tp_data.tps.TP_plane==2].groupby('event').TP_trueX.mean(), marker='x')

    plt.colorbar(s, label="drift origin [true X]")
    ax.set_ylabel("total collection SADC")
    ax.set_xlabel("particle kinetic energy [MeV]")
    
    return fig

def plot_sadc_vs_totQ(tp_data: BasicTPData, title_size=None, figsize=a4_landscape, **kwargs) -> Figure:
    # Correlation of SADC (collection) and Kinetic energy
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    fig.suptitle("SADC vs totQ_X", fontsize=title_size)

    tp_coll = tp_data.tps[tp_data.tps.TP_plane==2]
    sadc_event = tp_coll.groupby('event').TP_SADC.sum()

    s = ax.scatter(tp_data.mc.totQ_X, sadc_event, c=tp_data.tps[tp_data.tps.TP_plane==2].groupby('event').TP_trueX.mean(), marker='x')
    plt.colorbar(s, label="drift origin [true X]")
    ax.set_ylabel("total collection SADC")
    ax.set_xlabel("total collected charge [AU]")
    
    return fig

def plot_energy_qx_sadc(tp_data: BasicTPData, title_size=None, figsize=a4_landscape, **kwargs):

    tp_coll = tp_data.tps[tp_data.tps.TP_plane==2]
    total_sadc_coll = tp_coll.groupby('event').TP_SADC.sum()
    mean_true_x = tp_coll.groupby('event').TP_trueX.mean()

    fig, axes = plt.subplot_mosaic([['EvsQ','QvsSADC'],['EvsSADC','.']],figsize=figsize, **kwargs)
    fig.suptitle("Energy, Charge and SADC", fontsize=title_size)

    ax = axes['EvsSADC']
    s = ax.scatter(tp_data.mc.Ekin, total_sadc_coll, c=mean_true_x, marker='x')
    plt.colorbar(s, label="drift origin [true X]")
    ax.set_xlabel("particle kinetic energy [MeV]")
    ax.set_ylabel("total collection SADC [counts]")

    ax = axes['QvsSADC']
    s = ax.scatter(tp_data.mc.totQ_X, total_sadc_coll, c=mean_true_x, marker='x')
    plt.colorbar(s, label="drift origin [true X]")
    ax.set_xlabel("total collected charge [AU]")
    ax.set_ylabel("total collection SADC [counts]")

    ax = axes['EvsQ']
    s = ax.scatter(tp_data.mc.Ekin, tp_data.mc.totQ_X, c=mean_true_x, marker='x')
    plt.colorbar(s, label="drift origin [true X]")
    ax.set_xlabel("particle kinetic energy [MeV]")
    ax.set_ylabel("total collected charge [AU]")

    return fig



def plot_sadc_coll_vs_ind(tp_data: BasicTPData, title_size=None, figsize=a4_landscape, **kwargs) -> Figure:
  
    # Correlation of SADC vs induction planes
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    ax.scatter(tp_data.tps[tp_data.tps.TP_plane==2].groupby('event').TP_SADC.sum(), tp_data.tps[tp_data.tps.TP_plane==0].groupby('event').TP_SADC.sum() , label ='U')
    ax.scatter(tp_data.tps[tp_data.tps.TP_plane==2].groupby('event').TP_SADC.sum(), tp_data.tps[tp_data.tps.TP_plane==1].groupby('event').TP_SADC.sum() , label ='V')
    ax.legend()
    fig.suptitle("SADC, collection vs induction planes", fontsize=title_size)
    ax.set_xlabel("Total collection SADC per event")
    ax.set_ylabel("Total induction SADC per event")

    return fig


def plot_tp_peak_vs_thetay(tp_data: BasicTPData, title_size=None, figsize=a4_landscape, **kwargs) -> Figure:
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    b = (50,50)

    ax.hist2d(tp_data.angles[tp_data.mc.event.isin(tp_data.tps[(tp_data.tps.TP_plane==0) &(tp_data.tps.TP_signal==1)].event.unique())].theta_yU, 
        tp_data.tps[(tp_data.tps.TP_plane==0) &(tp_data.tps.TP_signal==1)].groupby('event').TP_peakADC.mean(),
        cmap='Greys',bins=b, norm = LogNorm())
    fig.suptitle(r"TP peak mean vs incident angle $\theta_y$", fontsize=title_size)
    ax.set_ylabel('mean induction TP peak ADC [ADC]')
    ax.set_xlabel(r"$\theta_y [^{o}]$")

    return fig



def _plot_var_vs_angle(tp_data: BasicTPData, var, title, var_label, title_size=None, figsize=a4_landscape, **kwargs) -> Figure:
    # Correlation of SADC and angles
    fig, axes = plt.subplots(2,2,figsize=figsize, **kwargs)

    # Data definition
    av_true_x = tp_data.tps[tp_data.tps.TP_plane==2].groupby('event').TP_trueX.mean()

    # Figure parameters
    fig_title = title
    color_label = "drift origin [true X]"
    y_label = var_label
    y_vals = var
    c_vals = av_true_x
    cmap='viridis'

    ax = axes[0][0]
    s = ax.scatter(tp_data.angles.theta_y, y_vals, c=c_vals, cmap=cmap, marker='x')
    plt.colorbar(s, label=color_label)
    ax.set_ylabel(y_label)
    ax.set_xlabel("$\\theta_y$ [collection]")

    ax = axes[0][1]
    s = ax.scatter(tp_data.angles.theta_beam, y_vals, c=c_vals, cmap=cmap, marker='x')
    plt.colorbar(s, label=color_label)
    ax.set_ylabel(y_label)
    ax.set_xlabel("$\\theta_z$ [beam]")
    
    ax = axes[1][0]
    s = ax.scatter(tp_data.angles.theta_drift, y_vals, c=c_vals, cmap=cmap, marker='x')
    plt.colorbar(s, label=color_label)
    ax.set_ylabel(y_label)
    ax.set_xlabel("$\\theta_{x}$ [drift]")
    
    ax = axes[1][1]
    ax.axis('off')

    fig.suptitle(fig_title, fontsize=title_size)

    return fig

def plot_sadc_vs_angle(tp_data: BasicTPData, title_size=None, figsize=a4_landscape, **kwargs):
    tp_coll = tp_data.tps[tp_data.tps.TP_plane==2]
    return _plot_var_vs_angle(tp_data, tp_coll.groupby('event').TP_SADC.sum(), "Sum(SADC) vs angle (collection TPs only)", "Total SADC", title_size=title_size, figsize=figsize, **kwargs )


def plot_qcoll_vs_angle(tp_data: BasicTPData, title_size=None, figsize=a4_landscape, **kwargs):

    return _plot_var_vs_angle(tp_data, tp_data.mc.totQ_X, "Collection Charge (totQ_X) vs angle (collection TPs only)", "Collection Charge", title_size=title_size, figsize=figsize, **kwargs )


## Commandline report generator
# TODO: move out
@click.command()
@click.option("-i", "--interactive", type=bool, is_flag=True, default=False)
@click.option("-o", "--output", type=str, default=None)
@click.argument("data_path", type=click.Path(exists=True))
def main(data_path, interactive, output) -> None:
    print("Hello from tpvalidator!")

    data_path = Path(data_path)
    report_name = output if not output is None else f'tp_val_{data_path.stem}.pdf'
    data = BasicTPData(data_path)


    if not interactive:
        a4_landscape = (11.69,8.27)
        a5_landscape = (8.27,5.85)
        title_size = 20
        print(f"Saving report {report_name}")
        with PdfPages(report_name) as pdf:

            
            fig = plt.figure(figsize=a4_landscape)
            fig.clf()
            title_text = "Trigger Primitives Simulation:\nBasic Overview Plots"
            meta_text = (
                f"""
    File: {data_path}
    Number of events: {len(data.events)}
    """
            )
            fig.text(0.1, 0.95, title_text, ha='left', va='top', wrap=True, fontsize=30)
            fig.text(0.1, 0.85, meta_text, ha='left', va='top', wrap=True, fontsize=20)
            fig.tight_layout()
            pdf.savefig()

            # summary
            fig = plot_data_summary(data, figsize=a4_landscape)
            fig.tight_layout()
            pdf.savefig()

            # MC plots
            fig = plot_mc_summary(data, figsize=a4_landscape)
            fig.tight_layout()
            pdf.savefig()

            # TP variables distributions
            fig = plot_tp_summary(data, figsize=a4_landscape)
            fig.tight_layout()
            pdf.savefig()
            
            fig = plot_angles_summary(data, title_size=title_size, figsize=a4_landscape)
            fig.tight_layout()
            pdf.savefig()

            fig = plot_3dev_points_of_origin(data, title_size=title_size, figsize=a4_landscape)
            fig.tight_layout()
            pdf.savefig() 

            fig = plot_3dev_plane_view(data, title_size=title_size, figsize=a4_landscape)
            fig.tight_layout()
            pdf.savefig()

            fig = plot_tp_plane(data, title_size=title_size, figsize=a4_landscape)
            fig.tight_layout()
            pdf.savefig()

            fig = plot_charge(data, title_size=title_size)
            fig.tight_layout()
            pdf.savefig()

            # fig = plot_charge2(data, title_size=title_size, figsize=a4_landscape)
            # fig.tight_layout()
            # pdf.savefig()


            # fig = plot_sadc_vs_ekin(data, title_size=title_size, figsize=a4_landscape)
            # fig.tight_layout()
            # pdf.savefig()


            fig = plot_energy_qx_sadc(data, title_size=title_size, figsize=a4_landscape)
            fig.tight_layout()
            pdf.savefig()

            fig = plot_qcoll_vs_angle(data, title_size=title_size, figsize=a4_landscape)
            fig.tight_layout()
            pdf.savefig()
            
            fig = plot_sadc_vs_angle(data, title_size=title_size, figsize=a4_landscape)
            fig.tight_layout()
            pdf.savefig()


            fig = plot_sadc_coll_vs_ind(data, title_size=title_size, figsize=a4_landscape)
            fig.tight_layout()
            pdf.savefig()

            fig = plot_tp_peak_vs_thetay(data, title_size=title_size, figsize=a4_landscape)
            fig.tight_layout()
            pdf.savefig()

        print("Done")

    if interactive:
        import IPython
        IPython.embed(colors='linux')
