import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..workspace import TriggerPrimitivesWorkspace
from typing import Tuple, Optional, Union, Sequence, Dict, List
from ..utils import subplot_autogrid
from rich import print

class BackTrackerPlotter:
    def __init__(self, ws: TriggerPrimitivesWorkspace, ev_num: int):

        if ws.simides is None:
            raise RuntimeError(f"No IDE data available in '{ws._data_path}'")

        self.ev_num = ev_num
        self.ws = ws

        self.inspect_tps = self.ws.tps[(self.ws.tps.event == ev_num)]
        # Focus on ides from this event only
        self.event_ides = self.ws.simides[self.ws.simides.event == ev_num]

        self.tpg_info = ws.info['tpg'][ws.tp_maker_name]
        self.tp_thresholds = [self.tpg_info[f'threshold_tpg_plane{i}'] for i in range(3)]
        self.bt_offsets = [ws.info['backtracker'][self.tpg_info['tool']][f'offset_{v}'] for v in ('U', 'V', 'X')]

        self.waveforms = self.ws.get_rawadcs(ev_num)
        if self.waveforms is None:
            print(f"[yellow]Warning: no waveform data found in workspace for event {ev_num}[/yellow]")


    def plot_tps_vs_ides(self, tps: Union[List[int], pd.DataFrame], layout: str = 'grid', figsize=(10, 10)):

        # IDEs search range around the TP
        ide_search_extension = 250
        tp_win_extension = 3.

        inspect_tps = self.inspect_tps
        event_ides = self.event_ides
        waves = self.waveforms
        tp_thresholds = self.tp_thresholds
        offsets = self.bt_offsets

        match layout:
            case 'lin':
                fig, axes = plt.subplots(1, len(tps), figsize=figsize)
            case 'grid':
                fig, axes = subplot_autogrid(len(tps), figsize=figsize)
            case _:
                raise ValueError(f"Layout value '{layout} unknown")

        # Color selection
        colors = mpl.colormaps['tab20'].colors
        ide_color = colors[0]
        ide_color_fill = colors[1]
        wave_color = colors[2]
        thres_color = colors[6]
        thres_fill_color = colors[7]
        tp_color = colors[7]
        match_color = colors[8]

        # Make a small local copy
        if isinstance(tps, list):
            selected_tps = inspect_tps.iloc[tps].copy()
        elif isinstance(tps, pd.DataFrame):
            selected_tps = tps.copy()
        else:
            raise TypeError(f"Argument 'tps' of unsupported type {type(tps)}")

        for i, (index, tp) in enumerate(selected_tps.iterrows()):
            ax = axes[i]
            tp_plane = int(tp.readout_view)
            ch_id = int(tp.channel)

            tp_start = int(tp.sample_start)
            tp_end = int(tp_start+tp.samples_over_threshold)

            win_start = tp_start-tp.samples_over_threshold*tp_win_extension
            win_end = tp_end+tp.samples_over_threshold*tp_win_extension

            # Initialise the axis range to the TP range
            ax.set_xlim(win_start, win_end)


            # Search for SimIDEs around the TP
            sel_ide = event_ides[
                (event_ides.channel == ch_id) &
                (event_ides.timestamp > (tp_start-ide_search_extension)) &
                (event_ides.timestamp < (tp_end+ide_search_extension))
                ]
            
            # plot them (if any)
            lns = ax.plot(sel_ide.timestamp, sel_ide.numelectrons, label='n$_{el}$', color=ide_color)
            ax.fill_between(sel_ide.timestamp, 0, sel_ide.numelectrons, color=ide_color_fill)
            ax.axhline(y=0, color='black', linewidth=1)
            
            # if any SimIDE is found, adjust the range
            if len(sel_ide) > 0:
                xmin, xmax = ax.get_xlim()
                ax.set_xlim(min(xmin, sel_ide.timestamp.min()), max(xmax, sel_ide.timestamp.max()))

            ax.set_xlabel("sample")
            ax.set_ylabel("n electrons$")

            if waves is None:
                print(f"[yellow]No waveforms found for event '{self.ev_num}[/yellow]'")
            else:
                from mpl_axes_aligner import shift
                shift.yaxis(ax, 0, 0.6, True)

                ## FIXME: using the position as sample_id
                wf = waves.reset_index()[ch_id]
                wf_mean = wf.mean()

                xmin, xmax = ax.get_xlim()
                ax_2 = ax.twinx()

                wf_zoom = wf.iloc[int(xmin):int(xmax)]
                lns += ax_2.plot(wf_zoom.index, wf_zoom.values, label="adcs", color=wave_color)

                tp_thres = wf_mean+tp_thresholds[tp_plane]
                ax_2.fill_between(wf_zoom.index, wf_zoom.where(wf_zoom < tp_thres, tp_thres).values, wf_zoom.values, color=thres_fill_color, alpha=0.3)

                ax_2.axhline(y=wf_mean, color='black', linewidth=1)
                ax_2.axhline(y=wf_mean+tp_thresholds[tp_plane], color=thres_color, linewidth=1)

                shift.yaxis(ax_2, wf_mean, 0.2, True)
                ax_2.set_ylabel("adcs")

            ymin, ymax = ax.get_ylim()
            print("ylim", ymin, ymax)

            rect_tp = patches.Rectangle((tp_start, ymin), tp.samples_over_threshold, ymax-ymin,
                                    linewidth=2, edgecolor=tp_color, facecolor=tp_color, alpha=0.2)
            ax.add_patch(rect_tp)
            rect_match = patches.Rectangle((tp_start+offsets[tp_plane], ymin), tp.samples_over_threshold, ymax-ymin,
                                    linewidth=2, edgecolor=match_color, fill=False, linestyle='-.', alpha=0.5)
            ax.add_patch(rect_match)

            ax.axvline(x=tp_start+tp.samples_to_peak, color='red', linewidth=1)


            ax.legend(lns+[rect_tp, rect_match], [l.get_label() for l in lns]+['TP', 'match win'], loc=0)

            ax.set_title(f"ch {int(tp['channel'])}, plane {int(tp.readout_view)} ")

        fig.tight_layout()


    def draw_eff_by_plane(self, bins=None, figsize=(12, 4)):
        ws=self.ws

        n_rops = 3
        fig, axes = plt.subplots(1,n_rops, figsize=figsize)
        for rop_id in range(n_rops):
            ax = axes[rop_id]
            tot_nel_df = ws.event_summary[['event_uid', f'tot_numelectrons_rop{rop_id}']].set_index('event_uid')
            bt_nel_df = pd.DataFrame(ws.tps.query(f'bt_is_signal == 1 & readout_plane_id == {rop_id}').groupby('event_uid').bt_numelectrons.sum())
            eff_df = tot_nel_df.merge(bt_nel_df, how='inner', on='event_uid').fillna(0)
            eff_df['ratio'] = eff_df.bt_numelectrons/eff_df[f'tot_numelectrons_rop{rop_id}']
            eff_df.ratio.hist(ax=ax, bins=bins)

            ax.set_xlabel(r"$Eff_{N_{el}}$")

        fig.suptitle("Backtracking Efficiency by plane")
        fig.tight_layout()
        return fig
    

    def plot_angular_correlations(self, det_type:str='vd', figsize=(14, 10)):
        """Plot correlations between particle angles in the wires/strips reference frames
           and backtracker ionisation charge matching efficiency.

        Args:
            figsize (tuple, optional): _description_. Defaults to (10, 10).

        Returns:
            _type_: _description_
        """

        n_rops = 3

        from tpvalidator.tpc_angles import calculate_angles

        # TODO: make it a ws member
        def mixnmatch(ws, **kwargs):
            keys=['event', 'run', 'subrun']

            dfs = []
            for k,v in kwargs.items():
                df = getattr(ws, k)
                dfs.append(df[keys+v].copy())

            from functools import reduce
            merged = reduce(lambda left, right: pd.merge(left, right, on=keys), dfs)

            return merged


        ws=self.ws

        # Sanity checks
        num_mcpart = ws.mctruths.groupby(['event', 'run', 'subrun']).size().unique()
        if len(num_mcpart) != 1 and num_mcpart[0] != 1:
            raise RuntimeError("The angular correlation plots required a particle gun data sample (1 particle/event)")

        # Calculate detected num electrons per rop
        det_numide_df = pd.DataFrame(ws.tps.query(f'bt_is_signal == 1').groupby(['run', 'subrun', 'event', 'readout_plane_id']).bt_numelectrons.sum()).unstack('readout_plane_id')
        det_numide_df.columns = [f"detected_num_electrons_rop{cat}" for col, cat in det_numide_df.columns]
        det_numide_df.reset_index()


        angles_df = mixnmatch(ws, mctruths=[ 'px','py', 'pz', 'p'], event_summary=[f'tot_numelectrons_rop{rop_id}' for rop_id in range(n_rops)])
        angles_df = angles_df.merge(det_numide_df, on=['event', 'run', 'subrun'])
        
        theta_y, theta_y_U, theta_y_V, theta_xz, theta_xz_U, theta_xz_V = calculate_angles(angles_df.px, angles_df.py, angles_df.pz, angles_df.p, det_type)

        angles_df['theta_y'] = theta_y
        angles_df['theta_yU'] = theta_y_U
        angles_df['theta_yV'] = theta_y_V
        angles_df['theta_xz'] = theta_xz
        angles_df['theta_xzU'] = theta_xz_U
        angles_df['theta_xzV'] = theta_xz_V


        for i in range(n_rops):
            angles_df[f'num_electrons_deteff_rop{i}'] = angles_df[f'detected_num_electrons_rop{i}']/angles_df[f'tot_numelectrons_rop{i}']


        # Style
        m='.'

        fig, ax = plt.subplots(3,3, figsize=figsize)


        # First row
        cmap='viridis'
        sU = ax[0][0].scatter(angles_df.theta_yU, angles_df.theta_xzU, c=angles_df.num_electrons_deteff_rop0, marker=m, cmap=cmap)
        sV = ax[0][1].scatter(angles_df.theta_yV, angles_df.theta_xzV, c=angles_df.num_electrons_deteff_rop1, marker=m, cmap=cmap)
        sX = ax[0][2].scatter(angles_df.theta_y, angles_df.theta_xz, c=angles_df.num_electrons_deteff_rop2, marker=m, cmap=cmap)
    
        ax[0][0].set_title(r"$\theta_y^U$ vs $\theta_{xz}^U$")
        ax[0][1].set_title(r"$\theta_y^V$ vs $\theta_{xz}^V$")
        ax[0][2].set_title(r"$\theta_y$ vs $\theta_{xz}$")

        ax[0][0].set_xlabel(r"$\theta_y^U$")
        ax[0][1].set_xlabel(r"$\theta_y^V$")
        ax[0][2].set_xlabel(r"$\theta_y$")

        ax[0][0].set_ylabel(r"$\theta_{xz}^U$")
        ax[0][1].set_ylabel(r"$\theta_{xz}^V$")
        ax[0][2].set_ylabel(r"$\theta_{xz}$")

        plt.colorbar(sU, label = r"$Eff_{N_{el}}^U$")
        plt.colorbar(sV, label = r"$Eff_{N_{el}}^V$")
        plt.colorbar(sX, label = r"$Eff_{N_{el}}^X$")

        # Second row
        cmap='plasma_r'
        sU = ax[1][0].scatter(angles_df.theta_yU, angles_df.num_electrons_deteff_rop0, c=angles_df.theta_xzU, marker=m, cmap=cmap)
        sV = ax[1][1].scatter(angles_df.theta_yV, angles_df.num_electrons_deteff_rop1, c=angles_df.theta_xzV, marker=m, cmap=cmap)
        sX = ax[1][2].scatter(angles_df.theta_y,  angles_df.num_electrons_deteff_rop2, c=angles_df.theta_xz , marker=m, cmap=cmap)

        plt.colorbar(sU, label = r"$\theta_{xz}^U$")
        plt.colorbar(sV, label = r"$\theta_{xz}^V$")
        plt.colorbar(sX, label = r"$\theta_{xz}$")

        ax[1][0].set_xlabel(r"$\theta_y^U$")
        ax[1][1].set_xlabel(r"$\theta_y^V$")
        ax[1][2].set_xlabel(r"$\theta_y$")

        ax[1][0].set_ylabel(r"$Eff_{N_{el}}^U$")
        ax[1][1].set_ylabel(r"$Eff_{N_{el}}^V$")
        ax[1][2].set_ylabel(r"$Eff_{N_{el}}$")


        # Third row
        cmap='magma_r'
        sU = ax[2][0].scatter(angles_df.theta_xzU, angles_df.num_electrons_deteff_rop0, c=angles_df.theta_yU,  marker=m, cmap=cmap)
        sV = ax[2][1].scatter(angles_df.theta_xzV, angles_df.num_electrons_deteff_rop1, c=angles_df.theta_yV,  marker=m, cmap=cmap)
        sX = ax[2][2].scatter(angles_df.theta_xz , angles_df.num_electrons_deteff_rop2, c=angles_df.theta_y,   marker=m, cmap=cmap)

        plt.colorbar(sU, label = r"$\theta_{xz}^U$")
        plt.colorbar(sV, label = r"$\theta_{xz}^V$")
        plt.colorbar(sX, label = r"$\theta_{xz}$")

        ax[2][0].set_xlabel(r"$\theta_{xz}^U$")
        ax[2][1].set_xlabel(r"$\theta_{xz}^V$")
        ax[2][2].set_xlabel(r"$\theta_{xz}$")

        ax[2][0].set_ylabel(r"$Eff_{N_{el}}^U$")
        ax[2][1].set_ylabel(r"$Eff_{N_{el}}^V$")
        ax[2][2].set_ylabel(r"$Eff_{N_{el}}$")

        return fig