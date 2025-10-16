import matplotlib
import pandas as pd
from .workspace import TriggerPrimitivesWorkspace
from .utilities import subplot_autogrid
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Optional, Union, Sequence, Dict, List

from rich import print


class BackTrackerPlotter:

    def __init__(self, tpws: TriggerPrimitivesWorkspace, ev_num: int):

        if tpws.ides is None:
            raise RuntimeError(f"No IDE data available in '{tpws._tp_path}'")

        self.ev_num = ev_num
        self.ws = tpws


        self.inspect_tps = self.ws.tps[(self.ws.tps.event == ev_num)  & (self.ws.tps.TP_signal == True)]

        self.offsets = [self.ws.tp_backtracker_offset(p) for p in range(3)]
        self.tp_thresholds = [self.ws.tp_threshold(p) for p in range(3)]


        self.waveforms = self.ws.get_waveforms(ev_num)
        if self.waveforms is None:
            print(f"[yellow]Warning: no waveform data found in {self.ws._rawdigits_path} for event {ev_num}[/yellow]")


    def old_plot_tps_vs_ides( self, tp_ids: list, layout:str = 'grid', figsize=(10, 10)):

        # tp_data = self.ws
        inspect_tps = self.inspect_tps
        waves = self.waveforms
        
        offsets = self.offsets
        tp_thresholds = self.tp_thresholds

        match layout:
            case 'lin':
                fig, axes = plt.subplots(1,len(tp_ids), figsize=figsize)
            case 'grid':
                fig, axes = subplot_autogrid(len(tp_ids), figsize=figsize)
            case _:
                raise ValueError(f"Layout value '{layout} unknown" )

        # Color selection
        colors = matplotlib.colormaps['tab20'].colors
        ide_color = colors[0]
        ide_color_fill = colors[1]
        wave_color = colors[2]
        thres_color = colors[6]
        thres_fill_color = colors[7]
        tp_color = colors[7]
        match_color = colors[8]


        selected_tps = inspect_tps.iloc[:0].copy()

        # Focus on ides from this event only 
        event_ides = self.ws.ides[self.ws.ides.event == self.ev_num]


        for i,tp_idx in enumerate(tp_ids):

            ax = axes[i]

            tp = inspect_tps.iloc[tp_idx]
            tp_plane = int(tp.TP_plane)
            selected_tps.loc[len(selected_tps)] = tp

            ch_id = int(tp.TP_channel)

            tp_start, tp_end = int(tp.TP_startT), int(tp.TP_startT+tp.TP_TOT)
            ide_win_extension = 500

            # print(f"TP start-end: {tp_start}-{tp_end}")
            # print(f"TP peak: {tp.TP_peakADC}")
            # print(f"TP channel: {tp.TP_channel}")

            # Plot IDE data
            q_ch = event_ides[
                (event_ides.channel == ch_id) & 
                (event_ides.time > (tp_start-ide_win_extension)) &
                (event_ides.time < (tp_end+ide_win_extension))
                ]
            
            lns = ax.plot(q_ch.time, q_ch.n_electrons, label='n$_{el}$', color=ide_color)
            ax.fill_between(q_ch.time, 0, q_ch.n_electrons, color=ide_color_fill)
            ax.axhline(y=0, color='black', linewidth=1)

            xmin, xmax = ax.get_xlim()
            ax.set_xlim(min(xmin, tp_start)-tp.TP_TOT, max(xmax, tp_end)+tp.TP_TOT)
            xmin, xmax = ax.get_xlim()

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
                wf_mean_2 = (wf[0:32].mean()+wf[-32:-1].mean())/2
                wf_mean_2 = wf_zoom[0:10].mean()
                wf_mean_2 = wf_mean
                # lns += ax_2.plot(wf_zoom.index, (((wf_zoom-wf_mean_2).cumsum()/6)+wf_mean_2).values, label="adcs", color="green")


                tp_thres = wf_mean+tp_thresholds[tp_plane];
                ax_2.fill_between(wf_zoom.index, wf_zoom.where(wf_zoom < tp_thres, tp_thres).values, wf_zoom.values, color=thres_fill_color, alpha=0.3)

                ax_2.axhline(y=wf_mean, color='black', linewidth=1)
                ax_2.axhline(y=wf_mean+tp_thresholds[tp_plane], color=thres_color, linewidth=1)

                shift.yaxis(ax_2, wf_mean, 0.2, True)
                ax_2.set_ylabel("adcs")


            # Fixme use the sample id
            # if ((waves.event == ev_num).any()):

            #     from mpl_axes_aligner import shift
            #     shift.yaxis(ax, 0, 0.6, True)
            #     wf = waves[['sample_id', ch_id]]
    
            #     xmin, xmax = ax.get_xlim()
            #     ax_2 = ax.twinx()

            #     # wf_zoom = wf.iloc[int(xmin):int(xmax)]
            #     wf_zoom = wf[(wf.sample_id >=  int(xmin)) | (wf.sample_id <= int(xmax))]
            #     lns += ax_2.plot(wf_zoom.sample_id, wf_zoom.values, label="adcs", color=wave_color)

            #     ch_mean  = wf[ch_id].mean()
            #     ax_2.axhline(y=ch_mean, color='black', linewidth=1)
            #     ax_2.axhline(y=ch_mean+thres[p], color=thres_color, linewidth=1)

            #     shift.yaxis(ax_2, ch_mean, 0.2, True)

            ymin, ymax = ax.get_ylim()

            rect_tp = patches.Rectangle((tp.TP_startT, ymin), tp.TP_TOT, ymax-ymin,
                                    linewidth=2, edgecolor=tp_color, facecolor=tp_color, alpha=0.2)
            ax.add_patch(rect_tp)
            rect_match = patches.Rectangle((tp.TP_startT+offsets[tp_plane], ymin), tp.TP_TOT, ymax-ymin,
                                    linewidth=2, edgecolor=match_color, fill=False, linestyle='-.', alpha=0.5)
            ax.add_patch(rect_match)

            ax.legend(lns+[rect_tp, rect_match], [l.get_label() for l in lns]+['TP', 'match win'], loc=0)

            ax.set_title(f"ch {int(tp.TP_channel)}, plane {int(tp.TP_plane)} ")
        
        fig.suptitle(f"Event {self.ev_num}", fontsize=16)


        self.selected_tps = selected_tps
        #fig.tight_layout()
        # return selected_tps
        return fig
    
    def plot_tps_vs_ides( self, tps: Union[List[int],pd.DataFrame], layout:str = 'grid', figsize=(10, 10)):

        ## Some parameters
        ide_win_extension = 500

        # tp_data = self.ws
        inspect_tps = self.inspect_tps
        waves = self.waveforms
        
        offsets = self.offsets
        tp_thresholds = self.tp_thresholds

        match layout:
            case 'lin':
                fig, axes = plt.subplots(1,len(tps), figsize=figsize)
            case 'grid':
                fig, axes = subplot_autogrid(len(tps), figsize=figsize)
            case _:
                raise ValueError(f"Layout value '{layout} unknown" )

        # Color selection
        colors = matplotlib.colormaps['tab20'].colors
        ide_color = colors[0]
        ide_color_fill = colors[1]
        wave_color = colors[2]
        thres_color = colors[6]
        thres_fill_color = colors[7]
        tp_color = colors[7]
        match_color = colors[8]

        import pandas as pd
        # Make a small local copy
        if isinstance(tps, list):
            selected_tps = inspect_tps.iloc[tps].copy()
        elif isinstance(tps, pd.DataFrame):
            selected_tps = tps.copy()
        else:
            raise TypeError(f"Argument 'tps' of unsupported type {type(tps)}" )


        # Focus on ides from this event only 
        event_ides = self.ws.ides[self.ws.ides.event == self.ev_num]


        for i, (index, tp) in enumerate(selected_tps.iterrows()):

            ax = axes[i]

            # tp = inspect_tps.iloc[tp_idx]
            tp_plane = int(tp.TP_plane)
            # selected_tps.loc[len(selected_tps)] = tp

            ch_id = int(tp.TP_channel)

            tp_start, tp_end = int(tp.TP_startT), int(tp.TP_startT+tp.TP_TOT)

            # print(f"TP start-end: {tp_start}-{tp_end}")
            # print(f"TP peak: {tp.TP_peakADC}")
            # print(f"TP channel: {tp.TP_channel}")

            # Plot IDE data
            q_ch = event_ides[
                (event_ides.channel == ch_id) & 
                (event_ides.time > (tp_start-ide_win_extension)) &
                (event_ides.time < (tp_end+ide_win_extension))
                ]
            
            lns = ax.plot(q_ch.time, q_ch.n_electrons, label='n$_{el}$', color=ide_color)
            ax.fill_between(q_ch.time, 0, q_ch.n_electrons, color=ide_color_fill)
            ax.axhline(y=0, color='black', linewidth=1)

            xmin, xmax = ax.get_xlim()
            ax.set_xlim(min(xmin, tp_start)-tp.TP_TOT, max(xmax, tp_end)+tp.TP_TOT)
            xmin, xmax = ax.get_xlim()

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
                wf_mean_2 = (wf[0:32].mean()+wf[-32:-1].mean())/2
                wf_mean_2 = wf_zoom[0:10].mean()
                wf_mean_2 = wf_mean
                # lns += ax_2.plot(wf_zoom.index, (((wf_zoom-wf_mean_2).cumsum()/6)+wf_mean_2).values, label="adcs", color="green")


                tp_thres = wf_mean+tp_thresholds[tp_plane];
                ax_2.fill_between(wf_zoom.index, wf_zoom.where(wf_zoom < tp_thres, tp_thres).values, wf_zoom.values, color=thres_fill_color, alpha=0.3)

                ax_2.axhline(y=wf_mean, color='black', linewidth=1)
                ax_2.axhline(y=wf_mean+tp_thresholds[tp_plane], color=thres_color, linewidth=1)

                shift.yaxis(ax_2, wf_mean, 0.2, True)
                ax_2.set_ylabel("adcs")


            # Fixme use the sample id
            # if ((waves.event == ev_num).any()):

            #     from mpl_axes_aligner import shift
            #     shift.yaxis(ax, 0, 0.6, True)
            #     wf = waves[['sample_id', ch_id]]
    
            #     xmin, xmax = ax.get_xlim()
            #     ax_2 = ax.twinx()

            #     # wf_zoom = wf.iloc[int(xmin):int(xmax)]
            #     wf_zoom = wf[(wf.sample_id >=  int(xmin)) | (wf.sample_id <= int(xmax))]
            #     lns += ax_2.plot(wf_zoom.sample_id, wf_zoom.values, label="adcs", color=wave_color)

            #     ch_mean  = wf[ch_id].mean()
            #     ax_2.axhline(y=ch_mean, color='black', linewidth=1)
            #     ax_2.axhline(y=ch_mean+thres[p], color=thres_color, linewidth=1)

            #     shift.yaxis(ax_2, ch_mean, 0.2, True)

            ymin, ymax = ax.get_ylim()

            rect_tp = patches.Rectangle((tp.TP_startT, ymin), tp.TP_TOT, ymax-ymin,
                                    linewidth=2, edgecolor=tp_color, facecolor=tp_color, alpha=0.2)
            ax.add_patch(rect_tp)
            rect_match = patches.Rectangle((tp.TP_startT+offsets[tp_plane], ymin), tp.TP_TOT, ymax-ymin,
                                    linewidth=2, edgecolor=match_color, fill=False, linestyle='-.', alpha=0.5)
            ax.add_patch(rect_match)

            ax.legend(lns+[rect_tp, rect_match], [l.get_label() for l in lns]+['TP', 'match win'], loc=0)

            ax.set_title(f"ch {int(tp.TP_channel)}, plane {int(tp.TP_plane)} ")
        
        fig.suptitle(f"Event {self.ev_num}", fontsize=16)


        self.selected_tps = selected_tps
        fig.tight_layout()
        # return selected_tps
        return fig
    
    def plot_tps_vs_ides_one_per_plane(self, tp_pos_by_plane: list, figsize=(10, 10)):
        inspect_tps = self.inspect_tps

        tp_pos_list = []
        for p in range(3):
            inspect_tps_plane = inspect_tps[inspect_tps.TP_plane == p]
            if len(inspect_tps_plane) == 0:
                continue

            tp_idx = inspect_tps_plane.index[tp_pos_by_plane[p]]

            tp_pos_list.append(inspect_tps.index.get_loc(tp_idx))


        return self.plot_tps_vs_ides(tp_pos_list, layout='lin', figsize=figsize)
    

    def plot_tps_vs_ides_by_plane(self, plane_id: int, tp_idx_in_plane: list, figsize=(10, 10)):

        selected_tps = self.inspect_tps[self.inspect_tps.TP_plane == plane_id].iloc[[tp_idx_in_plane]]

        return self.plot_tps_vs_ides(selected_tps, figsize=figsize)
    

    def plot_angular_correlations(self, figsize=(10, 10)):
        """Plot correlations between particle angles in the wires/strips reference frames
           and backtracker ionisation charge matching efficiency.

        Args:
            figsize (tuple, optional): _description_. Defaults to (10, 10).

        Returns:
            _type_: _description_
        """
        df_mc_copy = self.ws.mc.copy()
        df_mc_copy['rQ_U'] = df_mc_copy.detQ_U/df_mc_copy.totQ_U
        df_mc_copy['rQ_V'] = df_mc_copy.detQ_V/df_mc_copy.totQ_V
        df_mc_copy['rQ_X'] = df_mc_copy.detQ_X/df_mc_copy.totQ_X


        y = df_mc_copy.join(self.ws.angles.set_index('event'), on='event')

        m = '.'
        figsize=(12,10)

        fig, ax = plt.subplots(3,3, figsize=figsize)

        # ax[0][0].set_title(r"$\theta_y^U$ vs $\theta_{xz}^V$")
        # ax[0][1].set_title(r"$\theta_y^V$ vs $\theta_{xz}^V$")
        # ax[0][2].set_title(r"$\theta_y$ vs $\theta_{xz}$")

        ax[0][0].set_xlabel(r"$\theta_y^U$")
        ax[0][1].set_xlabel(r"$\theta_y^V$")
        ax[0][2].set_xlabel(r"$\theta_y$")

        ax[0][0].set_ylabel(r"$\theta_{xz}^U$")
        ax[0][1].set_ylabel(r"$\theta_{xz}^V$")
        ax[0][2].set_ylabel(r"$\theta_{xz}$")


        sU = ax[0][0].scatter(y.theta_yU, y.theta_xzU, c=y.rQ_U, marker=m)
        sV = ax[0][1].scatter(y.theta_yV, y.theta_xzV, c=y.rQ_V, marker=m)
        sX = ax[0][2].scatter(y.theta_y, y.theta_xz, c=y.rQ_X, marker=m)

        plt.colorbar(sU, label = r"$rQ^U$")
        plt.colorbar(sV, label = r"$rQ^V$")
        plt.colorbar(sX, label = r"$rQ$")

        # ax[1][0].set_title(r"$\theta_y^U$ vs $rQ^U$")
        # ax[1][1].set_title(r"$\theta_y^V$ vs $rQ^V$")
        # ax[1][2].set_title(r"$\theta_y$ vs $rQ_X$")

        ax[1][0].set_xlabel(r"$\theta_y^U$")
        ax[1][1].set_xlabel(r"$\theta_y^V$")
        ax[1][2].set_xlabel(r"$\theta_y$")

        ax[1][0].set_ylabel(r"$rQ^U$")
        ax[1][1].set_ylabel(r"$rQ^V$")
        ax[1][2].set_ylabel(r"$rQ$")



        sU = ax[1][0].scatter( y.theta_yU, y.rQ_U, c=y.theta_xzU, marker=m, cmap='plasma_r')
        sV = ax[1][1].scatter( y.theta_yV, y.rQ_V, c=y.theta_xzV, marker=m, cmap='plasma_r')
        sX = ax[1][2].scatter( y.theta_y,  y.rQ_X, c=y.theta_xz, marker=m, cmap='plasma_r')

        plt.colorbar(sU)
        plt.colorbar(sV)
        plt.colorbar(sX)

        # ax[2][0].set_title(r"$\theta_{xz}^U$ vs $rQ^U$")
        # ax[2][1].set_title(r"$\theta_{xz}^X$ vs $rQ^V$")
        # ax[2][2].set_title(r"$\theta_{xz}$ vs $rQ^X$")

        ax[2][0].set_xlabel(r"$\theta_{xz}^U$")
        ax[2][1].set_xlabel(r"$\theta_{xz}^V$")
        ax[2][2].set_xlabel(r"$\theta_{xz}$")

        ax[2][0].set_ylabel(r"$rQ^U$")
        ax[2][1].set_ylabel(r"$rQ^V$")
        ax[2][2].set_ylabel(r"$rQ$")


        sU = ax[2][0].scatter(y.theta_xzU, y.rQ_U, c=y.theta_yU, marker=m, cmap='magma_r')
        sV = ax[2][1].scatter(y.theta_xzV, y.rQ_V, c=y.theta_yV, marker=m, cmap='magma_r')
        sX = ax[2][2].scatter(y.theta_xz, y.rQ_X, c=y.theta_y, marker=m, cmap='magma_r')

        plt.colorbar(sU)
        plt.colorbar(sV)
        plt.colorbar(sX)
        
        return fig

    def plot_alternative_angular_correlations(self, figsize=(10, 10)):
        """Plot correlations between particle angles in the wires/strips reference frames
           and backtracker ionisation charge matching efficiency.

        Args:
            figsize (tuple, optional): _description_. Defaults to (10, 10).

        Returns:
            _type_: _description_
        """
        df_mc_copy = self.ws.mc.copy()
        df_mc_copy['rQ_U'] = df_mc_copy.detQ_U/df_mc_copy.totQ_U
        df_mc_copy['rQ_V'] = df_mc_copy.detQ_V/df_mc_copy.totQ_V
        df_mc_copy['rQ_X'] = df_mc_copy.detQ_X/df_mc_copy.totQ_X
        df_corr = df_mc_copy.join(self.ws.angles.set_index('event'), on='event')

        m='.'
        fig, ax = plt.subplots(3,3, figsize=figsize)
        sU = ax[0][0].scatter(x=df_corr.phi_drift_u, y=df_corr.theta_drift, c=df_corr.rQ_U, marker=m)
        sV = ax[0][1].scatter(x=df_corr.phi_drift_v, y=df_corr.theta_drift, c=df_corr.rQ_V, marker=m)
        sX = ax[0][2].scatter(x=df_corr.phi_drift,   y=df_corr.theta_drift, c=df_corr.rQ_X, marker=m)

        ax[0][0].set_xlabel(r"$\varphi_{drift}^U$")
        ax[0][1].set_xlabel(r"$\varphi_{drift}^V$")
        ax[0][2].set_xlabel(r"$\varphi_{drift}$")

        ax[0][0].set_ylabel(r"$\theta_{drift}^U$")
        ax[0][1].set_ylabel(r"$\theta_{drift}^V$")
        ax[0][2].set_ylabel(r"$\theta_{drift}$")

        plt.colorbar(sU, label = r"$q_{det}/Q_{tot}$ U")
        plt.colorbar(sV, label = r"$q_{det}/Q_{tot}$ V")
        plt.colorbar(sX, label = r"$q_{det}/Q_{tot}$ X")

        sU = ax[1][0].scatter(x=df_corr.phi_drift_u, c=df_corr.theta_drift, y=df_corr.rQ_U, marker=m, cmap='plasma_r')
        sV = ax[1][1].scatter(x=df_corr.phi_drift_v, c=df_corr.theta_drift, y=df_corr.rQ_V, marker=m, cmap='plasma_r')
        sX = ax[1][2].scatter(x=df_corr.phi_drift,   c=df_corr.theta_drift, y=df_corr.rQ_X, marker=m, cmap='plasma_r')

        ax[1][0].set_xlabel(r"$\varphi_{drift}^U$")
        ax[1][1].set_xlabel(r"$\varphi_{drift}^V$")
        ax[1][2].set_xlabel(r"$\varphi_{drift}$")

        ax[1][0].set_ylabel(r"$rQ^U$")
        ax[1][1].set_ylabel(r"$rQ^V$")
        ax[1][2].set_ylabel(r"$rQ$")

        plt.colorbar(sU, label=r"$\theta_{drift}$")
        plt.colorbar(sV, label=r"$\theta_{drift}$")
        plt.colorbar(sX, label=r"$\theta_{drift}$")


        sU = ax[2][0].scatter(c=df_corr.phi_drift_u, x=df_corr.theta_drift, y=df_corr.rQ_U, marker=m, cmap='magma_r')
        sV = ax[2][1].scatter(c=df_corr.phi_drift_v, x=df_corr.theta_drift, y=df_corr.rQ_V, marker=m, cmap='magma_r')
        sX = ax[2][2].scatter(c=df_corr.phi_drift,   x=df_corr.theta_drift, y=df_corr.rQ_X, marker=m, cmap='magma_r')

        ax[2][0].set_xlabel(r"$\theta_{drift}$")
        ax[2][1].set_xlabel(r"$\theta_{drift}$")
        ax[2][2].set_xlabel(r"$\theta_{drift}$")

        ax[2][0].set_ylabel(r"$rQ^U$")
        ax[2][1].set_ylabel(r"$rQ^V$")
        ax[2][2].set_ylabel(r"$rQ$")

        plt.colorbar(sU, label=r"$\varphi_{drift}^U$")
        plt.colorbar(sV, label=r"$\varphi_{drift}^V$")
        plt.colorbar(sX, label=r"$\varphi_{drift}$")

        return fig