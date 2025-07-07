import rich
import matplotlib

from .basic import BasicTPData
from .utilities import subplot_autogrid
import matplotlib.pyplot as plt
import matplotlib.patches as patches



class BackTrackerPlotter:

    def __init__(self, tp_data: BasicTPData, ev_num: int):

        if tp_data.ides is None:
            raise RuntimeError(f"No IDE data available in '{tp_data.data_path}'")

        self.ev_num = ev_num
        self.data = tp_data

        self.inspect_tps = self.data.tps[(self.data.tps.event == ev_num)  & (self.data.tps.TP_signal == True)]

        self.offsets = [self.data.tp_backtracker_offset(p) for p in range(3)]
        self.tp_thresholds = [self.data.tp_threshold(p) for p in range(3)]
        self.waveforms = self.data.waveforms.get(ev_num, None)
        if self.waveforms is None:
            rich.print(f"[yellow]Warning: no waveform data found in {self.data.data_path} for event {ev_num}[/yellow]")


    def plot_tps_vs_ides( self, tp_ids: list, layout:str = 'grid', figsize=(10, 10)):

        # tp_data = self.data
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
        wave_color = colors[2]
        thres_color = colors[6]
        tp_color = colors[7]
        match_color = colors[8]


        selected_tps = inspect_tps.iloc[:0].copy()

        for i,tp_idx in enumerate(tp_ids):

            ax = axes[i]

            tp = inspect_tps.iloc[tp_idx]
            tp_plane = int(tp.TP_plane)
            selected_tps.loc[len(selected_tps)] = tp

            ch_id = int(tp.TP_channel)

            tp_start, tp_end = int(tp.TP_startT), int(tp.TP_startT+tp.TP_TOT)
            # print(f"TP start-end: {tp_start}-{tp_end}")
            # print(f"TP peak: {tp.TP_peakADC}")

            # Plot IDE data
            q_ch = self.data.ides[(self.data.ides['event'] == self.ev_num) & (self.data.ides.channel == ch_id)]
            lns = ax.plot(q_ch.time, q_ch.nElectrons, label='n$_{el}$', color=ide_color)

            ax.axhline(y=0, color='black', linewidth=1)

            xmin, xmax = ax.get_xlim()
            ax.set_xlim(min(xmin, tp_start), max(xmax, tp_end))

            if waves is None:
                print(f"No waveforms found for event '{self.ev_num}'")
            else:
                from mpl_axes_aligner import shift
                shift.yaxis(ax, 0, 0.6, True)

                ## FIXME: using the position as sample_id
                wf = waves.reset_index()[ch_id]

                xmin, xmax = ax.get_xlim()
                ax_2 = ax.twinx()

                wf_zoom = wf.iloc[int(xmin):int(xmax)]
                lns += ax_2.plot(wf_zoom.index, wf_zoom.values, label="adcs", color=wave_color)

                wf_mean = wf.mean()
                ax_2.axhline(y=wf_mean, color='black', linewidth=1)
                ax_2.axhline(y=wf_mean+tp_thresholds[tp_plane], color=thres_color, linewidth=1)

                shift.yaxis(ax_2, wf_mean, 0.2, True)


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


        display(selected_tps)
        #fig.tight_layout()
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
    
    def plot_tps_vs_ides_by_plane(self, plane_id: int, tp_pos: list, figsize=(10, 10)):
        # FIXME : sort out tp position vs index
        inspect_tps = self.inspect_tps

        inspect_tps_plane = inspect_tps[inspect_tps.TP_plane == plane_id]

        tp_pos_list = []

        for tp_pos in tp_pos:
            tp_idx = inspect_tps_plane.index[tp_pos]
            tp_pos_list.append(inspect_tps.index.get_loc(tp_idx))

        return self.plot_tps_vs_ides(tp_pos_list, figsize=figsize)
