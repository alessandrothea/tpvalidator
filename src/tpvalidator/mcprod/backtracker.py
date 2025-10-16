import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .workspace import TriggerPrimitivesWorkspace
from typing import Tuple, Optional, Union, Sequence, Dict, List
from ..utilities import subplot_autogrid
from rich import print

class BackTrackerPlotter:
    def __init__(self, ws: TriggerPrimitivesWorkspace, ev_num: int):

        if ws.ides is None:
            raise RuntimeError(f"No IDE data available in '{ws._data_path}'")
        
        self.ev_num = ev_num
        self.ws = ws

        # self.inspect_tps = self.ws.tps[(self.ws.tps.Event == ev_num)  & (self.ws.tps.bt_numelectrons > 0)]
        self.inspect_tps = self.ws.tps[(self.ws.tps.event == ev_num)]
        # Focus on ides from this event only 
        self.event_ides = self.ws.ides[self.ws.ides.event == ev_num]


        self.tpg_info = ws.info['tpg'][ws.tp_maker_name]
        self.tp_thresholds = [self.tpg_info[f'threshold_tpg_plane{i}'] for i in range(3)]
        self.bt_offsets = [ ws.info['backtracker'][self.tpg_info['tool']][f'offset_{v}'] for v in ('U', 'V', 'X') ]


        self.waveforms = self.ws.get_waveforms(ev_num)
        if self.waveforms is None:
            print(f"[yellow]Warning: no waveform data found in {self.ws._rawdigits_path} for event {ev_num}[/yellow]")


    def plot_tps_vs_ides( self, tps: Union[List[int], pd.DataFrame], layout:str = 'grid', figsize=(10, 10)):

        # IDEs search range around the TP
        ide_win_extension = 250

        inspect_tps = self.inspect_tps
        event_ides = self.event_ides
        waves = self.waveforms
        tp_thresholds = self.tp_thresholds
        offsets = self.bt_offsets

        match layout:
            case 'lin':
                fig, axes = plt.subplots(1,len(tps), figsize=figsize)
            case 'grid':
                fig, axes = subplot_autogrid(len(tps), figsize=figsize)
            case _:
                raise ValueError(f"Layout value '{layout} unknown" ) 

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
            raise TypeError(f"Argument 'tps' of unsupported type {type(tps)}" )


        for i, (index, tp) in enumerate(selected_tps.iterrows()):
            ax = axes[i]
            tp_plane = int(tp.readout_view)
            ch_id = int(tp.channel)

            tp_start = int(tp.time_start//32)
            tp_end = int(tp_start+tp.samples_over_threshold)

            # print(tp_start, tp_end)

            sel_ide = event_ides[
                (event_ides.channel == ch_id) & 
                (event_ides.timestamp > (tp_start-ide_win_extension)) &
                (event_ides.timestamp < (tp_end+ide_win_extension))
                ]
            
            lns = ax.plot(sel_ide.timestamp, sel_ide.numelectrons, label='n$_{el}$', color=ide_color)
            ax.fill_between(sel_ide.timestamp, 0, sel_ide.numelectrons, color=ide_color_fill)
            ax.axhline(y=0, color='black', linewidth=1)
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(min(xmin, tp_start)-tp.samples_over_threshold, max(xmax, tp_end)+tp.samples_over_threshold)
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
                # wf_mean_2 = (wf[0:32].mean()+wf[-32:-1].mean())/2
                # wf_mean_2 = wf_zoom[0:10].mean()
                # wf_mean_2 = wf_mean
                # lns += ax_2.plot(wf_zoom.index, (((wf_zoom-wf_mean_2).cumsum()/6)+wf_mean_2).values, label="adcs", color="green")


                tp_thres = wf_mean+tp_thresholds[tp_plane];
                ax_2.fill_between(wf_zoom.index, wf_zoom.where(wf_zoom < tp_thres, tp_thres).values, wf_zoom.values, color=thres_fill_color, alpha=0.3)

                ax_2.axhline(y=wf_mean, color='black', linewidth=1)
                ax_2.axhline(y=wf_mean+tp_thresholds[tp_plane], color=thres_color, linewidth=1)

                shift.yaxis(ax_2, wf_mean, 0.2, True)
                ax_2.set_ylabel("adcs")

            ymin, ymax = ax.get_ylim()
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