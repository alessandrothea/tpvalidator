"""Signal and Noise analyzer


"""

import uproot

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cmocean.cm as cmo


from ...histograms import uproot_hist_mean_std, calculate_natural_bins
from ..workspace import TriggerPrimitivesWorkspace
from ...utilities import subplot_autogrid, df_to_tp_rates



# TODO: review and move elsewhere?
def draw_signal_and_noise_adc_distros(tpws: TriggerPrimitivesWorkspace, signal_label='Signal', figsize=(12,5)):

    adc_hists = tpws.rawdigits_hists

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    view_range = 20
    stat_x = 0.95
    stat_y = 0.95
    v_lw=0.5

    thres_x = 0.925
    thres_y = 0.85

    # -- draw
    fig,axes= plt.subplots(1,3, figsize=figsize)

    ax = axes[0]
    adc_hists['ADCsPlaneU'].to_hist().plot(ax=ax)
    adc_hists['ADCsNoisePlaneU'].to_hist().plot(ax=ax)
    ax.set_xlabel('adc value')
    ax.set_ylabel('counts')

    mu, sigma = uproot_hist_mean_std(adc_hists['ADCsNoisePlaneU'])[0]
    ax.set_xlim(mu-view_range*sigma, mu+view_range*sigma)
    thrs_3s=mu+3*sigma
    thrs_5s=mu+5*sigma

    textstr = '\n'.join((
        f'$\\mu={mu:.2f}$',
        f'$\\sigma={sigma:.2f}$'))

    # place a text box in top center in axes coords
    ax.text(stat_x, stat_y, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    textstr = '\n'.join((
        f'$3\\sigma={int(3*sigma)}$',
        f'$5\\sigma={int(5*sigma)}$'
        ))
    ax.text(thres_x, thres_y, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right')

    ax.axvline(mu, color='b', ls='--', lw=v_lw)
    ax.axvline(thrs_3s, color='r', lw=v_lw)
    ax.axvline(thrs_5s, color='g', lw=v_lw)
    ax.set_title('Plane U [0] ')

    ax = axes[1]
    adc_hists['ADCsPlaneV'].to_hist().plot(ax=ax)
    adc_hists['ADCsNoisePlaneV'].to_hist().plot(ax=ax)
    ax.set_xlabel('adc value')
    ax.set_ylabel('counts')

    mu, sigma = uproot_hist_mean_std(adc_hists['ADCsNoisePlaneV'])[0]
    ax.set_xlim(mu-view_range*sigma, mu+view_range*sigma)
    thrs_3s=mu+3*sigma
    thrs_5s=mu+5*sigma

    textstr = '\n'.join((
        f'$\\mu={mu:.2f}$',
        f'$\\sigma={sigma:.2f}$'))

    # place a text box in top center in axes coords
    ax.text(stat_x, stat_y, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    textstr = '\n'.join((
        f'$3\\sigma={int(3*sigma)}$',
        f'$5\\sigma={int(5*sigma)}$'
        ))
    ax.text(thres_x, thres_y, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right')

    ax.axvline(mu, color='b', ls='--', lw=v_lw)
    ax.axvline(thrs_3s, color='r', lw=v_lw)
    ax.axvline(thrs_5s, color='g', lw=v_lw)
    ax.set_title('Plane V [1] ')


    ax = axes[2]
    adc_hists['ADCsPlaneX'].to_hist().plot(ax=ax)
    adc_hists['ADCsNoisePlaneX'].to_hist().plot(ax=ax)
    ax.set_xlabel('adc value')
    ax.set_ylabel('counts')

    mu, sigma = uproot_hist_mean_std(adc_hists['ADCsNoisePlaneX'])[0]
    ax.set_xlim(mu-view_range*sigma, mu+view_range*sigma)
    thrs_3s=mu+3*sigma
    thrs_5s=mu+5*sigma

    textstr = '\n'.join((
        f'$\\mu={mu:.2f}$',
        f'$\\sigma={sigma:.2f}$'))
    # place a text box in top center in axes coords
    ax.text(stat_x, stat_y, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    textstr = '\n'.join((
        f'$3\\sigma={int(3*sigma)}$',
        f'$5\\sigma={int(5*sigma)}$'
        ))
    ax.text(thres_x, thres_y, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right')

    ax.axvline(mu, color='b', ls='--', lw=v_lw)
    ax.axvline(thrs_3s, color='r', lw=v_lw)
    ax.axvline(thrs_5s, color='g', lw=v_lw)
    ax.set_title('Plane X [2] ')

    for ax in axes:
        ax.set_yscale("log")
        ax.legend([f"{signal_label}+noise", 'noise'])

    fig.tight_layout()
    return fig

class TPSignalNoiseSelector:
    def __init__(self, tps):

        self.all = tps

        self.all_view_0 = tps.query('readout_view == 0')
        self.all_view_1 = tps.query('readout_view == 1')
        self.all_view_2 = tps.query('readout_view == 2')

        self.sig_view_0 = self.all_view_0.query('bt_is_signal != 0')
        self.sig_view_1 = self.all_view_1.query('bt_is_signal != 0')
        self.sig_view_2 = self.all_view_2.query('bt_is_signal != 0')

        self.noise_view_0 = self.all_view_0.query('bt_is_signal == 0')
        self.noise_view_1 = self.all_view_1.query('bt_is_signal == 0')
        self.noise_view_2 = self.all_view_2.query('bt_is_signal == 0')
        
        self.all_by_view = {
            0 : self.all_view_0,
            1 : self.all_view_1,
            2 : self.all_view_2,
        }
        
        self.sig_by_view = {
            0 : self.sig_view_0,
            1 : self.sig_view_1,
            2 : self.sig_view_2,
        }

        self.noise_by_view = {
            0 : self.noise_view_0,
            1 : self.noise_view_1,
            2 : self.noise_view_2,
        }  


        # TODO: remove
        # legacy aliases
        self.p0 = self.all_view_0
        self.p1 = self.all_view_1
        self.p2 = self.all_view_2

        self.sig_p0 = self.sig_view_0
        self.sig_p1 = self.sig_view_1
        self.sig_p2 = self.sig_view_2

        self.noise_p0 = self.noise_view_0
        self.noise_p1 = self.noise_view_1
        self.noise_p2 = self.noise_view_2

    def __len__(self) -> int:
        return len(self.all)
    
    def query(self, query: str):
        return TPSignalNoiseSelector(self.all.query(query))
        

class TPSignalNoiseAnalyzer:

    def __init__(self, 
                 tp_selection: TPSignalNoiseSelector, 
                 signal_name: str ='Signal'
                 ):
        self.tps = tp_selection
        self.signal_name = signal_name

    def draw_tp_sig_origin_2d_dist(self, signal_label='Signal', figsize=(12,10)):
        fig,axes= plt.subplots(3,3, figsize=figsize)

        # XY row
        ax = axes[0][0]
        self.tps.sig_p0.plot.scatter(x='bt_primary_y', y='bt_primary_x', alpha=0.01, s=1, ax=ax)
        ax.set_title('XY origin - Plane U [0]')

        ax = axes[0][1]
        self.tps.sig_p1.plot.scatter(x='bt_primary_y', y='bt_primary_x', alpha=0.01, s=1, ax=ax)
        ax.set_title('XY origin - Plane V [1]')

        ax = axes[0][2]
        self.tps.sig_p2.plot.scatter(x='bt_primary_y', y='bt_primary_x', alpha=0.01, s=1, ax=ax)
        ax.set_title('XY origin - Plane X [2]')

        # XZ row
        ax = axes[1][0]
        self.tps.sig_p0.plot.scatter(x='bt_primary_z', y='bt_primary_x', alpha=0.01, s=1, ax=ax)
        ax.set_title('XZ origin - Plane U [0]')

        ax = axes[1][1]
        self.tps.sig_p1.plot.scatter(x='bt_primary_z', y='bt_primary_x', alpha=0.01, s=1, ax=ax)
        ax.set_title('XZ origin - Plane V [1]')

        ax = axes[1][2]
        self.tps.sig_p2.plot.scatter(x='bt_primary_z', y='bt_primary_x', alpha=0.01, s=1, ax=ax)
        ax.set_title('XZ origin - Plane X [2]')

        # YZ row
        ax = axes[2][0]
        self.tps.sig_p0.plot.scatter(x='bt_primary_y', y='bt_primary_z', alpha=0.01, s=1, ax=ax)
        ax.set_title('YZ origin - Plane U [0]')

        ax = axes[2][1]
        self.tps.sig_p1.plot.scatter(x='bt_primary_y', y='bt_primary_z', alpha=0.01, s=1, ax=ax)
        ax.set_title('YZ origin - Plane V [1]')

        ax = axes[2][2]
        self.tps.sig_p2.plot.scatter(x='bt_primary_y', y='bt_primary_z', alpha=0.01, s=1, ax=ax)
        ax.set_title('YZ origin - Plane X [2]')

        fig.suptitle(f"{signal_label} TP point of origin on XY, XZ and YZ planes")

        fig.tight_layout()
        return fig
    
    def draw_tp_sig_drift_depth_dist(self, weight_by:str = None, bins=100, figsize=(12, 5)):

        tps=self.tps

        x_label = 'drift depth [x]'
        y_label = weight_by if weight_by else 'counts'
        weight_var = f'{weight_by}' if weight_by else ''
        sup_title = f"{ weight_by+" sum" if weight_by else 'TP count' } vs true drift depth for signal TPs"

        fig,axes = plt.subplots(1, 3, sharey=True, figsize=figsize)

        ax = axes[0]
        w = tps.sig_p0[weight_var] if weight_by else None
        tps.sig_p0.bt_primary_x.hist(bins=bins, weights=w, ax=ax)
        ax.set_title('U plane [0]')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax = axes[1]
        w = tps.sig_p1[weight_var] if weight_by else None
        tps.sig_p1.bt_primary_x.hist(bins=bins, weights=w, ax=ax)
        ax.set_title('V plane [1]')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax = axes[2]
        w = tps.sig_p2[weight_var] if weight_by else None
        tps.sig_p2.bt_primary_x.hist(bins=bins, weights=w,ax=ax)
        ax.set_title('X plane [2]')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        fig.suptitle(sup_title)
        fig.tight_layout()
        return fig


    def draw_tp_start_sample_dist(self, figsize=(12,5)):
        """Draw the distribution of start_sample TP variable

        Args:
            figsize (tuple, optional): size of the figure. Defaults to (12,5).

        Returns:
            _type_: _description_
        """
        fig,axes= plt.subplots(1,3, figsize=figsize, sharey=True)
        tps=self.tps
        n_bins=(tps.all.sample_start.max()-tps.all.sample_start.min())//100

        ax=axes[0]
        ax.hist([tps.sig_p0.sample_start,tps.noise_p0.sample_start], stacked=True, bins=n_bins, log=True)
        ax.set_title('Plane U [0]')
        ax.set_xlabel('TP peak time sample')
        ax.set_ylabel('counts')
        ax.set_ylim(bottom=1) 
        ax.legend(['Signal', 'Noise'])

        ax=axes[1]
        ax.hist([tps.sig_p1.sample_start,tps.noise_p1.sample_start], stacked=True, bins=n_bins, log=True)
        ax.set_title('Plane V [1]')
        ax.set_xlabel('TP peak time sample')
        ax.set_ylabel('counts')
        ax.set_ylim(bottom=1) 
        ax.legend(['Signal', 'Noise'])

        ax=axes[2]
        ax.hist([tps.sig_p2.sample_start,tps.noise_p2.sample_start], stacked=True, bins=n_bins, log=True)
        ax.set_title('Plane X [2]')
        ax.set_xlabel('TP peak time sample')
        ax.set_ylabel('counts')
        ax.set_ylim(bottom=1) 
        ax.legend(['Signal', 'Noise'])

        fig.suptitle(f"{self.signal_name} and Noise TPs distribution in peak time")
        fig.tight_layout()
        return fig
    

    def draw_tp_event(self, event, figsize=(12,5)):

        ev_tps = self.tps.query(f'event == {event}')
        if len(ev_tps) == 0:
            raise RuntimeError(f"Event {event} not found. Events: {self.tps.all.event.unique()}")

        fig, axes = plt.subplots(1,3, figsize=figsize, sharex=True, sharey=True)


        cmap = plt.get_cmap('tab10')

        # all_tps_ev10 = snn.TPSignalNoiseSelector(ws.tps[ws.tps.event == 10])

        # fig, axes = plt.subplots(2,2, figsize=(10,8), sharex=True, sharey=True)
        xlabel = 'channel'
        ylabel = 'time (sample_peak)'
        alpha = 0.5
        marker_size = 2
        ax = axes[0]
        ev_tps.noise_p0.plot.scatter(x='channel', y='sample_peak', color=cmap(1), alpha=alpha, s=marker_size, ax=ax)
        ev_tps.sig_p0.plot.scatter(x='channel', y='sample_peak', color=cmap(0), alpha=alpha, s=marker_size, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax = axes[1]
        ev_tps.noise_p1.plot.scatter(x='channel', y='sample_peak', color=cmap(1), alpha=alpha, s=marker_size, ax=ax)
        ev_tps.sig_p1.plot.scatter(x='channel', y='sample_peak', color=cmap(0), alpha=alpha, s=marker_size, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)


        ax = axes[2]
        ev_tps.noise_p2.plot.scatter(x='channel', y='sample_peak', color=cmap(1), alpha=alpha, s=marker_size, ax=ax)
        ev_tps.sig_p2.plot.scatter(x='channel', y='sample_peak', color=cmap(0), alpha=alpha, s=marker_size, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        fig.suptitle(f"Event {event}, signal and noise TPs")
        fig.tight_layout()
        return fig
    

    def draw_tp_signal_noise_dist(self, roview: int=2, figsize=(12,5)) -> mpl.figure.Figure:
        """Draw the distributions of adc_peak, adc_integral, and samples_over_threshold variables

        Args:
            figsize (tuple, optional): Figure size. Defaults to (12,5).

        Returns:
            mpl.figure.Figure: A figure containing the 3 distributions
        """
        max_bins = 1024
        
        fig, axes= plt.subplots(1,3, figsize=figsize, sharey=True)
        tps=self.tps

        # var='peakADC'

        var_list = ['adc_peak', 'adc_integral', 'samples_over_threshold']
        for i,var in enumerate(var_list):

            # Auto-bin
            # Find the variable range
            x_min=tps.all_by_view[roview][var].min()
            x_max=tps.all_by_view[roview][var].max()
            x_range=(x_max-x_min)
            if x_range < max_bins:
                n_bins=int(x_range)
                dx = x_range/n_bins
            else:
                dx = float((x_range // max_bins)+1)
                n_bins = int(x_range/dx)
            bins = [ (x_min + i*dx) for i in range(n_bins+1)]
            ax=axes[i]
            tps.sig_by_view[roview][var].hist(bins=bins, ax=ax,log=True, alpha=0.75)
            ax.set_ylabel(f"Counts")
            ax.set_xlabel(f"{var}")
            tps.noise_by_view[roview][var].hist(bins=bins, ax=ax,log=True, alpha=0.75)
            tps.all_by_view[roview][var].hist(bins=bins, ax=ax, edgecolor='black', histtype='step', log=True)
            ax.set_title(var)

            ax.legend((f'{self.signal_name}+noise', self.signal_name, 'noise'))

        fig.suptitle(f"{', '.join(var_list)} distributions for : {self.signal_name} vs noise")
        fig.tight_layout()
        return fig
    

    def draw(self, var, n_x_bins=30, log=False, figsize=(10,8)):
        """What does this draw? Obsolete!

        Args:
            var (_type_): _description_
            n_x_bins (int, optional): _description_. Defaults to 30.
            log (bool, optional): _description_. Defaults to False.
            figsize (tuple, optional): _description_. Defaults to (10,8).

        Returns:
            _type_: _description_
        """

        tps=self.tps

        # Split the dataset into bins by depth
        g = tps.sig_p2.groupby(pd.cut(tps.sig_p2.TP_trueX, n_x_bins))

        fig, axes = subplot_autogrid(len(g), figsize=figsize, sharey=True)

        col=f'TP_{var}'

        # x_min=tps.p2[col].min()
        # x_max=tps.p2[col].max()

        # x_range=(x_max-x_min)
        # n_bins=int(x_range)//10
        # dx = x_range/n_bins

        # bins = [ (x_min + i*dx) for i in range(n_bins+1)]
        bins=10

        for k, (i, df) in enumerate(g):
            ax = axes[k]
            df[col].hist(bins=bins,ax=ax, log=log)
            ax.set_title(f"x = ({i.left:.0f},{i.right:.0f})")

        fig.suptitle(col)
        fig.tight_layout()
        return fig
    

    def draw_variable_in_drift_grid(self, var, n_x_bins=30, downsampling=10, log=False, sharex=False, sharey=False, figsize=(10,8)):

        tps=self.tps
        # Split the dataset into bins by depth
        g = tps.sig_p2.groupby(pd.cut(tps.sig_p2.bt_primary_x, n_x_bins), observed=False)

        fig, axes = subplot_autogrid(len(g), figsize=figsize, sharex=sharex, sharey=sharey)

        col=f'{var}'

        # bins = [ (x_min + i*dx) for i in range(n_bins+1)]
        bins=calculate_natural_bins(tps.p2[col], downsampling)

        for k, (i, df) in enumerate(g):
            ax = axes[k]
            df[col].hist(bins=bins,ax=ax, log=log)
            ax.set_title(f"{i.left:.0f} < x < {i.right:.0f}")
            ax.set_xlabel(var)

        fig.suptitle(f"{var} - plane 2")
        fig.tight_layout()
        return fig
    

    def draw_variable_drift_stack(self, var, n_x_bins=30, downsampling=10, log=False, figsize=(10,8)):

        tps=self.tps
        # Split the dataset into bins by depth
        g = tps.sig_p2.groupby(pd.cut(tps.sig_p2.bt_primary_x, n_x_bins), observed=False)

        fig, ax = plt.subplots(figsize=figsize)

        col=f'{var}'

        bins=calculate_natural_bins(tps.p2[col], downsampling)

        l = []
        for k, (i, df) in enumerate(g):
            df[col].hist(bins=bins,ax=ax, log=log, histtype='step', lw=2)
            l+=[f"{i.left:.0f} < x < {i.right:.0f}"]

        ax.legend(l)
        ax.set_xlabel(var)
        ax.set_ylabel('counts')

        fig.suptitle(f"{col} distribution by drift depth bins (collection)")
        fig.tight_layout()
        return fig
    

    def draw_variable_cut_sequence(self, var:str , thresholds, log=False, figsize=(12,5)):
        
        """Draw the impact of a set of incremantal cuts on 'var' over 
         'adc_peak', 'adc_integral', 'samples_over_threshold' variables

        Args:
            var (_type_): _description_
            thresholds (_type_): _description_
            log (bool, optional): _description_. Defaults to False.
            figsize (tuple, optional): _description_. Defaults to (12,5).

        Returns:
            _type_: _description_
        """
        tps=self.tps

        var_list = ['adc_peak', 'adc_integral', 'samples_over_threshold']
        var_labels = {
            'samples_over_threshold': 'sot',
            'adc_peak': 'adc peak',
            'adc_integral': 'adc int.',
        }

        bins_adc_peak = calculate_natural_bins(tps.p2.adc_peak, 5)
        bins_adc_int = calculate_natural_bins(tps.p2.adc_integral, 5)
        bins_SOT = calculate_natural_bins(tps.p2.samples_over_threshold, 1)

        bins_map = {
            'samples_over_threshold': bins_SOT,
            'adc_peak': bins_adc_peak,
            'adc_integral': bins_adc_int,
        }

        # cmap='tab20'
        cmap_name='cividis'
        cmap = mpl.colormaps[cmap_name]
        # cmap=cmo.matter
        n_lines = len(thresholds)

        # Take colors at regular intervals spanning the colormap.
        colors = cmap(np.linspace(0, 1, n_lines))
    
        fig, axes = plt.subplots(3,3, figsize=figsize)
        gs = axes[2, 0].get_gridspec()
        # remove the underlying axes
        for ax in axes[-1,:]:
            ax.remove()
        ax_trailer = fig.add_subplot(gs[2, :-1])
        ax_table = fig.add_subplot(gs[2, -1:])

        df = tps.sig_p2

        for i,v in enumerate(var_list):
            b = bins_map[v]
            axes[0][i].hist([df[df[var] > thres][v] for thres in reversed(thresholds)], histtype='stepfilled', color=colors, bins=b)
            axes[0][i].set_title(self.signal_name)
            
        for i,v in enumerate(var_list):
            axes[0][i].set_xlabel(var_labels[v])
            axes[0][i].legend([f"{var_labels.get(var, var)} > {t}" for t in thresholds])

            if log:
                axes[0][i].set_yscale("log")

        df = tps.noise_p2

        for i,v in enumerate(var_list):
            b = bins_map[v]
            axes[1][i].hist([df[df[var] > thres][v] for thres in reversed(thresholds)], histtype='stepfilled', color=colors, bins=b)
            axes[1][i].set_title('Noise')
            
        for i,v in enumerate(var_list):
            axes[1][i].set_xlabel(var_labels[v])
            axes[1][i].legend([f"{var_labels.get(var, var)} > {t}" for t in thresholds])

            if log:
                axes[1][i].set_yscale("log")

        df = tps.sig_p2
        ax_trailer.hist([df[df[var] > thres].bt_primary_x for thres in reversed(thresholds)], histtype='stepfilled', color=colors, bins=100)
        ax_trailer.set_xlabel("drift coordinate")
        ax_trailer.set_title(self.signal_name)
        ax_trailer.legend([f"{var_labels.get(var, var)} > {t}" for t in thresholds])

        df = self.do_threshold_scan(2, var, thresholds)
        ax_table.axis('off')
        ax_table.axis('tight')
        df_table = df[['threshold', 'rate_sig', 'rate_noise']].copy()
        df_table[['rate_sig', 'rate_noise']] = (df_table[['rate_sig', 'rate_noise']]/1e6).map('{:,.3f} MHz'.format)

        the_table = ax_table.table(cellText=df_table.values, colLabels=df_table.columns, loc='center')
        the_table.auto_set_column_width([0,1,2])


        fig.suptitle(f"Incremental {var} cuts (collection)")

        fig.tight_layout()
        return fig
    

    def do_threshold_scan(self, plane_id: int, var: str, values: list) -> pd.DataFrame:
        """Performs a threshold scan on variable `var` on trigger primitives of plane `plane_id` 

        Args:
            plane_id (int): plane identifier
            var (str): name of the variable to cut on
            values (List[int]): threshold values

        Returns:
            pd.DataFrame: Dataframe containing number of signal and noise tps, and corresponding rates, for each threshold value
        """
        
        tps=self.tps

        # df_sig = getattr(tps, f'sig_p{plane_id:d}')
        # df_noise = getattr(tps, f'noise_p{plane_id:d}')
        df_sig = tps.sig_by_view[plane_id]
        df_noise = tps.noise_by_view[plane_id]

        df_sig_seq = [df_sig.query(f"{var} > {t}") for t in values]
        df_noise_seq = [df_noise.query(f"{var} > {t}") for t in values]


        n_sig_tps = [len(df) for df in df_sig_seq]
        n_noise_tps = [len(df) for df in df_noise_seq]
        rate_sig_tps = [df_to_tp_rates(df) for df in df_sig_seq]
        rate_noise_tps = [df_to_tp_rates(df) for df in df_noise_seq]


        df = pd.DataFrame({
            'threshold': values,
            'n_sig': n_sig_tps,
            'n_noise': n_noise_tps,
            'rate_sig': rate_sig_tps,
            'rate_noise': rate_noise_tps
        })

        df['n_tot'] = df.n_sig + df.n_noise
        df['rate_tot'] = df.rate_sig + df.rate_noise
        
        df['sig_frac'] = df.n_sig/len(df_sig)
        df['noise_frac'] = df.n_noise/len(df_noise)

        return df
    

    def draw_threshold_scan(self, var, thresholds, figsize=(12,6)):

        # Perform threshold scan on collection plane
        df = self.do_threshold_scan(2, var, thresholds)

        fig,axes= plt.subplots(2,3, figsize=figsize)

        df['noise_dominance'] = df.n_noise/df.n_sig
        df['purity'] = df.n_sig/(df.n_noise+df.n_sig)
        df['completeness'] = df.n_sig/len(self.tps.sig_p2)

        cmap='tab20'
    

        ax=axes[0][0]
        df.plot(x='noise_frac', y='sig_frac', lw=0.5,  ax=ax)
        s = df.plot.scatter(x='noise_frac', y='sig_frac', c='threshold', cmap=cmap, ax=ax)
        ax.set_ylabel(f'{self.signal_name} fraction')
        ax.set_xlabel('Noise fraction')


        ax=axes[0][1]
        df.plot(x='n_noise', y='n_sig', lw=0.5,  ax=ax)
        df.plot.scatter(x='n_noise', y='n_sig', c='threshold',  cmap=cmap, ax=ax)
        ax.set_ylabel(f'{self.signal_name} counts')
        ax.set_xlabel('Noise counts')

        ax=axes[0][2]
        df.plot(x='rate_noise', y='rate_sig', lw=0.5,  ax=ax)
        df.plot.scatter(x='rate_noise', y='rate_sig', c="threshold", cmap=cmap, ax=ax)
        ax.set_ylabel(f'{self.signal_name} rate [Hz]')
        ax.set_xlabel('Noise rate [Hz]')
        
        ax=axes[1][0]
        df.plot(x='threshold', y='noise_dominance', ax=ax)
        ax.axhline(1, color='r', lw=1)
        ax.set_xlabel(f'Threshold ({var})')

        ax.set_ylabel('Counts(Noise)/Counts(Signal)')
        ax=axes[1][1]
        df.plot(x='threshold', y='purity', ax=ax)
        df.plot(x='threshold', y='completeness', ax=ax)
        ax.set_xlabel(f'Threshold ({var})')

        # ax.axvline(37, color='r', lw=1)
        ax=axes[1][2]
        df.plot(x='threshold', y='rate_tot', ax=ax)
        ax.set_xlabel(f'Threshold ({var})')


        fig.tight_layout()
        return fig
    