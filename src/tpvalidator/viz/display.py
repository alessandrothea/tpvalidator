
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tpvalidator.detgeometry import get_by_geocfg_id



def equalize_ranges(df) -> pd.DataFrame:
    x = df.agg(['min', 'max'])
    med = (x.loc['max']+x.loc['min'])/2
    rng = (x.loc['max']-x.loc['min'])

    max_rng = rng.max()

    upper = med+max_rng/2
    lower = med-max_rng/2
    return pd.concat([lower,upper], axis=1).T

class TriggerPrimitivesEventViewer:
    
    labels = {
        'tps': 'tps',
        'mctruths': 'mctruths',
    }

    def __init__(self, ws, labels={}):

        df_labels = self.labels.copy()
        df_labels.update(labels)

        self.ws = ws
        self.tps = getattr(ws, df_labels['tps'])
        self.mctruths = getattr(ws, df_labels['mctruths'])
        self.geo = get_by_geocfg_id(ws.info['geo']['detector'])


    #----------
    def draw_tps_point_of_origin(self, ev_uid: int, ev_label: str = "", readout_view = 2, **kwargs) -> None:
        """Draw the trigger primitives point of origin, based on backtracked informations

        Args:
            ax (plt.Axes): _description_
            tp_data (BasicTPData): _description_
            ev_uid (int): _description_
            ev_label (str, optional): _description_. Defaults to "".
        """

        fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, **kwargs)


        # ws=self.ws
        tps=self.tps
        mctruths=self.mctruths

        selection = [
            'bt_is_signal == True',
            f'event_uid == {ev_uid}',
        ] 
        selection += [f'readout_view == {readout_view}'] if readout_view else []


        tps = tps.query(' & '.join([f'({q})' for q in selection]))

        if len(tps) == 0:
            return fig


        # equalize the range/
        ax_ranges = equalize_ranges(tps[['bt_primary_x', 'bt_primary_y', 'bt_primary_z']])
        
        # Draw 3D points
        # ax.scatter(tps.bt_primary_y, tps.bt_primary_z, tps.bt_primary_x)
        scat = ax.scatter(tps.bt_primary_y, tps.bt_primary_z, tps.bt_primary_x, s=tps.samples_over_threshold/2, c=tps.adc_integral)#, vmin=vmin, vmax=vmax)
        # scat = ax.scatter(tps.bt_primary_y, tps.bt_primary_z, tps.bt_primary_x, s=tps.samples_over_threshold/2, c=tps.ta_win_id)#, vmin=vmin, vmax=vmax)

        # Add projections on the YZ plane (CRP) and XY plane (collection/drift)
        ax.scatter(np.full_like(tps.bt_primary_y, ax_ranges.bt_primary_y[0]), tps.bt_primary_z, tps.bt_primary_x, s=tps.samples_over_threshold/2, c='gray')
        ax.scatter(tps.bt_primary_y, np.full_like(tps.bt_primary_z, ax_ranges.bt_primary_z[1]), tps.bt_primary_x, s=tps.samples_over_threshold/2, c='gray')
        ax.scatter(tps.bt_primary_y, tps.bt_primary_z, np.full_like(tps.bt_primary_x, ax_ranges.bt_primary_x[0]), s=tps.samples_over_threshold/2, c='gray')

        ax.set_xlim3d(*list(ax_ranges.bt_primary_y))
        ax.set_ylim3d(*list(ax_ranges.bt_primary_z))
        ax.set_zlim3d(*list(ax_ranges.bt_primary_x))

        ax.set_xlabel("y (collection)")
        ax.set_ylabel("z (beam)")
        ax.set_zlabel("x (drift)")
        # ax.set_title(ev_label)

        fig.colorbar(scat, shrink=0.5, aspect=15)
        fig.tight_layout()
        return fig


    #----------
    def draw_tps(self, ev_uid: int, **kwargs):

        fig, ax = plt.subplots(**kwargs)
        
        tps=self.tps.query(f'event_uid == {ev_uid}')
        

        ax.scatter(x=tps.channel, y=tps.samples_to_peak, c=tps.TPCSetID)

        
        
        fig.tight_layout()


        return fig
    


    def decorate_tpc_coords(self):

        tps = self.ws.tps.copy()
        tps['tpc_view_channel'] = tps.channel.apply(lambda c: self.geo.tpc_view_channel(c)[1]).astype('int16')
        tps[['tpc_j', 'tpc_k']] = pd.DataFrame(tps.TPCSetID.apply(self.geo.tpc_id_to_grid).tolist(), index=tps.index).rename({0:'tpc_j', 1:'tpc_k'}, axis=1)
        tps['tpc_z_channel'] = (tps['tpc_view_channel']+self.geo.tpc_view_2_num_chans_sim*tps['tpc_k']).where(tps['readout_view'] == 2, -999999)


        return tps
    

    def plot_yzt_event_view(self, ev_uid: int, ax:object=None):
        
        tps_event = self.ws.tps.query(f'event_uid == {ev_uid} & readout_plane_id == 2 & bt_is_signal==1').copy()
        tps_event['tpc_view_channel'] = tps_event.channel.apply(lambda c: self.geo.tpc_view_channel(c)[1]).astype('int16')
        tps_event[['tpc_j', 'tpc_k']] = pd.DataFrame(tps_event.TPCSetID.apply(self.geo.tpc_id_to_grid).tolist(), index=tps_event.index).rename({0:'tpc_j', 1:'tpc_k'}, axis=1)
        tps_event['tpc_z_channel'] = (tps_event['tpc_view_channel']+self.geo.tpc_view_2_num_chans_sim*tps_event['tpc_k']).where(tps_event['readout_view'] == 2, -999999)

        # TODO: cleanup
        jmin, jmax = 0, self.geo.num_y_rows-1
        n = self.geo.num_y_rows

        tab10_colors = plt.get_cmap('tab20').colors
        cmap = mcolors.ListedColormap(tab10_colors[:n])
        norm = mcolors.BoundaryNorm(boundaries=np.arange(jmin-0.5, jmax + 0.5+1, 1), ncolors=n)


        create_fig = ax is None
        if create_fig:
            fig, ax = plt.subplots() if create_fig else (ax.figure, ax)

        tps_event.plot.scatter(x='tpc_z_channel', y='sample_start', c='tpc_j', cmap=cmap, norm=norm, ax=ax)

        if create_fig:
            fig.tight_layout()
        return fig