import click
from pathlib import Path
from rich import print
import math

from .utilities import load_data, calculate_angles, calculate_angles_2
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# name of branches in the TTree to read into the pandas df
MC_BRANCHES = ["event", "Eng", "Ekin", "startX", "startY", "startZ",  "Px", "Py", "Pz", "P"]#, "label"]
TP_BRANCHES = ["event", "n_TPs", "TP_channel", "TP_startT", "TP_peakT", "TP_peakADC", 
               "TP_SADC", "TP_TOT", "TP_plane", "TP_TPC", "TP_trueX", "TP_trueY", 'TP_trueZ', 'TP_signal']

def get_hist_layout(n_items, layout=None):
    if layout is not None:
        return layout
    ncols = math.ceil(math.sqrt(n_items))
    nrows = math.ceil(n_items / ncols)
    return (nrows, ncols)

@click.command()
@click.option("-i", "--interactive", type=bool, is_flag=True, default=False)
@click.option("-o", "--output", type=str, default=None)
@click.argument("data_path", type=click.Path(exists=True))
def main(data_path, interactive, output) -> None:
    print("Hello from tpvalidator!")

    data_path = Path(data_path)
    report_name = output if not output is None else f'tp_val_{data_path.stem}.pdf'

    df_events = load_data(data_path, branch_names=['event'])

    df_tps = load_data(data_path, branch_names=TP_BRANCHES)
    df_mc = load_data(data_path, branch_names=MC_BRANCHES)
    df_all = load_data(data_path)

    df_angles = df_mc[['event']].copy(deep=True)

    theta_y, theta_y_U, theta_y_V, theta_xz, theta_xz_U, theta_xz_V = calculate_angles(df_mc.Px, df_mc.Py, df_mc.Pz, df_mc.P)
    df_angles['theta_y'] = theta_y
    df_angles['theta_yU'] = theta_y_U
    df_angles['theta_yV'] = theta_y_V
    df_angles['theta_xz'] = theta_xz
    df_angles['theta_xzU'] = theta_xz_U
    df_angles['theta_xzV'] = theta_xz_V

    theta_drift, theta_beam, theta_coll, theta_u, theta_v, phi_coll, phi_ind_u, phi_ind_v = calculate_angles_2(df_mc.Px, df_mc.Py, df_mc.Pz, df_mc.P)
    df_angles['theta_drift'] = theta_drift
    df_angles['theta_beam'] = theta_beam
    df_angles['theta_coll'] = theta_coll
    df_angles['theta_u'] = theta_u
    df_angles['theta_v'] = theta_v
    df_angles['phi_coll'] = phi_coll
    df_angles['phi_ind_u'] = phi_ind_u
    df_angles['phi_ind_v'] = phi_ind_v


    events = [
        df_mc[(s / df_mc.P).abs() > 0.95].sort_values('P', ascending=False).iloc[0]['event']
        for s in (df_mc.Px, df_mc.Py, df_mc.Pz)
    ]
    

    def equalize_ranges(df) -> pd.DataFrame:
        x = df.agg(['min', 'max'])
        med = (x.loc['max']+x.loc['min'])/2
        rng = (x.loc['max']-x.loc['min'])

        max_rng = rng.max()

        upper = med+max_rng/2
        lower = med-max_rng/2
        return pd.concat([upper, lower], axis=1).T
         
        

    if not interactive:
        a4_landscape = (11.69,8.27)
        title_size = 20
        with PdfPages(report_name) as pdf:


            fig = plt.figure(figsize=a4_landscape)
            fig.clf()
            title_text = "Trigger Primitives Simulation:\nBasic Overview Plots"
            meta_text = (
                f"""
    File: {data_path}
    Number of events: {len(df_events)}
    """
            )
            fig.text(0.1, 0.95, title_text, ha='left', va='top', wrap=True, fontsize=30)
            fig.text(0.1, 0.85, meta_text, ha='left', va='top', wrap=True, fontsize=20)
            fig.tight_layout()
            pdf.savefig()

            df_all.hist(bins=50, figsize=a4_landscape, color='g')
            fig = plt.gcf()
            fig.suptitle("Tree variables overview", fontsize=title_size)
            fig.tight_layout()
            pdf.savefig()

            df_tps.hist(bins=50, figsize=a4_landscape)
            fig = plt.gcf()
            fig.suptitle("TP distributions", fontsize=title_size)
            fig.tight_layout()
            pdf.savefig()

            df_mc.hist(bins=50, figsize=a4_landscape, color='k')
            fig = plt.gcf()
            fig.suptitle("MC distributions", fontsize=title_size)
            fig.tight_layout()
            pdf.savefig()

            df_angles.hist(bins=50, figsize=a4_landscape, color='g')
            fig = plt.gcf()
            fig.suptitle("Angle distributions", fontsize=title_size)
            fig.tight_layout()
            pdf.savefig()




            #-----
            # Event display 3d
            title=['U','V', "X"]
            n_ev = len(events)
            vmax = df_tps[(df_tps.event.isin(events)) & (df_tps.TP_signal == 1)].TP_SADC.max()/5
            vmin = df_tps[(df_tps.event.isin(events)) & (df_tps.TP_signal == 1)].TP_SADC.min()

            # Row labels on the left side
            ev_labels = []
            for e, l in zip(events, ['x', 'y', 'z']):
                ev_row = df_mc[df_mc.event==e]
                Pax = ev_row[f'P{l}']
                ev_labels.append(f"Event {int(e)},\n[$\\vec{{P}} \\parallel \\hat{{{l}}}$, $P_{l}$={float(Pax.values)*1000:.2f} MeV]")


            fig, axes = plt.subplots(1,n_ev, subplot_kw={'projection': '3d'}, figsize=a4_landscape)
            fig.suptitle(f"TP point of origin for highest energy events along X, Y and Z)", fontsize=title_size)

            for e, en_num in enumerate(events):
                ax = axes[e]
                ev_label = ev_labels[e]
                mcf = df_mc[df_mc.event==en_num]
                tps = df_tps[(df_tps.event == en_num) & (df_tps.TP_signal == 1)]

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
            fig.tight_layout(rect=[0., 0., 0.95, 1])
            pdf.savefig()


            #-----
            # Event display
            fig, ax = plt.subplots(n_ev,3, figsize=a4_landscape)
            fig.suptitle(f"TP plane/time for highest energy events along X, Y and Z", fontsize=title_size)
                
            for e, ev_num in enumerate(events):
                mcf = df_mc[df_mc.event==ev_num]
                for plane in [0,1,2]:
                    x = df_tps[(df_tps.event == ev_num) & (df_tps.TP_plane == plane) & (df_tps.TP_signal == 1)]
                    # x = df_tps[(df_tps.event == event) & (df_tps.TP_plane == plane)]
                    s = ax[e][plane].scatter(x.TP_channel, x.TP_peakT,s=x.TP_TOT/2, c=x.TP_SADC, vmin=vmin, vmax=vmax, label = f"{int(mcf.Ekin.values)} MeV electron")
                    if plane == 2:
                        plt.colorbar(s, ax=ax[e][plane], label="SADC")

                for i in range(0,3):
                    ax[e][i].grid(alpha=0)
                    # ax[e][i].set_xlim(11e3,16e3)
                    # ax[e][i].set_ylim(np.mean(x.TP_peakT.values) -500, np.mean(x.TP_peakT.values) +500 )
                    ax[e][i].set_title(title[i])
                    ax[e][i].set_xlabel("channel ID")
                    ax[e][i].set_ylabel("time")

                    ax[e][i].legend()

            for i, label in enumerate(ev_labels):
                # Place text to the left of the row
                fig.text(0.02, 0.8 - i * 0.3, label, va='center', ha='left', fontsize=12, rotation=90)


            fig.tight_layout(rect=[0.05, 0, 0.98, 1])  # leave space on the left for labels

            # fig.tight_layout()
            pdf.savefig()
            # fig = plt.figure(figsize=a4_landscape)
            # ax = fig.add_subplot(111, projection='3d')




            #-----
            # Charge
            x = df_all[['totQ_X', 'totQ_U', 'totQ_V', 'detQ_X', 'detQ_U', 'detQ_V']].copy()
            x['rQ_X'] = x.detQ_X/x.totQ_X
            x['rQ_U'] = x.detQ_U/x.totQ_U
            x['rQ_V'] = x.detQ_V/x.totQ_V
            x.hist(bins=100, figsize=(a4_landscape[0], a4_landscape[1]), layout=(3,3), color='r')
            fig = plt.gcf()
            fig.suptitle("Charge distributions", fontsize=title_size)
            fig.tight_layout()
            pdf.savefig()

            fig, ax = plt.subplots()
            df_tps["TP_plane"].hist(ax=ax, figsize=a4_landscape)
            fig.suptitle("TP distribution by plane", fontsize=title_size)
            fig.tight_layout()
            pdf.savefig()


            fig, ax = plt.subplots(figsize=a4_landscape)
            ax.scatter(df_mc.Ekin, df_tps[df_tps.TP_plane==2].groupby('event').TP_SADC.sum(), c='k', marker='x')
            fig.suptitle("SADC vs Ekin", fontsize=title_size)
            ax.set_ylabel("total collection SADC")
            ax.set_xlabel("electron kinetic energy [MeV]")
            fig.tight_layout()
            pdf.savefig()

            fig, ax = plt.subplots(figsize=a4_landscape)
            ax.scatter(df_tps[df_tps.TP_plane==2].groupby('event').TP_SADC.sum(), df_tps[df_tps.TP_plane==0].groupby('event').TP_SADC.sum() , label ='U')
            ax.scatter(df_tps[df_tps.TP_plane==2].groupby('event').TP_SADC.sum(), df_tps[df_tps.TP_plane==1].groupby('event').TP_SADC.sum() , label ='V')
            ax.legend()
            fig.suptitle("SADC, collection vs induction planes", fontsize=title_size)
            ax.set_xlabel("Total collection SADC per event")
            ax.set_ylabel("Total induction SADC per event")
            fig.tight_layout()
            pdf.savefig()

        print("Done")

    if interactive:
        import IPython
        IPython.embed(colors='linux')


