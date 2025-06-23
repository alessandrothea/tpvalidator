import click
from pathlib import Path
from rich import print

from .utilities import load_data, calculate_angles, calculate_angles_2
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


# name of branches in the TTree to read into the pandas df
MC_BRANCHES = ["event", "Eng", "Ekin", "startX", "startY", "startZ",  "Px", "Py", "Pz", "P"]#, "label"]
TP_BRANCHES = ["event", "n_TPs", "TP_channel", "TP_startT", "TP_peakT", "TP_peakADC", 
               "TP_SADC", "TP_TOT", "TP_plane", "TP_TPC", "TP_trueX", "TP_trueY", 'TP_trueZ', 'TP_signal']



@click.command()
@click.option("-o", "--output", type=str, default=None)
@click.argument("data_path", type=click.Path(exists=True))
def main(data_path, output) -> None:
    print("Hello from tpvalidator!")

    data_path = Path(data_path)
    report_name = output if not output is None else f'tp_val_{data_path.stem}.pdf'


    df_tps = load_data(data_path, branch_names=TP_BRANCHES)
    df_mc = load_data(data_path, branch_names=MC_BRANCHES)
    df_all = load_data(data_path)


    theta_y, theta_y_U, theta_y_V, theta_xz, theta_xz_U, theta_xz_V = calculate_angles(df_mc.Px, df_mc.Py, df_mc.Pz, df_mc.P)

    df_mc['theta_y'] = theta_y
    df_mc['theta_yU'] = theta_y_U
    df_mc['theta_yV'] = theta_y_V
    df_mc['theta_xz'] = theta_xz
    df_mc['theta_xzU'] = theta_xz_U
    df_mc['theta_xzV'] = theta_xz_V


    theta_drift, theta_beam, theta_coll, phi_bd, phi_ind_u, phi_ind_v = calculate_angles_2(df_mc.Px, df_mc.Py, df_mc.Pz, df_mc.P)
    df_mc['theta_drift'] = theta_drift
    df_mc['theta_beam'] = theta_beam
    df_mc['theta_coll'] = theta_coll
    df_mc['phi_coll'] = phi_bd
    df_mc['phi_ind_u'] = phi_ind_u
    df_mc['phi_ind_v'] = phi_ind_v

    with PdfPages(report_name) as pdf:
        df_all.hist(bins=100, figsize=(11.69,8.27), color='g')
        plt.tight_layout()
        pdf.savefig()
        # fig, ax = plt.subplots()
        df_tps.hist(bins=50, figsize=(11.69,8.27))
        plt.tight_layout()
        pdf.savefig()
        df_mc.hist(bins=100, figsize=(11.69,8.27), color='k')
        plt.tight_layout()
        pdf.savefig()
        df_all[['totQ_X', 'totQ_U', 'totQ_V', 'detQ_X', 'detQ_U', 'detQ_V']].hist(bins=100, figsize=(11.69,8.27), color='r')
        plt.tight_layout()
        pdf.savefig()


    print("aa")

    import IPython
    IPython.embed(colors='linux')


