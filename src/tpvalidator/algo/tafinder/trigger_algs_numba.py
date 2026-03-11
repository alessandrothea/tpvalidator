import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed 
from numba import njit


@njit
def numba_dbscan(z, t, eps, min_samples):
    N = len(z) #number of points in window 
    #initialise dbscan labels to -1 (i.e. noise)
    labels = -1 * np.ones(N, dtype=np.int32) 
    visited = np.zeros(N, dtype=np.uint8)
    cluster_id = 0
    eps2 = eps * eps

    for i in range(N):
        if visited[i]:
            continue

        visited[i] = 1

        # Finding neighbours
        neigh = []
        zi, ti = z[i], t[i]

        for j in range(N):
            dz = z[j] - zi
            dt = t[j] - ti
            if dz*dz + dt*dt <= eps2:
                neigh.append(j)

        if len(neigh) < min_samples:
            labels[i] = -1
        else:
            labels[i] = cluster_id

            k = 0
            while k < len(neigh):
                j = neigh[k]

                if not visited[j]:
                    visited[j] = 1

                    # Expanding neighbours of j
                    zj, tj = z[j], t[j]
                    neigh2 = []
                    for m in range(N):
                        dz = z[m] - zj
                        dt = t[m] - tj
                        if dz*dz + dt*dt <= eps2:
                            neigh2.append(m)

                    if len(neigh2) >= min_samples:
                        for m in neigh2:
                            if m not in neigh:
                                neigh.append(m)

                if labels[j] == -1:
                    labels[j] = cluster_id

                k += 1

            cluster_id += 1

    return labels


def add_dbscan_variables(tps: pd.DataFrame) -> None:
    """Add DB scan variables to the tps dataframe, namely

    `dbs_t`: conversion from time in the local readout window to cm
    `dbs_z`: conversoon from channel number to cm

    Args:
        tps (_type_): _description_
    """
    # conversion parameters to move from channel ID & ticks to cm. 
    hd_wire_pitch = 0.48 #cm 
    vd_wire_pitch = 0.51 #cm 
    wire_pitch = vd_wire_pitch
    drift_velocity = 0.16 #cm/us
    sampling_rate = 0.5 #us/tick 
    cm_per_tick = drift_velocity * sampling_rate

    tps['dbs_t'] = (tps["sample_start"].to_numpy() * cm_per_tick).astype(np.float32)
    tps['dbs_z'] = (tps["channel"].to_numpy() * wire_pitch).astype(np.float32)

    return tps


def apply_dbscan(window_tps, epsilon=2, min_samples=2):

    # conversion parameters to move from channel ID & ticks to cm. 
    hd_wire_pitch = 0.48 #cm 
    vd_wire_pitch = 0.51 #cm 
    wire_pitch = vd_wire_pitch
    drift_velocity = 0.16 #cm/us
    sampling_rate = 0.5 #us/tick 
    cm_per_tick = drift_velocity * sampling_rate

    df = window_tps

    time_cm = (df["sample_start"].to_numpy() * cm_per_tick).astype(np.float32)
    z = (df["channel"].to_numpy() * wire_pitch).astype(np.float32)
    adc = df["adc_integral"].to_numpy()
    df_index = df.index.to_numpy()

    labels = numba_dbscan(z, time_cm, eps=epsilon, min_samples=min_samples) 

    valid = labels != -1
    if not np.any(valid):
        return pd.Series({
            "n_clusters": 0,
            "mean_cluster_sadc": 0,
            "total_cluster_sadc": 0,
            "max_cluster_sadc": 0,
            "tp_index": [],
            "dbscan_label": []
        })

    cluster_ids = np.unique(labels[valid])
    clusters = [adc[labels == cid] for cid in cluster_ids]
    cluster_sums = np.array([c.sum() for c in clusters])


    return pd.Series({
        "n_clusters": len(cluster_ids),
        "mean_cluster_sadc": cluster_sums.mean(),
        "total_cluster_sadc": cluster_sums.sum(),
        "max_cluster_sadc": cluster_sums.max(),
        "tp_index": df_index,
        "dbscan_label": list(labels)
    })
