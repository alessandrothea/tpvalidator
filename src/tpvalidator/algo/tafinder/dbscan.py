from sklearn.cluster import DBSCAN
import pandas as pd

def ApplyDBScan(window_tps: pd.DataFrame, epsilon=1.5, min_samples=2):
    # Detector params
    # wire_pitch = 4.8        # mm HD
    wire_pitch = 5.1        # mm
    drift_velocity = 1.6     # mm/us
    sampling_rate = 0.5      # us/tick
    mm_per_tick = drift_velocity * sampling_rate  # mm/tick
    
    df = window_tps.copy()  
    df['dbscan_label'] = -1
    df['sample_start'] = df['sample_start']  * mm_per_tick / (10)  # convert to cm
    df['z'] = df['channel'] * wire_pitch /10  # channel index to distance in cm
    input_data = df[['z', 'sample_start']].values
    
    labels = DBSCAN(eps=epsilon, min_samples=min_samples).fit_predict(input_data)
    df['dbscan_label'] = labels

    n_clusters = df[df['dbscan_label'] != -1]['dbscan_label'].nunique()
    
    if n_clusters > 0:
        mean_cluster_sadc = df[df['dbscan_label'] != -1].groupby('dbscan_label')['adc_integral'].sum().mean() #mean of total cluster energies 
        total_cluster_sadc = df[df['dbscan_label'] != -1].groupby('dbscan_label')['adc_integral'].sum().sum() #sum of total cluster_energies 
        max_cluster_sadc = df[df['dbscan_label'] != -1].groupby('dbscan_label')['adc_integral'].sum().max() #max of  cluster_energies 

    else:
        mean_cluster_sadc = total_cluster_sadc = max_cluster_sadc = 0
    
    return n_clusters, mean_cluster_sadc, total_cluster_sadc, max_cluster_sadc


