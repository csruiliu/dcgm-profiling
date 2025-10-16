import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt


def plot_single_job_metric(job_data_path, job_id, metric):
    pq_df = pd.read_parquet(f"{job_data_path}/{job_id}.pq", engine='pyarrow')
    plt.figure(figsize=(12, 6))
    ax = sns.scatterplot(data=pq_df, x=pq_df.index, y=metric, hue="hostname", style="gpu_id")
    ax.set_xlabel("Timestep")
    plt.legend(loc='best')
    plt.savefig
    plt.savefig(f"{job_id}_{metric}.png", dpi=600, format='png', bbox_inches='tight')
    plt.close()


def main():
    ###################################
    # get all parameters
    ###################################
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--job_id', action='store', type=str, required=True, 
                        help='plot the job associate with the job_id')
    parser.add_argument('--job_path', action='store', type=str, required=True, 
                        help='indicate the folder that consists of various jobs')
    args = parser.parse_args()

    job_id = args.job_id
    job_data_path = args.job_path
    
    metric_list = ["nersc_ldms_dcgm_gr_engine_active",
                   "nersc_ldms_dcgm_dram_active",
                   "nersc_ldms_dcgm_fp64_active",
                   "nersc_ldms_dcgm_fp32_active",
                   "nersc_ldms_dcgm_tensor_active",
                   "nersc_ldms_dcgm_pcie_tx_bytes",
                   "nersc_ldms_dcgm_pcie_rx_bytes"]
    
    for m in metric_list:
        plot_single_job_metric(job_data_path, job_id, m)


if __name__=="__main__":
    main()