import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse
import json
import glob
import matplotlib.pyplot as plt


# Mapping from metric names to GPU spec keys
metric_ref_mappings = {
    'TENSO': 'ref_fp64_tensor',
    'FP64A': 'ref_fp64',
    'FP32A': 'ref_fp32',
    'FP16A': 'ref_fp16'
}

metric_target_mappings = {
    'TENSO': 'target_fp64_tensor',
    'FP64A': 'target_fp64',
    'FP32A': 'target_fp32',
    'FP16A': 'target_fp16'
}

prec_ref_mappings = {
    'double': 'ref_fp64_tensor',
    'single': 'ref_tf32_tensor',
    'half': 'ref_fp16_tensor'
}

prec_target_mappings = {
    'double': 'target_fp64_tensor',
    'single': 'target_tf32_tensor',
    'half': 'target_fp16_tensor'
}

# I got the numbers from nvidia official website and https://www.techpowerup.com/gpu-specs
GPU_SPECS = {
    "A100-40": {
        "fp64": 9.7, "fp64_tensor": 19.5, "fp32": 19.5, "tf32_tensor": 156, "fp16": 78, "fp16_tensor": 312, 
        "mem_bw": 1555, "pcie_bw": 64, "nvlink_bw": 600, "base_clock": 1065, "boost_clock": 1410, "num_streams": 108
    },
    "A100-80": {
        "fp64": 9.7, "fp64_tensor": 19.5, "fp32": 19.5, "tf32_tensor": 156, "fp16": 78, "fp16_tensor": 312, 
        "mem_bw": 1935, "pcie_bw": 64, "nvlink_bw": 600, "base_clock": 1065, "boost_clock": 1410, "num_streams": 108
    },
    "A40": {
        "fp64": 0.58, "fp64_tensor": 0, "fp32": 37.4, "tf32_tensor": 74.8, "fp16": 37.4, "fp16_tensor": 149.7, 
        "mem_bw": 696, "pcie_bw": 64, "nvlink_bw": 112.5, "base_clock": 1305, "boost_clock": 1740, "num_streams": 84
    },
    "H100": {  # H100 SXM (default)
        "fp64": 34, "fp64_tensor": 67, "fp32": 67, "tf32_tensor": 989, "fp16": 133.8, "fp16_tensor": 1979, 
        "mem_bw": 3350, "pcie_bw": 128, "nvlink_bw": 900, "base_clock": 1590, "boost_clock": 1980, "num_streams": 132
    },
    "HYP-M-IO-A": {
        "fp64": 9.7*3.0, "fp64_tensor": 19.5*3.0, "fp32": 19.5*6.0, "tf32_tensor": 156*6.0, "fp16": 78*3.0, "fp16_tensor": 312*3.0, 
        "mem_bw": 1555*8.0, "pcie_bw": 64*25.0, "nvlink_bw": 600*6.0, "alpha_gpu": 4.0, "alpha_cpu": 3.0,
    },
    "HYP-F-IO-A": {
        "fp64": 9.7*4.0, "fp64_tensor": 19.5*4.0, "fp32": 19.5*8.0, "tf32_tensor": 156*8.0, "fp16": 78*4.0, "fp16_tensor": 312*4.0, 
        "mem_bw": 1555*1.5, "pcie_bw": 64*25.0, "nvlink_bw": 600*6.0, "alpha_gpu": 4.0, "alpha_cpu": 3.0,
    }
}


def get_gpu_specs(gpu_arch, prefix):
    """Get GPU specifications with appropriate prefix."""
    try:
        specs = GPU_SPECS.get(gpu_arch)
        return {f"{prefix}_{key}": value for key, value in specs.items()}
    except KeyError:
        print("GPU architect is not found in GPU SPEC DICT")


def read_metadata_file_json(f):
    with open(f,'r') as file:
        return json.load(file)


def get_runtime_job(metadata):
    runtime = 0
    for entry in metadata['entries']:
        runtime += entry['end_ts'] - entry['start_ts']
    return runtime // 1000


# Define a custom argument type for a list of strings
def list_of_strings(arg):
    return arg.split(',')

def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def preprocess_df(df):
    df.rename( 
        columns={
            'nersc_ldms_dcgm_gr_engine_active':'GRACT',
            'nersc_ldms_dcgm_fp16_active':'FP16A',
            'nersc_ldms_dcgm_fp32_active':'FP32A',
            'nersc_ldms_dcgm_fp64_active':'FP64A',
            'nersc_ldms_dcgm_tensor_active':'TENSA',
            'nersc_ldms_dcgm_nvlink_rx_bytes':'NVRX',
            'nersc_ldms_dcgm_nvlink_tx_bytes':'NVTX',
            'nersc_ldms_dcgm_pcie_rx_bytes':'PCRX',
            'nersc_ldms_dcgm_pcie_tx_bytes':'PCTX',
            'nersc_ldms_dcgm_dram_active':'DRAMA',
            'nersc_ldms_dcgm_tensor_hmma_active':'TENSA_HMMA'
        },
        inplace=True 
    )
    
    df['timestamp_s'] = df['timestamp'] // 1000
    df['timestamp_10s'] = (df['timestamp_s'] // 10) * 10
    df.drop(['timestamp'], axis=1, inplace=True, errors='ignore')
    df.drop(['nersc_ldms_dcgm_power_usage','nersc_ldms_dcgm_total_energy_consumption'], axis=1, inplace=True, errors='ignore')
    
    filtered_df = df.loc[
            (df['FP64A'].between(0, 1)) & 
            (df['FP32A'].between(0, 1)) & 
            (df['FP16A'].between(0, 1)) & 
            (df['TENSA'].between(0, 1)) &
            (df['DRAMA'].between(0, 1)) & 
            (df['GRACT'].between(0,1))  &
            (df['TENSA_HMMA'].between(0,1)) & 
            (df['TENSA_HMMA'] <= df['TENSA'])
        ].copy()
    
    filtered_df['TENSA'] = np.where(filtered_df['TENSA_HMMA'] <= 0.1, filtered_df['TENSA'] - filtered_df['TENSA_HMMA'], filtered_df['TENSA'])
    filtered_df['PCRXTX'] = filtered_df['PCRX'] + filtered_df['PCTX']
    filtered_df['NVRXTX'] = filtered_df['NVRX'] + filtered_df['NVTX']
    filtered_df['RXTX'] = filtered_df['NVRXTX'] + filtered_df['PCRXTX']
    
    return filtered_df


def model_time_per_job(df, job_total_runtime, ref_gpu_arch, tgt_gpu_arch, sampling_intv):
    df = preprocess_df(df)

    # Get specifications for both reference and target GPUs
    ref_gpu_spec = get_gpu_specs(ref_gpu_arch, "ref")
    target_gpu_spec = get_gpu_specs(tgt_gpu_arch, "target")

    
    '''
        Modeling Performance on Reference Hardwre
    '''
    df["FLOPS_ref"] = df['FP64A'] + df['FP32A'] + df['FP16A'] + df['TENSA']
    df["bound_type_ref"] = np.where(df['FLOPS_ref'] > df['DRAMA'], 'cb', 'mb')
    df["t_roofline_overlap_ref"] = np.maximum(df['FLOPS_ref'], df['DRAMA'])
    df["t_roofline_sequential_ref"] = np.minimum(1.0, df['FLOPS_ref'] + df['DRAMA'])
    df["t_otherGPU_overlap_ref"] = np.maximum(0, df['GRACT'] - df['t_roofline_overlap_ref'])
    df["t_otherGPU_sequential_ref"] = np.maximum(0, df["GRACT"] - df["t_roofline_sequential_ref"])
    df["t_PCIE_ref"] = df["PCRXTX"] / 1e9 * ref_gpu_spec["ref_pcie_bw"]
    df["t_NVLINK_ref"] = df["NVRXTX"] / 1e9 * ref_gpu_spec["ref_nvlink_bw"]

    
    '''
        Predicting Performance on Target Hardwre
    '''
    df["FP64A_tgt"] = df["FP64A"] * ref_gpu_spec["ref_fp64"] / target_gpu_spec["target_fp64"]
    df["FP32A_tgt"] = df["FP32A"] * ref_gpu_spec["ref_fp32"] / target_gpu_spec["target_fp32"]
    df["FP16A_tgt"] = df["FP16A"] * ref_gpu_spec["ref_fp16"] / target_gpu_spec["target_fp16"]
    # Assume most jobs use double precision
    df["TENSA_tgt"] = df["TENSA"] * ref_gpu_spec["ref_fp64_tensor"] / target_gpu_spec["target_fp64_tensor"]
    df["DRAMA_tgt"] = df["DRAMA"] * ref_gpu_spec["ref_mem_bw"] / target_gpu_spec["target_mem_bw"]

    df["FLOPS_tgt"] = df["FP64A_tgt"] + df["FP32A_tgt"] + df["FP16A_tgt"] + df["TENSA_tgt"]
    df["bound_type_tgt"] = np.where(df["FLOPS_tgt"] > df["DRAMA_tgt"], 'cb', 'mb')
    df["t_roofline_overlap_tgt"] = np.maximum(df["FLOPS_tgt"], df["DRAMA_tgt"])
    df["t_roofline_sequential_tgt"] = np.minimum(1.0, df["FLOPS_tgt"] + df["DRAMA_tgt"])
    df["t_otherGPU_overlap_tgt"] = np.maximum(0, df['GRACT'] - df['t_roofline_overlap_tgt'])
    df["t_otherGPU_sequential_tgt"] = np.maximum(0, df['GRACT'] - df['t_roofline_sequential_tgt'])

    df["t_PCIE_tgt"] = (df["t_PCIE_ref"]) * ref_gpu_spec["ref_pcie_bw"] / target_gpu_spec["target_pcie_bw"]
    df["t_NVLINK_tgt"] = (df["t_NVLINK_ref"]) * ref_gpu_spec["ref_nvlink_bw"] / target_gpu_spec["target_nvlink_bw"]
    df["t_otherNode_tgt"] = np.maximum(0, 1 - df['GRACT'] - df['t_PCIE_tgt'] - df['t_NVLINK_tgt'])

    #calculate per timestamp syncing across GPUs
    cols_to_sync=["t_roofline_overlap_ref",
                  "t_roofline_sequential_ref",
                  "t_PCIE_ref",
                  "t_NVLINK_ref",
                  "t_roofline_overlap_tgt",
                  "t_roofline_sequential_tgt",
                  "t_PCIE_tgt",
                  "t_NVLINK_tgt"]
    df_synced = df.groupby("timestamp_10s")[cols_to_sync].max().reset_index()
    
    df_synced = df_synced * sampling_intv
    
    roofline_overlap_ref = df_synced["t_roofline_overlap_ref"].sum()
    roofline_sequential_ref = df_synced["t_roofline_sequential_ref"].sum()
    roofline_overlap_tgt = df_synced["t_roofline_overlap_tgt"].sum()
    roofline_sequential_tgt = df_synced["t_roofline_sequential_tgt"].sum()

    pcie_ref = df_synced["t_PCIE_ref"].sum()
    pcie_tgt = df_synced["t_PCIE_tgt"].sum()
    nvlink_ref = df_synced["t_NVLINK_ref"].sum()
    nvlink_tgt = df_synced["t_NVLINK_tgt"].sum()
    count_mb_ref = (df["bound_type_ref"] == "mb").sum()
    count_cb_ref = (df["bound_type_ref"] == "cb").sum()
    count_mb_tgt = (df["bound_type_tgt"] == "mb").sum()
    count_cb_tgt = (df["bound_type_tgt"] == "cb").sum()
    flip_count = (df['bound_type_ref'] != df['bound_type_tgt']).sum()

    count_mb_ref_percent = 100.0 * count_mb_ref / (count_mb_ref + count_cb_ref) if (count_mb_ref + count_cb_ref) > 0.0 else 0.0
    count_mb_tgt_percent = 100.0 * count_mb_tgt / (count_mb_tgt + count_cb_tgt) if (count_mb_tgt + count_cb_tgt) > 0.0 else 0.0
    flip_percent= 100.0 * flip_count / (count_mb_ref + count_cb_ref) if (count_mb_ref + count_cb_ref) > 0.0 else 0.0 
    roofline_overlap_ref_runtime_percent = 100.0 * roofline_overlap_ref / job_total_runtime if job_total_runtime > 0.0 else 0.0
    roofline_sequential_ref_runtime_percent = 100.0 * roofline_sequential_ref / job_total_runtime if job_total_runtime > 0.0 else 0.0

    roofline_overlap_speedup = roofline_overlap_ref / roofline_overlap_tgt if roofline_overlap_tgt > 0.0 else 0
    roofline_sequential_speedup = roofline_sequential_ref / roofline_sequential_tgt if roofline_sequential_tgt > 0.0 else 0

    if roofline_overlap_speedup > 0.0 and roofline_sequential_speedup > 0.0:
        total_other_time_overlap = job_total_runtime - (roofline_overlap_ref + pcie_ref + nvlink_ref)
        total_other_time_sequential = job_total_runtime - (roofline_sequential_ref + pcie_ref + nvlink_ref)

        job_total_overlap_runtime_tgt = roofline_overlap_tgt + pcie_tgt + nvlink_tgt + total_other_time_overlap
        job_total_sequential_runtime_tgt = roofline_sequential_tgt + pcie_tgt + nvlink_tgt + total_other_time_sequential

        total_overlap_speedup = job_total_runtime / job_total_overlap_runtime_tgt if job_total_overlap_runtime_tgt > 0.0 else 0.0
        total_sequential_speedup = job_total_runtime / job_total_sequential_runtime_tgt if job_total_sequential_runtime_tgt > 0.0 else 0.0
    else:
        total_overlap_speedup = 0.0
        total_sequential_speedup = 0.0

    time_distributions = {
            f'total_measured_runtime':job_total_runtime,
            f't_roofline_overlap_ref_{ref_gpu_arch}':roofline_overlap_ref,
            f't_roofline_overlap_tgt_{tgt_gpu_arch}':roofline_overlap_tgt,
            f't_roofline_sequential_ref_{ref_gpu_arch}':roofline_sequential_ref,
            f't_roofline_sequential_tgt_{tgt_gpu_arch}':roofline_sequential_tgt,
            f't_pcie_ref_{ref_gpu_arch}':pcie_ref,
            f't_pcie_tgt_{tgt_gpu_arch}':pcie_tgt,
            f't_nvlink_ref_{ref_gpu_arch}':nvlink_ref,
            f't_nvlink_tgt_{tgt_gpu_arch}':nvlink_tgt,
            f'count_mb_ref_{ref_gpu_arch}':count_mb_ref,
            f'count_cb_ref_{ref_gpu_arch}':count_cb_ref,
            f'count_mb_ref_percent_{ref_gpu_arch}':count_mb_ref_percent,
            f'count_mb_tgt_{tgt_gpu_arch}':count_mb_tgt,
            f'count_cb_tgt_{tgt_gpu_arch}':count_cb_tgt,
            f'count_mb_tgt_percent_{tgt_gpu_arch}':count_mb_tgt_percent,
            f'flip_count_{ref_gpu_arch}_{tgt_gpu_arch}':flip_count,
            f'flip_percent_{ref_gpu_arch}_{tgt_gpu_arch}': flip_percent,
            f'roofline_overlap_ref_percent_runtime_{ref_gpu_arch}':roofline_overlap_ref_runtime_percent,
            f'roofline_sequential_ref_percent_runtime_{ref_gpu_arch}':roofline_sequential_ref_runtime_percent,
            f'roofline_overlap_speedup_{ref_gpu_arch}_{tgt_gpu_arch}':roofline_overlap_speedup,
            f'roofline_sequential_speedup_{ref_gpu_arch}_{tgt_gpu_arch}':roofline_sequential_speedup,
            f'total_overlap_speedup_{ref_gpu_arch}_{tgt_gpu_arch}':total_overlap_speedup,
            f'total_sequential_speedup_{ref_gpu_arch}_{tgt_gpu_arch}':total_sequential_speedup
        }

    return time_distributions


def plot_percentage_histogram(df, column, output_name, bins=10, decimal=False, color='skyblue'):
    data = df[column]
    if decimal:
        data = data * 100
    (n, bins, patches) = plt.hist(data, bins=bins, range=(0, 100), alpha=0.7, color=color)
    plt.xlabel(f'{column} (%)')
    plt.ylabel('Number of Jobs')
    plt.xticks(range(0, 101, 10))
    plt.tight_layout()

    plt.savefig(output_name + '.png', 
                dpi=400,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                format='png',
                metadata={'Creator': 'Matplotlib'})


def plot_multiple_histograms_cumsum(df, columns, output_name, xlabel=None, bin_width=10, max_x_value=100, decimal=False, colors=None, labels=None):
    bins = np.arange(0, max_x_value + bin_width, bin_width)
    returns = []

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()  # Right Y-axis for cumulative percentage

    for i, col in enumerate(columns):
        data = df[col].dropna()
        if decimal:
            data = data * 100

        color = colors[i] if colors and i < len(colors) else None
        label = labels[i] if labels and i < len(labels) else col

        # Plot histogram
        n, bins, patches = ax1.hist(data, bins=bins, alpha=0.5, color=color,label=label, edgecolor='black')
        returns.append((n, bins, patches))
        
        # Compute cumulative percentage
        cumulative = np.cumsum(n)
        cumulative_pct = 100 * cumulative / cumulative[-1]
        bin_centers = (bins[:-1] + bins[1:]) / 2 #(leftedge + rightedge)/2

        ax2.plot(bin_centers, cumulative_pct, linestyle='--', marker='o',color=color, label=f'{label} Cumulative %')

    if xlabel is None:
        ax1.set_xlabel('Value (%)' if decimal else 'Value')
    else:
        ax1.set_xlabel(xlabel)

    ax1.set_ylabel('Number of Jobs')
    ax2.set_ylabel('Number of Jobs Cumulative Percentage (%)')
    ax2.set_ylim(0, 105)

    if decimal:
        ax1.set_xticks(range(0, 101, 10))

    plt.xlabel('count_memory_bound_percent (%)')
    # Merge legends
    lines_labels_1 = ax1.get_legend_handles_labels()
    lines_labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(*[sum(lol, []) for lol in zip(lines_labels_1, lines_labels_2)], loc='best')
    
    plt.tight_layout()
    plt.savefig(output_name + '.png', 
                dpi=400,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                format='png',
                metadata={'Creator': 'Matplotlib'})


def main():
    ###################################
    # get all parameters
    ###################################
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-s', '--sample_interval_second', action='store', type=int, required=True,
                        help='indicate the sample interval in milliseconds')
    parser.add_argument('-rg', '--ref_gpu_architect', action='store', type=str, required=True, 
                        choices=['A100-40', 'A100-80'], help='indicate the reference gpu architecture')
    parser.add_argument('-tg', '--target_gpu_architect', action='store', type=str, default=None, 
                        choices=['A100-40', 'A100-80', 'A40', 'H100', 'R100', 'R100-UNI', 
                                 'GPU-M-IO-A-H14', 'GPU-F-IO-A-H14', 'GPU-M-IO-A-H22', 'GPU-F-IO-A-H22', 'GPU-M-IO-A-H24', 'GPU-F-IO-A-H24'], 
                        help='indicate the target gpu architecture')
    parser.add_argument('--job_paths', type=list_of_strings, required=True, help='List of job_paths')
    
    
    args = parser.parse_args()

    sampling_intv = args.overall_runtime_second
    ref_gpu_arch = args.ref_gpu_architect
    tgt_gpu_arch = args.target_gpu_architect
    jobwise_data_paths = args.job_paths
    
    summaries=dict()
    
    for path in jobwise_data_paths:
        pattern = os.path.join(path, "*.pq")
        pq_file_list = glob.glob(pattern)
        print(f"{path} has {len(pq_file_list)} parquet files")

        for pq_file in pq_file_list:
            jobid, userid = Path(pq_file).stem.rsplit('_', 1)
            jobid_userid = jobid + "_" + userid
            metadatafile = path + "/" + jobid_userid + ".json"

            metadata = read_metadata_file_json(metadatafile)

            pq_df = pd.read_parquet(pq_file, engine='pyarrow')

            time_distributions = model_time_per_job(pq_df, get_runtime_job(metadata), ref_gpu_arch, tgt_gpu_arch, sampling_intv)

            summaries[jobid_userid]=time_distributions
        
        summary_df = pd.DataFrame.from_dict(summaries, orient='index').reset_index()
        summary_df = summary_df.rename(columns={'index': 'jobid'})
        # summary_df.to_parquet(output_file)

        plot_percentage_histogram(summary_df, column=f"count_mb_ref_percent_{ref_gpu_arch}", output_name="count_mb_ref_percent")
        
        plot_multiple_histograms_cumsum(summary_df, 
                                        columns=[f"count_mb_ref_percent_{ref_gpu_arch}", f"count_mb_tgt_percent_{tgt_gpu_arch}"], 
                                        output_name="count_mb_percent",
                                        xlabel="count_mb_percent", 
                                        bin_width = 10, max_x_value=100)

        plot_multiple_histograms_cumsum(summary_df,
                                        columns=[f"flip_percent_{ref_gpu_arch}_{tgt_gpu_arch}"],
                                        output_name="samples in job flipped", xlabel="samples in job flipped (%)", bin_width = 1, max_x_value=100)

        plot_multiple_histograms_cumsum(summary_df,
                                        columns=[f"roofline_overlap_ref_percent_runtime_{ref_gpu_arch}", f"roofline_sequential_ref_percent_runtime_{ref_gpu_arch}"],
                                        output_name="roofline_ref_percent_runtime", xlabel="roofline_ref_percent_runtime (%)",
                                        bin_width = 10, max_x_value=100)

        plot_multiple_histograms_cumsum(summary_df,
                                        columns=[f"roofline_overlap_speedup_{ref_gpu_arch}_{tgt_gpu_arch}",f"roofline_sequential_speedup_{ref_gpu_arch}_{tgt_gpu_arch}"],
                                        output_name="roofline_speedup_ratio", xlabel="roofline_speedup_ratio",
                                        bin_width = 0.1, max_x_value=4)

if __name__=="__main__":
    main()