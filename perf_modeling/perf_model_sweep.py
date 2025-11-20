import pandas as pd
import numpy as np
import os
import ast
from pathlib import Path
from collections import Counter
import argparse
import json
import glob
import time
from dataclasses import dataclass, replace
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

from pm_node_config import nodes_80gb_set


@dataclass
class Device:
    name: str
    fp16: float
    fp32: float
    fp64: float
    membw: float
    tf16: float
    tf32: float
    tf64: float
    pcie: float
    nvlink: float
    alpha_gpu: float
    alpha_cpu: float


# Define a custom argument type for a list of strings
def list_of_strings(arg):
    return arg.split(',')


def read_metadata_file_json(f):
    with open(f,'r') as file:
        return json.load(file)


def get_node_hours_job(metadata):
    node_hours=0
    for entry in metadata['entries']:
        node_hours+=(((entry['end_ts']-entry['start_ts'])/1000.0)/3600.0)*len(entry['nodelist'])
    return node_hours


def get_runtime_job(metadata):
    runtime = 0
    for entry in metadata['entries']:
        runtime += entry['end_ts'] - entry['start_ts']
    return runtime // 1000


def get_all_parquet_files(job_data_path):
    """Recursively find all parquet files in job_data_path and its subdirectories"""
    pq_file_list = []
    
    # Method 1: Using pathlib (recommended)
    job_path = Path(job_data_path)
    pq_file_list = [str(f) for f in job_path.rglob("*.pq")]
    
    # Method 2: Using glob with recursive pattern (alternative)
    # pattern = os.path.join(job_data_path, "**", "*.pq")
    # pq_file_list = glob.glob(pattern, recursive=True)
    
    return pq_file_list


def check_device_support(ref_gpu, tgt_gpu_list):
    if ref_gpu in ["A100-40", "A100-80", "H100"]:
        print(f"Reference GPU {ref_gpu} is supported")
    if set(tgt_gpu_list).issubset(["A100-40", "A100-80", "H100", "M_H14_base", "M_H14_base", "R100", "R100_UNI"]):
        print(f"Target GPUs {tgt_gpu_list} are supported")


def init_devices(ref_gpu_str, tgt_gpu_str_list):
    A100_40G = Device(name="A100-40G", 
                      fp16=78.0, fp32=19.5, fp64=9.7, tf16=312.0, tf32=156.0, tf64=19.5, 
                      membw=1.555, pcie=64.0*1e9, nvlink=600*1e9, 
                      alpha_gpu=1.0, alpha_cpu=1.0)
    
    A100_80G = Device(name="A100-80G", 
                      fp16=78.0, fp32=19.5, fp64=9.7, tf16=312.0, tf32=156.0, tf64=19.5, 
                      membw=1.935, pcie=64.0*1e9, nvlink=600*1e9, 
                      alpha_gpu=1.0, alpha_cpu=1.0)

    H100 = Device(name="H100", 
                  fp16=133.8, fp32= 67.0, fp64=34.0, tf16=1979.0, tf32=989.0, tf64=67.0, 
                  membw=3.350, pcie=128.0*1e9, nvlink=900.0*1e9, 
                  alpha_gpu=4.0, alpha_cpu=3.0)
    
    M_H14_base = Device(name="M-H14-base", 
                        fp16=H100.fp16, fp32=H100.fp32, fp64=H100.fp64, tf16=H100.tf16, tf32=H100.tf32, tf64=H100.tf64, 
                        membw=H100.membw*4.0, pcie=H100.pcie*4.0,  nvlink=H100.nvlink*4.0,
                        alpha_gpu=1.0, alpha_cpu=1.0)

    F_H14_base = Device(name="F-H14-base",
                        fp16=H100.fp16*4.0, fp32=H100.fp32*4.0, fp64=H100.fp64*4.0, tf16=H100.tf16*4.0, tf32=H100.tf32*4.0, tf64=H100.tf64*4.0, 
                        membw=H100.membw*1.0, pcie=H100.pcie*4.0,  nvlink=H100.nvlink*4.0,
                        alpha_gpu=1.0, alpha_cpu=1.0)
    
    # create a list of new devices with alpha_gpu_vals
    alpha_gpu_vals = np.arange(1., 11., 1.) 
    gpu_sweep_M = [replace(M_H14_base, alpha_gpu=ag) for ag in alpha_gpu_vals]
    gpu_sweep_F = [replace(F_H14_base, alpha_gpu=ag) for ag in alpha_gpu_vals]
    gpu_sweep = gpu_sweep_M + gpu_sweep_F

    # create a list of new devices with alpha_cpu_vals
    alpha_cpu_vals = np.arange(1., 6., 1.)
    cpu_sweep_M = [replace(M_H14_base, alpha_cpu=ac) for ac in alpha_cpu_vals]
    cpu_sweep_F = [replace(F_H14_base, alpha_cpu=ac) for ac in alpha_cpu_vals]
    cpu_sweep = cpu_sweep_M + cpu_sweep_F

    # Create a dictionary mapping device names to Device objects
    device_map = {
        "A100-40G": A100_40G,
        "A100-40": A100_40G,  # Alias
        "A100-80G": A100_80G,
        "A100-80": A100_80G,  # Alias
        "H100": H100,
        "M-H14-base": M_H14_base,
        "M_H14_base": M_H14_base,  # Alias
        "F-H14-base": F_H14_base,
        "F_H14_base": F_H14_base,  # Alias
    }

    # Get reference GPU
    if ref_gpu_str not in device_map:
        raise ValueError(f"Reference GPU '{ref_gpu_str}' not found. Available devices: {list(device_map.keys())}")
    ref_gpu = device_map[ref_gpu_str]

    # Get target GPU list
    tgt_gpu_list = []
    for tgt_gpu_str in tgt_gpu_str_list:
        if tgt_gpu_str not in device_map:
            raise ValueError(f"Target GPU '{tgt_gpu_str}' not found. Available devices: {list(device_map.keys())}")
        tgt_gpu_list.append(device_map[tgt_gpu_str])

    return ref_gpu, tgt_gpu_list


def get_scale_factors(ref_gpu: Device, tgt_gpu: Device):
    return {
            'fp64': ref_gpu.fp64 / tgt_gpu.fp64,
            'fp32': ref_gpu.fp32 / tgt_gpu.fp32,
            'fp16': ref_gpu.fp16 / tgt_gpu.fp16,
            'tf16': ref_gpu.tf16 / tgt_gpu.tf16,
            'tf32': ref_gpu.tf32 / tgt_gpu.tf32,
            'tf64': ref_gpu.tf64 / tgt_gpu.tf64,
            'membw': ref_gpu.membw / tgt_gpu.membw,
            'pcie': ref_gpu.pcie / tgt_gpu.pcie,
            'nvlink': ref_gpu.nvlink / tgt_gpu.nvlink,
            'alpha_cpu': ref_gpu.alpha_cpu / tgt_gpu.alpha_cpu,
            'alpha_gpu': ref_gpu.alpha_gpu / tgt_gpu.alpha_gpu
        }


def process_job_metadata(jobs_metadata_csv):
    job_metadata_df = pd.read_csv(jobs_metadata_csv)
    job_metadata_qos_filtered_df = job_metadata_df[job_metadata_df["QOS"].isin(["gpu_premium","gpu_regular"])]
    # Build a set of allowed (JobID, User) pairs
    mask = job_metadata_df['nodes'].apply(lambda x: ast.literal_eval(x)[0] not in nodes_80gb_set)
    
    # get non-interactive jobs using a100-40gb
    non_interactive_jobs_40gb = set(
        job_metadata_df.loc[
            (job_metadata_df['QOS'].isin(["gpu_regular", "gpu_premium"])) & mask,
            ['JobID', 'User']
        ].itertuples(index=False, name=None)
    )
    print(f"The number of non-interactive jobs using A100-40GB: {len(non_interactive_jobs_40gb)}")
    
    return non_interactive_jobs_40gb
    

def preprocess_job_data(df, ref_gpu):
    tensa_index_threshold = 0.02
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
    
    filtered_df['TENSA_hp'] = np.where(filtered_df['TENSA_HMMA'] > tensa_index_threshold, filtered_df['TENSA_HMMA'], 0)
    filtered_df['TENSA_non_hp'] = np.maximum(0,filtered_df['TENSA'] - filtered_df['TENSA_HMMA'])

    ts = (ref_gpu.fp64 / ref_gpu.fp32) * (ref_gpu.tf32 / ref_gpu.tf64)
    filtered_df['TENSA_sp'] = np.where((filtered_df['FP64A'] > 0.0) | (filtered_df['FP32A'] > 0.0), filtered_df['TENSA_non_hp'] * filtered_df['FP32A'] / (ts*filtered_df['FP64A'] + filtered_df['FP32A']), 0)
    filtered_df['TENSA_dp'] = filtered_df['TENSA_non_hp'] - filtered_df['TENSA_sp']
   
    filtered_df['PCRXTX'] = filtered_df['PCRX'] + filtered_df['PCTX']
    filtered_df['NVRXTX'] = filtered_df['NVRX'] + filtered_df['NVTX']
    filtered_df['RXTX'] = filtered_df['NVRXTX'] + filtered_df['PCRXTX']
    
    return filtered_df


def model_time_per_job(df, jobid_userid, job_total_runtime, job_node_hours, ref_gpu: Device, tgt_gpu_list: list):
    df = preprocess_job_data(df, ref_gpu)
    
    '''
        Modeling Performance on Reference Hardwre
    '''
    
    df['FLOPS_ref'] = df['FP64A'] + df['FP32A'] + df['FP16A'] + df['TENSA_hp'] + df['TENSA_sp'] + df['TENSA_dp']
    df['t_roofline_overlap_ref'] = np.maximum(df['FLOPS_ref'], df['DRAMA'])
    df['t_roofline_sequential_ref'] = np.minimum(1.0, df['FLOPS_ref'] + df['DRAMA'])
    df['t_otherGPU_overlap_ref'] = np.maximum(0, df['GRACT'] - df['t_roofline_overlap_ref'])
    df['t_otherGPU_sequential_ref'] = np.maximum(0, df['GRACT'] - df['t_roofline_sequential_ref'])
    df['t_PCIE_ref'] = df['PCRXTX'] / ref_gpu.pcie
    df['t_NVLINK_ref'] = df['NVRXTX'] / ref_gpu.nvlink
    df['t_otherNode_ref'] = np.maximum(0, 1 - df['GRACT'] - df['t_PCIE_ref'] - df['t_NVLINK_ref'])

    # if all time is in otherNode, then this is a spurious sample. Drop all these rows.
    df=df[df['t_otherNode_ref']!=1.0]

    df['t_sample_roofline_overlap_ref'] = df[['t_roofline_overlap_ref',
                                              't_otherGPU_overlap_ref',
                                              't_PCIE_ref',
                                              't_NVLINK_ref',
                                              't_otherNode_ref']].sum(axis=1)
    
    df['t_sample_roofline_sequential_ref'] = df[['t_roofline_sequential_ref',
                                                 't_otherGPU_sequential_ref',
                                                 't_PCIE_ref',
                                                 't_NVLINK_ref',
                                                 't_otherNode_ref']].sum(axis=1)

    roofline_cols_ref = ['t_sample_roofline_overlap_ref', 't_sample_roofline_sequential_ref']
    
    # groups rows by the "timestamp_10s" column and max-aggregate on two cols in cols_to_sync_speedup_only_ref
    df_synced_ref = df.groupby('timestamp_10s')[roofline_cols_ref].max().reset_index()

    # Get max value for any timestamp 
    runtime_roofline_overlap_sync_ref = df_synced_ref['t_sample_roofline_overlap_ref'].sum(axis=0)
    runtime_roofline_sequential_sync_ref = df_synced_ref['t_sample_roofline_sequential_ref'].sum(axis=0)

    # Get sum value for any timestamp
    runtime_roofline_overlap_independent_ref = df['t_sample_roofline_overlap_ref'].sum(axis=0)
    runtime_roofline_sequential_independent_ref = df['t_sample_roofline_sequential_ref'].sum(axis=0)

    runtime_otherNode_ref = df['t_otherNode_ref'].sum(axis=0)

    runtime_otherGPU_overlap_ref = df['t_otherGPU_overlap_ref'].sum(axis=0)
    runtime_otherGPU_sequential_ref = df['t_otherGPU_sequential_ref'].sum(axis=0)
    runtime_gract_ref = df['GRACT'].sum(axis=0)

    if runtime_gract_ref != 0:
        frac_otherGPU_overlap_gract_ref = runtime_otherGPU_overlap_ref / runtime_gract_ref
        frac_otherGPU_sequential_gract_ref = runtime_otherGPU_sequential_ref / runtime_gract_ref
    else:
        frac_otherGPU_overlap_gract_ref = -1
        frac_otherGPU_sequential_gract_ref = -1

    if runtime_roofline_overlap_independent_ref != 0:
        frac_otherNode_roofline_overlap_ref = runtime_otherNode_ref / runtime_roofline_overlap_independent_ref
        frac_otherGPU_roofline_overlap_ref = runtime_otherGPU_overlap_ref / runtime_roofline_overlap_independent_ref
    else:
        frac_otherNode_roofline_overlap_ref = -1
        frac_otherGPU_roofline_overlap_ref = -1

    if runtime_roofline_sequential_independent_ref != 0:
        frac_otherNode_roofline_sequential_ref = runtime_otherNode_ref / runtime_roofline_sequential_independent_ref
        frac_otherGPU_roofline_sequential_ref = runtime_otherGPU_sequential_ref / runtime_roofline_sequential_independent_ref
    else:
        frac_otherNode_roofline_sequential_ref = -1
        frac_otherGPU_roofline_sequential_ref = -1

    time_distribution_per_job = dict()
    for tgt_gpu in tgt_gpu_list:
        scales = get_scale_factors(ref_gpu, tgt_gpu)
    
        df['FP64A_tgt'] = df['FP64A'] * scales['fp64']
        df['FP32A_tgt'] = df['FP32A'] * scales['fp32']
        df['FP16A_tgt'] = df['FP16A'] * scales['fp16']

        df['TENSA_hp_tgt'] = df['TENSA_hp'] * scales['tf16']
        df['TENSA_sp_tgt'] = df['TENSA_sp'] * scales['tf32']
        df['TENSA_dp_tgt'] = df['TENSA_dp'] * scales['tf64']
        df['TENSA_tgt'] = df['TENSA_hp_tgt'] + df['TENSA_sp_tgt'] + df['TENSA_dp_tgt']

        df['DRAMA_tgt'] = df['DRAMA'] * scales['membw']    
        df['FLOPS_tgt'] = df['FP64A_tgt'] + df['FP32A_tgt'] + df['FP16A_tgt'] + df['TENSA_tgt']

        df['t_roofline_overlap_tgt'] = np.maximum(df['FLOPS_tgt'], df['DRAMA_tgt'])
        df['t_otherGPU_overlap_tgt'] = df['t_otherGPU_overlap_ref'] * scales['alpha_gpu']

        #this minimum isnt strictly required, since it will always be less than 1 if going to faster systems
        df['t_roofline_sequential_tgt'] = np.minimum(1.0, df['FLOPS_tgt'] + df['DRAMA_tgt']) 
        df['t_otherGPU_sequential_tgt'] = df['t_otherGPU_sequential_ref'] * scales['alpha_gpu']

        df['t_PCIE_tgt'] = df['t_PCIE_ref'] * scales['pcie']
        df['t_NVLINK_tgt'] = df['t_NVLINK_ref'] * scales['nvlink']

        df['t_otherNode_tgt']=df['t_otherNode_ref'] * scales['alpha_cpu'] 

        #sample time on target
        df['t_sample_roofline_overlap_tgt'] = df[['t_roofline_overlap_tgt',
                                                  't_otherGPU_overlap_tgt',
                                                  't_PCIE_tgt',
                                                  't_NVLINK_tgt',
                                                  't_otherNode_tgt']].sum(axis=1)
                                                  
        df['t_sample_roofline_sequential_tgt'] = df[['t_roofline_sequential_tgt',
                                                     't_otherGPU_sequential_tgt',
                                                     't_PCIE_tgt',
                                                     't_NVLINK_tgt',
                                                     't_otherNode_tgt']].sum(axis=1)

        roofline_cols_tgt = ['t_sample_roofline_overlap_tgt', 't_sample_roofline_sequential_tgt']
        df_synced_tgt = df.groupby('timestamp_10s')[roofline_cols_tgt].max().reset_index()

        runtime_roofline_overlap_sync_tgt = df_synced_tgt['t_sample_roofline_overlap_tgt'].sum(axis=0)
        runtime_roofline_sequential_sync_tgt = df_synced_tgt['t_sample_roofline_sequential_tgt'].sum(axis=0)
        runtime_roofline_overlap_independent_tgt = df['t_sample_roofline_overlap_tgt'].sum(axis=0)
        runtime_roofline_sequential_independent_tgt = df['t_sample_roofline_sequential_tgt'].sum(axis=0)

        if runtime_roofline_overlap_sync_tgt > 0.0:
            speedup_roofline_overlap_sync = runtime_roofline_overlap_sync_ref / runtime_roofline_overlap_sync_tgt
        else:
            speedup_roofline_overlap_sync = 0

        if runtime_roofline_sequential_sync_tgt > 0.0:
            speedup_roofline_sequential_sync = runtime_roofline_sequential_sync_ref / runtime_roofline_sequential_sync_tgt
        else:
            speedup_roofline_sequential_sync = 0

        if runtime_roofline_overlap_independent_tgt > 0.0:
            speedup_roofline_overlap_independent = runtime_roofline_overlap_independent_ref / runtime_roofline_overlap_independent_tgt
        else:
            speedup_roofline_overlap_independent = 0

        if runtime_roofline_sequential_independent_tgt > 0.0:
            speedup_roofline_sequential_independent = runtime_roofline_sequential_independent_ref / runtime_roofline_sequential_independent_tgt
        else:
            speedup_roofline_sequential_independent = 0

        # each target gpu will generate a time ditribution dict
        job_time_distribution_tgt_gpu = {
            'jobid_userid': jobid_userid, 
            'total_measured_runtime':job_total_runtime,
            'total_node_hours_job':job_node_hours,
            f'frac_otherNode_roofline_overlap_{ref_gpu.name}': frac_otherNode_roofline_overlap_ref,
            f'frac_otherNode_roofline_sequential_{ref_gpu.name}': frac_otherNode_roofline_sequential_ref,
            f'frac_otherGPU_overlap_gract_{ref_gpu.name}':frac_otherGPU_overlap_gract_ref,
            f'frac_otherGPU_sequential_gract_{ref_gpu.name}':frac_otherGPU_sequential_gract_ref,
            f'frac_otherGPU_roofline_overlap_{ref_gpu.name}': frac_otherGPU_roofline_overlap_ref,
            f'frac_otherGPU_roofline_sequential_{ref_gpu.name}': frac_otherGPU_roofline_sequential_ref,
            'tgt_gpu': tgt_gpu.name,
            f'speedup_roofline_overlap_sync_{tgt_gpu.name}_{tgt_gpu.alpha_gpu}_{tgt_gpu.alpha_cpu}':speedup_roofline_overlap_sync,
            f'speedup_roofline_sequential_sync_{tgt_gpu.name}_{tgt_gpu.alpha_gpu}_{tgt_gpu.alpha_cpu}':speedup_roofline_sequential_sync,
            f'speedup_roofline_overlap_independent_{tgt_gpu.name}_{tgt_gpu.alpha_gpu}_{tgt_gpu.alpha_cpu}': speedup_roofline_overlap_independent,
            f'speedup_roofline_sequential_independent_{tgt_gpu.name}_{tgt_gpu.alpha_gpu}_{tgt_gpu.alpha_cpu}': speedup_roofline_sequential_independent,
        }
        time_distribution_per_job[tgt_gpu.name] = job_time_distribution_tgt_gpu

    return time_distribution_per_job


def process_jobs(file_chunks, chunk_idx, non_interactive_jobs_40gb, ref_gpu, tgt_gpu_list):
    """Worker function to process a subset of parquet files"""
    # print(f"Chunk {chunk_idx} is being process")
    
    local_summaries = dict()

    for pq_file in file_chunks:
        jobid, userid = Path(pq_file).stem.rsplit('_', 1)
        if (jobid, userid) not in non_interactive_jobs_40gb:
            continue
        
        jobid_userid = jobid + "_" + userid
        metadata_file = pq_file.replace(".pq", ".json")
        metadata = read_metadata_file_json(metadata_file)

        pq_df = pd.read_parquet(pq_file, engine='pyarrow')

        time_distribution_per_job = model_time_per_job(pq_df, jobid_userid, get_runtime_job(metadata), get_node_hours_job(metadata), ref_gpu, tgt_gpu_list)

        local_summaries[jobid_userid] = time_distribution_per_job

    # print(f"Chunk {chunk_idx} has been processed")
    return local_summaries


def main():
    ###################################
    # get all parameters
    ###################################
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-rg', '--ref_gpu', action='store', type=str, required=True, 
                        help='indicate the reference gpu architecture')
    parser.add_argument('-tgs', '--target_gpu_list', action='store', type=list_of_strings, required=True, 
                        help='indicate the target gpu architecture')
    parser.add_argument('--job_master_file', action='store', type=str, required=True, 
                        help='indicate the job master file')
    parser.add_argument('--job_path', action='store', type=str, required=True, 
                        help='indicate the job path that consists of various jobs')
    parser.add_argument('--max_workers', action='store', type=int, default=32,
                        help='maximum number of worker processes (defaults to CPU count)')    
    parser.add_argument('--chunk_size', action='store', type=int, default=1,
                        help='the number of chunks that each worker processes, 1 is maximum parallelism, large size has less overhead')
    args = parser.parse_args()

    ref_gpu_str = args.ref_gpu
    tgt_gpu_str_list = args.target_gpu_list
    job_master_file = args.job_master_file
    job_data_path = args.job_path
    max_workers = args.max_workers
    chunk_size = args.chunk_size

    print("==============================")
    print(f"Reference GPU: {ref_gpu_str}")
    print(f"Target GPU List: {tgt_gpu_str_list}")
    print(f"Job Data: {job_data_path}")
    print(f"Job Metadata File: {job_master_file}")

    check_device_support(ref_gpu_str, tgt_gpu_str_list)

    ref_gpu_device, tgt_gpu_device_list = init_devices(ref_gpu_str, tgt_gpu_str_list)

    non_interactive_jobs_40gb = process_job_metadata(job_master_file)
    
    # Get list of parquet files recursively from job_data_path and subdirectories
    pq_file_list = get_all_parquet_files(job_data_path)
    print(f"{job_data_path} has {len(pq_file_list)} parquet files (including subdirectories)")
    
    if len(pq_file_list) == 0:
        print("No parquet files found. Exiting...")
        return

    # Split files into chunks for parallel processing
    file_chunks = [pq_file_list[i:i + chunk_size] for i in range(0, len(pq_file_list), chunk_size)]
    print(f"Processing with {max_workers} workers, each has {chunk_size} files")

    timer_start = time.perf_counter()
    # Sequential Processing   
    # process_jobs(pq_file_list, 1, non_interactive_jobs_40gb, ref_gpu_device, tgt_gpu_device_list)
    
    # Initialize global output organized by target GPU, each key is with an empty dict 
    global_output_by_tgt = {f'{tgt.name}': {} 
                            for tgt in tgt_gpu_device_list}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_jobs, 
                chunk, 
                chunk_idx, 
                non_interactive_jobs_40gb, 
                ref_gpu_device, 
                tgt_gpu_device_list
            )
            for chunk_idx, chunk in enumerate(file_chunks)
        ]

        # Collect results from all processes
        for chunk_idx, future in enumerate(futures):
            try:
                result = future.result()
                # Merge results for each target GPU
                for jobid_userid, time_distribution_per_job in result.items():
                    for tgt_gpu_name, tgt_gpu_time_distribution in time_distribution_per_job.items():
                        # Find matching device to get alpha values for the key
                        matching_device = next((d for d in tgt_gpu_device_list if d.name == tgt_gpu_name), None)
                        if matching_device:
                            # key = f'{matching_device.name}_{matching_device.alpha_gpu}_{matching_device.alpha_cpu}'
                            global_output_by_tgt[matching_device.name][jobid_userid] = tgt_gpu_time_distribution
            except Exception as e:
                print(f"Error processing chunk: {e}")
                traceback.print_exc()
    
    timer_end = time.perf_counter()
    print(f"The process took {timer_end - timer_start:.6f} seconds")

    # Write individual CSV file for each target GPU
    for tgt_gpu_key, job_results in global_output_by_tgt.items():
        if job_results:  # Only write if there are results
            output_file = f"{tgt_gpu_key}_jobwise_dcgm_hmma_may_2025.parquet"
            df_output = pd.DataFrame.from_dict(job_results, orient='index')
            print(df_output)
            df_output.to_parquet(output_file, engine='pyarrow', index=False)
            print(f"Wrote {len(job_results)} job results to {output_file}")
        else:
            print(f"No results for target GPU: {tgt_gpu_key}")

    print("==============================")
    print("All output files generated successfully!")


if __name__=="__main__":
    main()