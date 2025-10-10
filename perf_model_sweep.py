import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse
import json
import glob
from dataclasses import dataclass, replace
import matplotlib.pyplot as plt

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


def read_ldmsdata_pq(f):
    data=pd.read_parquet(f,engine='pyarrow')
    return data


def init_devices():
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
    
    M_H14_base = Device(name="M_H14_base", 
                        fp16=H100.fp16, fp32=H100.fp32, fp64=H100.fp64, tf16=H100.tf16, tf32=H100.tf32, tf64=H100.tf64, 
                        membw=H100.membw*4.0, pcie=H100.pcie*4.0,  nvlink=H100.nvlink*4.0,
                        alpha_gpu=1.0, alpha_cpu=1.0)

    F_H14_base = Device(name="F_H14_base",
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


def model_time_per_job(df, job_total_runtime, job_node_hours, ref_gpu: Device, tgt_gpu_list: list, sampling_intv, mad=False):
    df = preprocess_df(df)
    
    # if all time is in otherNode, then this is a spurious sample. Drop all these rows.
    df=df[df['t_otherNode']!=1.0]

    '''
        Modeling Performance on Reference Hardwre
    '''
    
    df['FLOPS_ref'] = df['FP64A'] + df['FP32A'] + df['FP16A'] + df['TENSA_hp'] + df['TENSA_sp'] + df['TENSA_dp']
    df["t_roofline_overlap_ref"] = np.maximum(df['FLOPS_ref'], df['DRAMA'])
    df["t_roofline_sequential_ref"] = np.minimum(1.0, df['FLOPS_ref'] + df['DRAMA'])
    df["t_otherGPU_overlap_ref"] = np.maximum(0, df['GRACT'] - df['t_roofline_overlap_ref'])
    df["t_otherGPU_sequential_ref"] = np.maximum(0, df["GRACT"] - df["t_roofline_sequential_ref"])
    df["t_PCIE_ref"] = df["PCRXTX"] / ref_gpu.pcie
    df["t_NVLINK_ref"] = df["NVRXTX"] / ref_gpu.nvlink
    df['t_otherNode_ref'] = np.maximum(0, 1 - df['GRACT'] - df['t_PCIE_ref'] - df['t_NVLINK_ref'])

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

    summary_dicts_list = list()
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

        time_distributions = {
            'total_measured_runtime':job_total_runtime,
            'total_node_hours_job':job_node_hours,
            f'frac_otherNode_roofline_overlap_{ref_gpu.name}': frac_otherNode_roofline_overlap_ref,
            f'frac_otherNode_roofline_sequential_{ref_gpu.name}': frac_otherNode_roofline_sequential_ref,
            f'frac_otherGPU_overlap_gract_{ref_gpu.name}':frac_otherGPU_overlap_gract_ref,
            f'frac_otherGPU_sequential_gract_{ref_gpu.name}':frac_otherGPU_sequential_gract_ref,
            f'frac_otherGPU_roofline_overlap_{ref_gpu.name}': frac_otherGPU_roofline_overlap_ref,
            f'frac_otherGPU_roofline_sequential_{ref_gpu.name}': frac_otherGPU_roofline_sequential_ref,
            f'speedup_roofline_overlap_sync_{tgt_gpu.name}_{tgt_gpu.alpha_gpu}_{tgt_gpu.alpha_cpu}':speedup_roofline_overlap_sync,
            f'speedup_roofline_sequential_sync_{tgt_gpu.name}_{tgt_gpu.alpha_gpu}_{tgt_gpu.alpha_cpu}':speedup_roofline_sequential_sync,
            f'speedup_roofline_overlap_independent_{tgt_gpu.name}_{tgt_gpu.alpha_gpu}_{tgt_gpu.alpha_cpu}': speedup_roofline_overlap_independent,
            f'speedup_roofline_sequential_independent_{tgt_gpu.name}_{tgt_gpu.alpha_gpu}_{tgt_gpu.alpha_cpu}': speedup_roofline_sequential_independent,
        }
        summary_dicts_list.append(time_distributions)
    combined = {}

    for d in summary_dicts_list:
        combined.update(d)
    
    return combined


def main():
    ###################################
    # get all parameters
    ###################################
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-s', '--sample_interval_second', action='store', type=int, default=10, required=True,
                        help='indicate the sample interval in milliseconds')
    parser.add_argument('-rg', '--ref_gpu_architect', action='store', type=str, required=True, 
                        choices=['A100-40', 'A100-80'], help='indicate the reference gpu architecture')
    parser.add_argument('-tgs', '--target_gpu_architect_list', action='store', type=list_of_strings, default=None, 
                        choices=['A100-40', 'A100-80', 'A40', 'H100', 'R100', 'R100-UNI'], 
                        help='indicate the target gpu architecture')
    parser.add_argument('--job_master_file', action='store', type=str, required=True, help='indicate the job master file')
    parser.add_argument('--job_paths', type=list_of_strings, required=True, help='indicate the list of job_paths')    
    args = parser.parse_args()

    sampling_intv = args.overall_runtime_second
    ref_gpu_arch = args.ref_gpu_architect
    tgt_gpu_arch = args.target_gpu_architect
    jobwise_data_paths = args.job_paths
    job_master_file = args.job_master_file

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


    init_devices()
    
    jobs_master_df=pd.read_csv(job_master_file)


if __name__=="__main__":
    main()