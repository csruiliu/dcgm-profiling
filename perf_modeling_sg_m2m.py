import pandas as pd
import argparse
import re
import numpy as np


# Define a custom argument type for a list of strings
def list_of_strings(arg):
    return arg.split(',')

def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# Function to read the file and process the data
def process_file(file_path, metric_names):
    # List to store data for the single GPU
    gpu_data = list()

    gpu_pattern = re.compile(rf'^GPU 0\s')

    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        if gpu_pattern.match(line):
            # Split the line by three or more spaces
            values = re.split(r'\s{3,}', line.strip())

            # Only get number values, so "GPU x" will be discarded
            numeric_values = [float(value) for value in values if is_number(value)]

            if len(numeric_values) == len(metric_names):
                gpu_data.append(numeric_values)
            else:
                raise ValueError(f"The number of data columns doesn't match the number of metric names")
        
    gpu_dfs = pd.DataFrame(gpu_data, columns=metric_names)

    # returns a single DataFrame
    return gpu_dfs  


def perf_modeling(profiled_df, metrics, overall_runtime_ms, sample_interval_ms, sleep_period_ms, sleep_marks_list, gpu_arch):
    sample_intv = sample_interval_ms / 1000
    
    if gpu_arch == 'A100-40' or gpu_arch == 'A100-80':
        hw_pcie_gb = 64
        hw_nvlink_gb = 600
    elif gpu_arch == 'A40':
        hw_pcie_gb = 64
        hw_nvlink_gb = 112.5
    else: 
        # this case is H100
        hw_pcie_gb = 128
        hw_nvlink_gb = 900

    t_total_list = list()
    
    for row in profiled_df.itertuples(index=False, name='MetricRow'):
        # row is a namedtuple, you can access columns via row.<colname>
        # For example, if your metric_names are ["GPUTL", "SMACT", "TENSO"]
        # you can access row.GPUTL, row.SMACT, row.TENSO, etc.
        metric_values = list(getattr(row, metric) for metric in metrics)
        
        t_flop = sample_intv * (metric_values[metrics.index('TENSO')] + metric_values[metrics.index('FP64A')] + 
                               metric_values[metrics.index('FP32A')] + metric_values[metrics.index('FP16A')])  
        t_dram = sample_intv * metric_values[metrics.index('DRAMA')]
        
        t_roofline = max(t_flop, t_dram)
        
        t_otherGPU = max(0, sample_intv * metric_values[metrics.index('GRACT')] - t_roofline)

        t_pcie = (metric_values[metrics.index('PCITX')] + metric_values[metrics.index('PCIRX')]) * sample_intv / (1000 * 1000 * 1000 * hw_pcie_gb) 

        t_nvlink = (metric_values[metrics.index('NVLTX')] + metric_values[metrics.index('NVLRX')]) * sample_intv / (1000 * 1000 * 1000 * hw_nvlink_gb)

        t_otherNode = max(0, sample_intv * (1 - metric_values[metrics.index('GRACT')]) - t_pcie - t_nvlink)

        t_total = t_roofline + t_otherGPU + t_pcie + t_nvlink + t_otherNode

        t_total_list.append(t_total)

    # get the sleep and finish index according to the actual sleep time
    sleep_idx_list = [int(x / sample_interval_ms) for x in sleep_marks_list]
    finish_idx = int(overall_runtime_ms / sample_interval_ms)

    if finish_idx < len(t_total_list):
        t_total_list_finish = t_total_list[:finish_idx]
    else:
        t_total_list_finish = t_total_list

    start_mark = 0

    time_sum_segments = list()

    for sleep_mark in sleep_idx_list:
        if sleep_mark <= len(t_total_list_finish):
            segment_sum = sum(t_total_list_finish[start_mark:sleep_mark])
            # First segment: keep original sum, others: subtract sleep time
            if len(time_sum_segments) == 0:
                time_sum_segments.append(segment_sum)
            else:
                time_sum_segments.append(segment_sum - sleep_period_ms / 1000)
            start_mark = sleep_mark

    # Add remaining elements
    if start_mark < len(t_total_list_finish):
        remaining_sum = sum(t_total_list_finish[start_mark:])
        if len(time_sum_segments) == 0:  # If this is the first (and only) segment
            time_sum_segments.append(remaining_sum)
        else:
            time_sum_segments.append(remaining_sum - sleep_period_ms / 1000)
    
    time_sum_segments_final = [0 if x < 0 else x for x in time_sum_segments]
    print(time_sum_segments_final)


def perf_predict(gpu_dfs, metrics, overall_runtime_ms_ref, sample_interval_ms, ref_gpu_arch, target_gpu_arch):
    sample_intv = sample_interval_ms / 1000
    
    # I got the numbers from nvidia official website and https://www.techpowerup.com/gpu-specs
    GPU_SPECS = {
        "A100-40": {
            "fp64": 9.7, "fp64_tensor": 19.5, "fp32": 19.5, "fp32_tensor": 156,
            "fp16": 78, "fp16_tensor": 312, "mem_bw": 1555, "pcie_bw": 64, "nvlink_bw": 600
        },
        "A100-80": {
            "fp64": 9.7, "fp64_tensor": 19.5, "fp32": 19.5, "fp32_tensor": 156,
            "fp16": 78, "fp16_tensor": 312, "mem_bw": 1935, "pcie_bw": 64, "nvlink_bw": 600
        },
        "A40": {
            "fp64": 0.58, "fp64_tensor": 0, "fp32": 37.4, "fp32_tensor": 74.8,
            "fp16": 37.4, "fp16_tensor": 149.7, "mem_bw": 696, "pcie_bw": 64, "nvlink_bw": 112.5
        },
        "H100": {  # H100 SXM (default)
            "fp64": 34, "fp64_tensor": 67, "fp32": 67, "fp32_tensor": 494.7,
            "fp16": 133.8, "fp16_tensor": 989.4, "mem_bw": 3350, "pcie_bw": 128, "nvlink_bw": 900
        }
    }

    def get_gpu_specs(gpu_arch, prefix):
        """Get GPU specifications with appropriate prefix."""
        try:
            specs = GPU_SPECS.get(gpu_arch)
            return {f"{prefix}_{key}": value for key, value in specs.items()}
        except KeyError:
            print("GPU architect is not found in GPU SPEC DICT")
        
    # Get specifications for both reference and target GPUs
    ref_gpu_spec = get_gpu_specs(ref_gpu_arch, "ref")
    target_gpu_spec = get_gpu_specs(target_gpu_arch, "target")

    # Taking FP64 tensor as general tensor
    ref_tensor = ref_gpu_spec["ref_fp64_tensor"]
    target_tensor = target_gpu_spec["target_fp64_tensor"]
    
    t_total_target_list = list()
    
    for row in gpu_dfs.itertuples(index=False, name='MetricRow'):
        # row is a namedtuple, you can access columns via row.<colname>
        # For example, if your metric_names are ["GPUTL", "SMACT", "TENSO"]
        # you can access row.GPUTL, row.SMACT, row.TENSO, etc.
        metric_values = list(getattr(row, metric) for metric in metrics)
        
        t_flop_ref = sample_intv * (metric_values[metrics.index('TENSO')] + 
                                    metric_values[metrics.index('FP64A')] + 
                                    metric_values[metrics.index('FP32A')] + 
                                    metric_values[metrics.index('FP16A')])  
        t_dram_ref = sample_intv * metric_values[metrics.index('DRAMA')]
        
        t_roofline_ref = max(t_flop_ref, t_dram_ref)
        
        t_otherGPU_ref = max(0, sample_intv * metric_values[metrics.index('GRACT')] - t_roofline_ref)

        t_pcie_ref = (metric_values[metrics.index('PCITX')] + metric_values[metrics.index('PCIRX')]) * sample_intv / (1000 * 1000 * 1000 * ref_gpu_spec["ref_gpu_pcie_bw"]) 

        t_nvlink_ref = (metric_values[metrics.index('NVLTX')] + metric_values[metrics.index('NVLRX')]) * sample_intv / (1000 * 1000 * 1000 * ref_gpu_spec["ref_gpu_nvlink_bw"])

        t_otherNode_ref = max(0, sample_intv * (1 - metric_values[metrics.index('GRACT')]) - t_pcie_ref - t_nvlink_ref)

        t_flop_target = (sample_intv * metric_values[metrics.index('TENSO')] * (ref_gpu_spec["ref_fp64_tensor"] / target_gpu_spec["target_fp64_tensor"]) +
                         sample_intv * metric_values[metrics.index('FP64A')] * (ref_gpu_spec["ref_fp64"] / target_gpu_spec["target_fp64"]) + 
                         sample_intv * metric_values[metrics.index('FP32A')] * (ref_gpu_spec["ref_fp32"] / target_gpu_spec["target_fp32"]) + 
                         sample_intv * metric_values[metrics.index('FP16A')] * (ref_gpu_spec["ref_fp16"] / target_gpu_spec["target_fp16"]))
        
        t_dram_target = sample_intv * metric_values[metrics.index('DRAMA')] * (ref_gpu_spec["ref_gpu_mem_bw"] / target_gpu_spec["target_gpu_mem_bw"])
        
        t_roofline_target = max(t_flop_target, t_dram_target)
        
        t_otherGPU_target = t_otherGPU_ref

        t_pcie_target = t_pcie_ref * (ref_gpu_spec["ref_gpu_pcie_bw"] / target_gpu_spec["target_gpu_pcie_bw"])

        t_nvlink_target = t_nvlink_ref * (ref_gpu_spec["ref_gpu_nvlink_bw"] / target_gpu_spec["target_gpu_nvlink_bw"])

        t_otherNode_target = t_otherNode_ref

        t_total_target = t_roofline_target + t_otherGPU_target + t_pcie_target + t_nvlink_target + t_otherNode_target

        t_total_target_list.append(t_total_target)    

    finish_idx = int(overall_runtime_ms_ref / sample_interval_ms)
    if finish_idx < len(t_total_target_list):
        t_total_target_list_finish = t_total_target_list[:finish_idx]
    else:
        t_total_target_list_finish = t_total_target_list
 
    print(sum(t_total_target_list_finish))


def main():
    ###################################
    # get all parameters
    ###################################
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-f', '--dcgm_file', action='store', type=str, required=True,
                        help='indicate the dcgm output file')
    parser.add_argument('-d', '--sample_interval_ms', action='store', type=int, required=True,
                        help='indicate the sample interval in milliseconds')
    parser.add_argument('-s', '--sleep_period_ms', action='store', type=int, required=True,
                        help='indicate the sleep period during GPU execution in milliseconds')  
    parser.add_argument('-o', '--overall_runtime_ms', action='store', type=int, required=True,
                        help='indicate the timestamp for overall runtime in milliseconds')
    parser.add_argument('-rg', '--ref_gpu_architect', action='store', type=str, required=True, choices=['A100-40', 'A100-80', 'A40', 'H100'],
                        help='indicate the gpu architecture')
    parser.add_argument('-tg', '--target_gpu_architect', action='store', type=str, default=None, choices=['A100-40', 'A100-80', 'A40', 'H100'],
                        help='indicate the gpu architecture')
    parser.add_argument('--sleep_marks', action='store', type=float, nargs='+', required=True,
                        help='indicate the space-separated list of sleep starting time marks in milliseconds')  
    parser.add_argument('--metrics', type=list_of_strings, required=True, 
                        help='List of metrics, basically the not-none col names')
    parser.add_argument('-h', '--help', action='help',
                        help='Example: python3 dcgm_analyze.py -f gpu_util/results/xx.100.out -o ./gpu_util/results -d 100 --metrics GRACT,PCITX,PCIRX')
    args = parser.parse_args()

    dcgm_metric_file = args.dcgm_file
    sample_interval_ms = args.sample_interval_ms
    sleep_period_ms = args.sleep_period_ms
    overall_runtime_ms = args.overall_runtime_ms
    sleep_marks = args.sleep_marks
    metrics = args.metrics
    ref_gpu_arch = args.ref_gpu_architect
    target_gpu_arch = args.target_gpu_architect
    
    profiled_df = process_file(dcgm_metric_file, metrics)
    
    perf_modeling(profiled_df, metrics, overall_runtime_ms, sample_interval_ms, sleep_period_ms, sleep_marks, ref_gpu_arch)
    
    if target_gpu_arch is not None:
        perf_predict(profiled_df, metrics, overall_runtime_ms, sample_interval_ms, ref_gpu_arch, target_gpu_arch)

if __name__=="__main__":
    main()