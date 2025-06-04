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
    
    if gpu_arch == 'A100':
        hw_pcie_gb = 64
        hw_nvlink_gb = 600
    elif gpu_arch == 'A40':
        hw_pcie_gb = 64
        hw_nvlink_gb = 112.5
    else:
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


def perf_predict(gpu_dfs, metrics, sample_interval_ms, sleep_period_ms, sleep_marks_list, gpu_arch, precision):
    sample_intv = sample_interval_ms / 1000
    
    # I got the numbers from nvidia official website and https://www.techpowerup.com/gpu-specs

    # FP64 (Double Precision) [TFLOPS] 
    ref_fp64 = 9.7
    # FP64 Tensor [TFLOPS] 
    ref_fp64_tensor = 19.5
    # FP32 (Single Precision) [TFLOPS]
    ref_fp32 = 19.5
    # FP32 Tensor [TFLOPS], Take TF32
    ref_fp32_tensor = 156
    # FP16 (Half Precision) [TFLOPS]
    ref_fp16 = 78
    # FP16 Tensor [TFLOPS]
    ref_fp16_tensor = 312
    # GPU Memory Bandwidth [GB/s]
    ref_gpu_mem_bw = 1555
    # PCIe Bandwidth (GB/s)
    ref_gpu_pcie_bw = 64
    # NVLINK Bandwidth (GB/s)
    ref_gpu_nvlink_bw = 600

    if gpu_arch == 'A40':
        target_fp64 = 0.58
        target_fp64_tensor = 0
        target_fp32 = 37.4
        target_fp32_tensor = 74.8
        target_fp16 = 37.4
        target_fp16_tensor = 149.7
        target_gpu_mem_bw = 696
        target_gpu_pcie_bw = 64
        target_gpu_nvlink_bw = 112.5
    else:
        # H100 SXM
        target_fp64 = 34
        target_fp64_tensor = 67
        target_fp32 = 67
        target_fp32_tensor = 494.7
        target_fp16 = 133.8
        target_fp16_tensor = 989.4
        target_gpu_mem_bw = 3350
        target_gpu_pcie_bw = 128
        target_gpu_nvlink_bw = 900
    
    # Taking FP64 tensor as general tensor
    ref_tensor = ref_fp64_tensor
    target_tensor = target_fp64_tensor

    t_total_target_list = list()
    
    for row in gpu_dfs.itertuples(index=False, name='MetricRow'):
        # row is a namedtuple, you can access columns via row.<colname>
        # For example, if your metric_names are ["GPUTL", "SMACT", "TENSO"]
        # you can access row.GPUTL, row.SMACT, row.TENSO, etc.
        metric_values = list(getattr(row, metric) for metric in metrics)
        
        t_flop_target = (sample_intv * metric_values[metrics.index('TENSO')] * (ref_tensor / target_tensor) +
                         sample_intv * metric_values[metrics.index('FP64A')] * (ref_fp64 / target_fp64) + 
                         sample_intv * metric_values[metrics.index('FP32A')] * (ref_fp32 / target_fp32) + 
                         sample_intv * metric_values[metrics.index('FP16A')] * (ref_fp16 / target_fp16))
        
        t_dram_target = sample_intv * metric_values[metrics.index('DRAMA')] * (ref_gpu_mem_bw / target_gpu_mem_bw)
        
        t_roofline_target = max(t_flop_target, t_dram_target)
        
        #t_otherGPU_target = 

        #t_pcie_target =  

        #t_nvlink_target = 

        #t_otherNode_target = 

        #t_total_target = t_roofline_target + t_otherGPU_target + t_pcie_target + t_nvlink_target + t_otherNode_target

        #t_total_target_list.append(t_total_target)    

    print(sum(t_total_target_list))


def main():
    ###################################
    # get all parameters
    ###################################
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-f', '--dcgm_file', action='store', type=str, required=True,
                        help='indicate the dcgm output file')
    parser.add_argument('-g', '--gpu_architect', action='store', type=str, required=True, choices=['A100', 'A40', 'H100'],
                        help='indicate the gpu architecture')
    parser.add_argument('-p', '--precision', action='store', type=str, required=True, choices=['D', 'S', 'H'],
                        help='indicate the main precision for computation [D, S, H]')
    parser.add_argument('-d', '--sample_interval_ms', action='store', type=int, required=True,
                        help='indicate the sample interval in milliseconds')
    parser.add_argument('-s', '--sleep_period_ms', action='store', type=int, required=True,
                        help='indicate the sleep period during GPU execution in milliseconds')  
    parser.add_argument('-o', '--overall_runtime_ms', action='store', type=int, required=True,
                        help='indicate the timestamp for overall runtime in milliseconds')
    parser.add_argument('--sleep_marks', action='store', type=float, nargs='+', required=True,
                        help='indicate the space-separated list of sleep starting time marks in milliseconds')  
    parser.add_argument('--metrics', type=list_of_strings, required=True, 
                        help='List of metrics, basically the not-none col names')
    parser.add_argument('-h', '--help', action='help',
                        help='Example: python3 dcgm_analyze.py -f gpu_util/results/xx.100.out -o ./gpu_util/results -d 100 --metrics GRACT,PCITX,PCIRX')
    args = parser.parse_args()

    dcgm_metric_file = args.dcgm_file
    gpu_arch = args.gpu_architect
    precision = args.precision
    sample_interval_ms = args.sample_interval_ms
    sleep_period_ms = args.sleep_period_ms
    overall_runtime_ms = args.overall_runtime_ms
    sleep_marks = args.sleep_marks
    metrics = args.metrics

    profiled_df = process_file(dcgm_metric_file, metrics)
    
    perf_modeling(profiled_df, metrics, overall_runtime_ms, sample_interval_ms, sleep_period_ms, sleep_marks, gpu_arch)
    
    # perf_predict(profiled_df, metrics, sample_interval_ms, sleep_period_ms, sleep_marks, gpu_arch, precision)

if __name__=="__main__":
    main()