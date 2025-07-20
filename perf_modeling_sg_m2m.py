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

    # Read the header to get column names and their positions
    header_columns = None
    metric_indices = None

    gpu_pattern = re.compile(rf'^GPU 0\s')
    header_pattern = re.compile(r'^#Entity')

    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find and parse the header (use the first occurrence)
    for line in lines:
        if header_pattern.match(line):
            # Split the header by multiple spaces to get column names
            header_parts = re.split(r'\s{2,}', line.strip())
            header_columns = [col.strip() for col in header_parts]
            
            # Map requested metrics to their column indices
            metric_indices = []
            for metric in metric_names:
                if metric in header_columns:
                    # Find index (subtract 1 because 'Entity' column will be discarded)
                    metric_indices.append(header_columns.index(metric) - 1)
                else:
                    raise ValueError(f"Metric '{metric}' not found in data file. Available metrics: {header_columns[1:]}")
            break
    
    if header_columns is None:
        raise ValueError("Could not find header line in the data file")
    
    # Process all lines, skipping headers
    for line in lines:
        # Skip header lines
        if header_pattern.match(line):
            continue
            
        # Process GPU data lines
        if gpu_pattern.match(line):
            # Split the line by three or more spaces
            values = re.split(r'\s{3,}', line.strip())

            # Only get number values, so "GPU x" will be discarded
            numeric_values = [float(value) for value in values if is_number(value)]

            if len(numeric_values) >= len(header_columns) - 1:  # -1 for Entity column
                # Extract only the requested metrics in the specified order
                selected_values = [numeric_values[i] for i in metric_indices]
                gpu_data.append(selected_values)
            else:
                print(f"Warning: Line has insufficient data columns: {line.strip()}")
        
    gpu_dfs = pd.DataFrame(gpu_data, columns=metric_names)
    
    # returns a single DataFrame
    return gpu_dfs 
    


def perf_modeling(profiled_df, metrics, overall_runtime_ms, sample_interval_ms, sleep_period_ms, sleep_marks, gpu_arch):
    sample_intv = sample_interval_ms / 1000
    
    if gpu_arch == 'A100-40' or gpu_arch == 'A100-80':
        hw_pcie_gb = 64
        hw_nvlink_gb = 600
    else:
        raise ValueError("Reference GPU arch is not recognized")

    t_total_list = list()
    
    for row in profiled_df.itertuples(index=False):
        # row is a namedtuple, you can access columns via row.<colname>
        # For example, if your metric_names are ["GPUTL", "SMACT", "TENSO"]
        # you can access row.GPUTL, row.SMACT, row.TENSO, etc.
        metric_values = list(getattr(row, metric) for metric in metrics)
        
        t_flop = sample_intv * (metric_values[metrics.index('TENSO')] + metric_values[metrics.index('FP64A')] + 
                               metric_values[metrics.index('FP32A')] + metric_values[metrics.index('FP16A')])  
        t_dram = sample_intv * metric_values[metrics.index('DRAMA')]
        
        t_roofline = max(t_flop, t_dram)
        
        t_otherGPU = max(0, sample_intv * metric_values[metrics.index('GRACT')] - t_roofline)

        t_pcie = (metric_values[metrics.index('PCITX')] + metric_values[metrics.index('PCIRX')]) * sample_intv / (hw_pcie_gb * 1e9) 

        t_nvlink = (metric_values[metrics.index('NVLTX')] + metric_values[metrics.index('NVLRX')]) * sample_intv / (hw_nvlink_gb * 1e9)

        t_otherNode = max(0, sample_intv * (1 - metric_values[metrics.index('GRACT')]) - t_pcie - t_nvlink)

        t_total = t_roofline + t_otherGPU + t_pcie + t_nvlink + t_otherNode

        t_total_list.append(t_total)

    # get the sleep and finish index according to the actual sleep time
    if sleep_period_ms != 0:
        sleep_idx_list = [int(x / sample_interval_ms) for x in sleep_marks]
    
    finish_idx = int(overall_runtime_ms / sample_interval_ms)

    if finish_idx < len(t_total_list):
        t_total_list_finish = t_total_list[:finish_idx]
    else:
        t_total_list_finish = t_total_list

    if sleep_period_ms != 0 and sleep_marks is None:
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
    else:
        print(f"Estimate Runtime On Reference Hardware: {sum(t_total_list_finish):0.2f}")


def check_bound_switch(ref_gpu_spec, target_gpu_spec, t_flop_ref, t_dram_ref):
    balance_ref = ref_gpu_spec['ref_fp64'] * 1000 / ref_gpu_spec['ref_mem_bw']

    balance_target = target_gpu_spec['target_fp64'] * 1000 / target_gpu_spec['target_mem_bw']

    if t_dram_ref != 0:
        t_intensity_balance = t_flop_ref * ref_gpu_spec['ref_fp64'] * 1000 / (t_dram_ref * ref_gpu_spec['ref_mem_bw'])
        
    else:
        t_intensity_balance = float('-inf')
    
    bound_ref = "compute" if t_intensity_balance > balance_ref else "memory"
    bound_target = "compute" if t_intensity_balance > balance_target else "memory"
    
    return bound_ref, bound_target


def perf_predict(gpu_dfs, metrics, overall_runtime_ms_ref, sample_interval_ms, ref_gpu_arch, target_gpu_arch, flop_util_bound_switch, mem_util_bound_switch):
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
        },
        "Rubin": {
            "fp64": 9.7, "fp64_tensor": 19.5, "fp32": 312, "fp32_tensor": 156,
            "fp16": 78, "fp16_tensor": 312, "mem_bw": 1944, "pcie_bw": 64, "nvlink_bw": 600
        },
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
    
    t_total_bursty_target_list = list()
    t_total_interleave_target_list = list()

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

        t_pcie_ref = (metric_values[metrics.index('PCITX')] + metric_values[metrics.index('PCIRX')]) * sample_intv / (1000 * 1000 * 1000 * ref_gpu_spec["ref_pcie_bw"]) 

        t_nvlink_ref = (metric_values[metrics.index('NVLTX')] + metric_values[metrics.index('NVLRX')]) * sample_intv / (1000 * 1000 * 1000 * ref_gpu_spec["ref_nvlink_bw"])

        t_otherNode_ref = max(0, sample_intv * (1 - metric_values[metrics.index('GRACT')]) - t_pcie_ref - t_nvlink_ref)

        bound_ref, bound_target = check_bound_switch(ref_gpu_spec, target_gpu_spec, t_flop_ref, t_dram_ref)

        if bound_ref == bound_target:
            t_flop_target = (sample_intv * metric_values[metrics.index('TENSO')] * (ref_gpu_spec["ref_fp64_tensor"] / target_gpu_spec["target_fp64_tensor"]) +
                            sample_intv * metric_values[metrics.index('FP64A')] * (ref_gpu_spec["ref_fp64"] / target_gpu_spec["target_fp64"]) + 
                            sample_intv * metric_values[metrics.index('FP32A')] * (ref_gpu_spec["ref_fp32"] / target_gpu_spec["target_fp32"]) + 
                            sample_intv * metric_values[metrics.index('FP16A')] * (ref_gpu_spec["ref_fp16"] / target_gpu_spec["target_fp16"]))
            t_dram_target = sample_intv * metric_values[metrics.index('DRAMA')] * (ref_gpu_spec["ref_mem_bw"] / target_gpu_spec["target_mem_bw"])
        
        elif bound_ref != bound_target and bound_target == "memory":
            print("compute-bound switch to memory-bound")
            t_flop_target = (sample_intv * metric_values[metrics.index('TENSO')] * (ref_gpu_spec["ref_fp64_tensor"] / target_gpu_spec["target_fp64_tensor"]) +
                            sample_intv * metric_values[metrics.index('FP64A')] * (ref_gpu_spec["ref_fp64"] / target_gpu_spec["target_fp64"]) + 
                            sample_intv * metric_values[metrics.index('FP32A')] * (ref_gpu_spec["ref_fp32"] / target_gpu_spec["target_fp32"]) + 
                            sample_intv * metric_values[metrics.index('FP16A')] * (ref_gpu_spec["ref_fp16"] / target_gpu_spec["target_fp16"]))
            t_dram_target = sample_intv * metric_values[metrics.index('DRAMA')] * (ref_gpu_spec["ref_mem_bw"] / target_gpu_spec["target_mem_bw"]) * mem_util_bound_switch
        
        elif bound_ref != bound_target and bound_target == "compute":
            print("memory-bound switch to compute-bound")
            t_flop_target = (sample_intv * metric_values[metrics.index('TENSO')] * (ref_gpu_spec["ref_fp64_tensor"] / target_gpu_spec["target_fp64_tensor"]) +
                            sample_intv * metric_values[metrics.index('FP64A')] * (ref_gpu_spec["ref_fp64"] / target_gpu_spec["target_fp64"]) + 
                            sample_intv * metric_values[metrics.index('FP32A')] * (ref_gpu_spec["ref_fp32"] / target_gpu_spec["target_fp32"]) + 
                            sample_intv * metric_values[metrics.index('FP16A')] * (ref_gpu_spec["ref_fp16"] / target_gpu_spec["target_fp16"])) * flop_util_bound_switch
            
            t_dram_target = sample_intv * metric_values[metrics.index('DRAMA')] * (ref_gpu_spec["ref_mem_bw"] / target_gpu_spec["target_mem_bw"])
        
        else:
            raise ValueError("Impossible Error")

        t_roofline_target_interleave = max(t_flop_target, t_dram_target)
        
        t_roofline_target_bursty = t_flop_target + t_dram_target

        t_otherGPU_target = t_otherGPU_ref

        t_pcie_target = t_pcie_ref * (ref_gpu_spec["ref_pcie_bw"] / target_gpu_spec["target_pcie_bw"])

        t_nvlink_target = t_nvlink_ref * (ref_gpu_spec["ref_nvlink_bw"] / target_gpu_spec["target_nvlink_bw"])

        t_otherNode_target = t_otherNode_ref

        t_total_target_bursty = t_roofline_target_bursty + t_otherGPU_target + t_pcie_target + t_nvlink_target + t_otherNode_target

        t_total_target_interleave = t_roofline_target_interleave + t_otherGPU_target + t_pcie_target + t_nvlink_target + t_otherNode_target

        t_total_bursty_target_list.append(t_total_target_bursty)    

        t_total_interleave_target_list.append(t_total_target_interleave)

    finish_idx = int(overall_runtime_ms_ref / sample_interval_ms)
    
    if finish_idx < len(t_total_interleave_target_list):
        t_total_interleave_target_list_finish = t_total_interleave_target_list[:finish_idx]
        t_total_bursty_target_list_finish = t_total_bursty_target_list[:finish_idx]
    else:
        t_total_interleave_target_list_finish = t_total_interleave_target_list
        t_total_bursty_target_list_finish = t_total_bursty_target_list

    print(f"Estimate Runtime On Target Hardware [Interleave Scenario]: {sum(t_total_interleave_target_list_finish):0.2f}")
    print(f"Estimate Runtime On Target Hardware [Bursty Scenario]: {sum(t_total_bursty_target_list_finish):0.2f}")


def main():
    ###################################
    # get all parameters
    ###################################
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-f', '--dcgm_file', action='store', type=str, required=True,
                        help='indicate the dcgm output file')
    parser.add_argument('-d', '--sample_interval_ms', action='store', type=int, required=True,
                        help='indicate the sample interval in milliseconds')
    parser.add_argument('-s', '--sleep_period_ms', action='store', type=int, default=0,
                        help='indicate the sleep period during GPU execution in milliseconds')  
    parser.add_argument('-o', '--overall_runtime_ms', action='store', type=int, required=True,
                        help='indicate the timestamp for overall runtime in milliseconds')
    parser.add_argument('-rg', '--ref_gpu_architect', action='store', type=str, required=True, 
                        choices=['A100-40', 'A100-80'], help='indicate the reference gpu architecture')
    parser.add_argument('-tg', '--target_gpu_architect', action='store', type=str, default=None, 
                        choices=['A100-40', 'A100-80', 'A40', 'H100', 'Rubin'], help='indicate the target gpu architecture')
    parser.add_argument('--sleep_marks', action='store', type=float, nargs='+', default=None,
                        help='indicate the space-separated list of sleep starting time marks in milliseconds')  
    parser.add_argument('--metrics', type=list_of_strings, required=True, 
                        help='List of metrics, basically the not-none col names')
    parser.add_argument('-fu', '--flop_util', action='store', type=float, required=True,
                        help='indicate the estimated flops utlization when bound swtich')
    parser.add_argument('-mu', '--mem_util', action='store', type=float, required=True,
                        help='indicate the estimated memory utlization when bound swtich')
    args = parser.parse_args()

    dcgm_metric_file = args.dcgm_file
    sample_interval_ms = args.sample_interval_ms
    sleep_period_ms = args.sleep_period_ms
    overall_runtime_ms = args.overall_runtime_ms
    sleep_marks = args.sleep_marks
    metrics = args.metrics
    ref_gpu_arch = args.ref_gpu_architect
    target_gpu_arch = args.target_gpu_architect
    flop_util = args.flop_util # such as 0.3
    mem_util = args.mem_util # such as 0.33

    profiled_df = process_file(dcgm_metric_file, metrics)
    
    perf_modeling(profiled_df, metrics, overall_runtime_ms, sample_interval_ms, sleep_period_ms, sleep_marks, ref_gpu_arch)
    
    if target_gpu_arch is not None:
        perf_predict(profiled_df, metrics, overall_runtime_ms, sample_interval_ms, ref_gpu_arch, target_gpu_arch, flop_util, mem_util)


if __name__=="__main__":
    main()