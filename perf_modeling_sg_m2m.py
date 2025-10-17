import pandas as pd
import argparse
import re
import numpy as np


# I got the numbers from nvidia official website and https://www.techpowerup.com/gpu-specs
GPU_SPECS = {
    "A100-40": {
        "fp64": 9.7, "fp64_tensor": 19.5, "fp32": 19.5, "fp32_tensor": 156, "fp16": 78, "fp16_tensor": 312, 
        "mem_bw": 1555, "pcie_bw": 64, "nvlink_bw": 600, "base_clock": 1065, "boost_clock": 1410, "num_streams": 108
    },
    "A100-80": {
        "fp64": 9.7, "fp64_tensor": 19.5, "fp32": 19.5, "fp32_tensor": 156, "fp16": 78, "fp16_tensor": 312, 
        "mem_bw": 1935, "pcie_bw": 64, "nvlink_bw": 600, "base_clock": 1065, "boost_clock": 1410, "num_streams": 108
    },
    "A40": {
        "fp64": 0.58, "fp64_tensor": 0, "fp32": 37.4, "fp32_tensor": 74.8, "fp16": 37.4, "fp16_tensor": 149.7, 
        "mem_bw": 696, "pcie_bw": 64, "nvlink_bw": 112.5, "base_clock": 1305, "boost_clock": 1740, "num_streams": 84
    },
    "H100-SXM": {
        "fp64": 34, "fp64_tensor": 67, "fp32": 67, "fp32_tensor": 989, "fp16": 133.8, "fp16_tensor": 1979, 
        "mem_bw": 3350, "pcie_bw": 128, "nvlink_bw": 900, "base_clock": 1590, "boost_clock": 1980, "num_streams": 132
    },
    "H100-NVL": {
        "fp64": 30, "fp64_tensor": 60, "fp32": 60, "fp32_tensor": 835, "fp16": 133.8, "fp16_tensor": 1671, 
        "mem_bw": 3900, "pcie_bw": 128, "nvlink_bw": 600, "base_clock": 1080, "boost_clock": 1785, "num_streams": 132
    },
    "R100": {
        "fp64": 9.7*3.0, "fp64_tensor": 19.5*3.0, "fp32": 19.5*6.0, "fp32_tensor": 156*6.0, "fp16": 78*3.0, "fp16_tensor": 312*3.0, 
        "mem_bw": 1555*8.0, "pcie_bw": 64*25.0, "nvlink_bw": 600*6.0, "alpha_gpu": 4.0, "alpha_cpu": 3.0,
    },
    "R100-UNI": {
        "fp64": 9.7*4.0, "fp64_tensor": 19.5*4.0, "fp32": 19.5*8.0, "fp32_tensor": 156*8.0, "fp16": 78*4.0, "fp16_tensor": 312*4.0, 
        "mem_bw": 1555*1.5, "pcie_bw": 64*25.0, "nvlink_bw": 600*6.0, "alpha_gpu": 4.0, "alpha_cpu": 3.0,
    },
    "GPU-M-IO-A-H14": {
        "fp64": 9.7*1.0, "fp64_tensor": 19.5*1.0, "fp32": 19.5*1.0, "fp32_tensor": 156*1.0, "fp16": 78*1.0, "fp16_tensor": 312*1.0, 
        "mem_bw": 1555*4.0, "pcie_bw": 64*4.0, "nvlink_bw": 600*4.0, "alpha_gpu": 1.0, "alpha_cpu": 3.0,
    },
    "GPU-F-IO-A-H14": {
        "fp64": 9.7*4.0, "fp64_tensor": 19.5*4.0, "fp32": 19.5*4.0, "fp32_tensor": 156*4.0, "fp16": 78*4.0, "fp16_tensor": 312*4.0, 
        "mem_bw": 1555*1.0, "pcie_bw": 64*4.0, "nvlink_bw": 600*4.0, "alpha_gpu": 4.0, "alpha_cpu": 3.0,
    },
    "GPU-M-IO-A-H22": {
        "fp64": 9.7*2.0, "fp64_tensor": 19.5*2.0, "fp32": 19.5*2.0, "fp32_tensor": 156*2.0, "fp16": 78*2.0, "fp16_tensor": 312*2.0, 
        "mem_bw": 1555*2.0, "pcie_bw": 64*4.0, "nvlink_bw": 600*4.0, "alpha_gpu": 2.0, "alpha_cpu": 3.0,
    },
    "GPU-F-IO-A-H22": {
        "fp64": 9.7*2.0, "fp64_tensor": 19.5*2.0, "fp32": 19.5*2.0, "fp32_tensor": 156*2.0, "fp16": 78*2.0, "fp16_tensor": 312*2.0, 
        "mem_bw": 1555*2.0, "pcie_bw": 64*4.0, "nvlink_bw": 600*4.0, "alpha_gpu": 2.0, "alpha_cpu": 3.0,
    },
    "GPU-M-IO-A-H24": {
        "fp64": 9.7*2.0, "fp64_tensor": 19.5*2.0, "fp32": 19.5*2.0, "fp32_tensor": 156*2.0, "fp16": 78*2.0, "fp16_tensor": 312*2.0, 
        "mem_bw": 1555*4.0, "pcie_bw": 64*4.0, "nvlink_bw": 600*4.0, "alpha_gpu": 2.0, "alpha_cpu": 3.0,
    },
    "GPU-F-IO-A-H24": {
        "fp64": 9.7*4.0, "fp64_tensor": 19.5*4.0, "fp32": 19.5*4.0, "tf32_tensor": 156*4.0, "fp16": 78*4.0, "fp16_tensor": 312*4.0, 
        "mem_bw": 1555*2.0, "pcie_bw": 64*4.0, "nvlink_bw": 600*4.0, "alpha_gpu": 4.0, "alpha_cpu": 3.0,
    }
}

HOST_SPECS = {
    "Perlmutter": {
        "cores": 64, "threads": 128, "base_clock": 2.45, "boost_clock": 3.5, "mem_bw": 3200
    },
    "Einsteinium": {
        "cores": 56, "threads": 112, "base_clock": 2.0, "boost_clock": 3.8, "mem_bw": 4800
    },
    "Eos": {
        "cores": 56, "threads": 112, "base_clock": 2.0, "boost_clock": 3.8, "mem_bw": 4800
    }
}

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
    'tf64': 'ref_fp64_tensor',
    'tf32': 'ref_fp32_tensor',
    'tf16': 'ref_fp16_tensor'
}

prec_target_mappings = {
    'tf64': 'target_fp64_tensor',
    'tf32': 'target_fp32_tensor',
    'tf16': 'target_fp16_tensor'
}

def get_gpu_specs(gpu_arch, prefix):
    """Get GPU specifications with appropriate prefix."""
    try:
        specs = GPU_SPECS.get(gpu_arch)
        return {f"{prefix}_{key}": value for key, value in specs.items()}
    except KeyError:
        print("GPU architect is not found in GPU SPEC DICT")


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

    gpu_pattern = re.compile(rf'^GPU \d+\s')
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
    

def perf_modeling(profiled_df, metrics, overall_runtime_ms, sample_interval_ms, start_ts, end_ts, gpu_arch, precision):
    sample_intv = sample_interval_ms / 1000
    
    t_flop_ref_list = list()
    t_dram_ref_list = list()
    t_roofline_overlap_ref_list = list()
    t_roofline_sequential_ref_list = list()
    t_otherGPU_overlap_ref_list = list()
    t_otherGPU_sequential_ref_list = list()
    t_otherNode_ref_list = list()
    t_total_overlap_ref_list = list()
    t_total_sequential_ref_list = list()

    ref_gpu_spec = get_gpu_specs(gpu_arch, "ref")

    for row in profiled_df.itertuples(index=False):
        # row is a namedtuple, you can access columns via row.<colname>
        # For example, if your metric_names are ["GPUTL", "SMACT", "TENSO"]
        # you can access row.GPUTL, row.SMACT, row.TENSO, etc.
        metric_values = list(getattr(row, metric) for metric in metrics)
        
        t_flop_ref = sample_intv * (metric_values[metrics.index('TENSO')] + 
                                    metric_values[metrics.index('FP64A')] + 
                                    metric_values[metrics.index('FP32A')] + 
                                    metric_values[metrics.index('FP16A')])  
        t_flop_ref_list.append(t_flop_ref)
        
        t_dram_ref = sample_intv * metric_values[metrics.index('DRAMA')]
        t_dram_ref_list.append(t_dram_ref)

        t_roofline_overlap_ref = max(t_flop_ref, t_dram_ref)
        t_roofline_overlap_ref_list.append(t_roofline_overlap_ref)
        t_roofline_sequential_ref = t_flop_ref + t_dram_ref
        t_roofline_sequential_ref_list.append(t_roofline_sequential_ref)

        t_otherGPU_overlap_ref = max(0, sample_intv * metric_values[metrics.index('GRACT')] - t_roofline_overlap_ref)
        t_otherGPU_overlap_ref_list.append(t_otherGPU_overlap_ref)

        t_otherGPU_sequential_ref = max(0, sample_intv * metric_values[metrics.index('GRACT')] - t_roofline_sequential_ref)
        t_otherGPU_sequential_ref_list.append(t_otherGPU_sequential_ref)

        t_pcie_ref = (metric_values[metrics.index('PCITX')] + metric_values[metrics.index('PCIRX')]) * sample_intv / (ref_gpu_spec["ref_pcie_bw"] * 1e9) 

        t_nvlink_ref = (metric_values[metrics.index('NVLTX')] + metric_values[metrics.index('NVLRX')]) * sample_intv / (ref_gpu_spec["ref_nvlink_bw"] * 1e9)

        t_otherNode_ref = max(0, sample_intv * (1 - metric_values[metrics.index('GRACT')]) - t_pcie_ref - t_nvlink_ref)
        t_otherNode_ref_list.append(t_otherNode_ref)

        t_total_overlap_ref = t_roofline_overlap_ref + t_otherGPU_overlap_ref + t_pcie_ref + t_nvlink_ref + t_otherNode_ref
        t_total_overlap_ref_list.append(t_total_overlap_ref)

        t_total_sequential = t_roofline_sequential_ref + t_otherGPU_sequential_ref + t_pcie_ref + t_nvlink_ref + t_otherNode_ref
        t_total_sequential_ref_list.append(t_total_sequential)

    # Calculate finish index based on overall runtime
    finish_idx = int(overall_runtime_ms / sample_interval_ms)
    
    if finish_idx < len(t_total_overlap_ref_list):
        t_flop_ref_list_finish = t_flop_ref_list[:finish_idx]
        t_dram_ref_list_finish = t_dram_ref_list[:finish_idx]
        t_roofline_overlap_ref_list_finish = t_roofline_overlap_ref_list[:finish_idx]
        t_roofline_sequential_ref_list_finish = t_roofline_sequential_ref_list[:finish_idx]
        t_otherGPU_overlap_ref_list_finish = t_otherGPU_overlap_ref_list[:finish_idx]
        t_otherGPU_sequential_ref_list_finish = t_otherGPU_sequential_ref_list[:finish_idx]
        t_otherNode_ref_list_finish = t_otherNode_ref_list[:finish_idx]
        t_total_overlap_ref_list_finish = t_total_overlap_ref_list[:finish_idx]
        t_total_sequential_ref_list_finish = t_total_sequential_ref_list[:finish_idx]
    else:
        t_flop_ref_list_finish = t_flop_ref_list
        t_dram_ref_list_finish = t_dram_ref_list
        t_roofline_overlap_ref_list_finish = t_roofline_overlap_ref_list
        t_roofline_sequential_ref_list_finish = t_roofline_sequential_ref_list
        t_otherGPU_overlap_ref_list_finish = t_otherGPU_overlap_ref_list
        t_otherGPU_sequential_ref_list_finish = t_otherGPU_sequential_ref_list
        t_otherNode_ref_list_finish = t_otherNode_ref_list
        t_total_overlap_ref_list_finish = t_total_overlap_ref_list
        t_total_sequential_ref_list_finish = t_total_sequential_ref_list

    if start_ts is not None or end_ts is not None:
        start_idx = 0
        end_idx = len(t_total_overlap_ref_list_finish)

        if start_ts is not None:
            start_idx = max(0, int(start_ts / sample_interval_ms))
        
        if end_ts is not None:
            end_idx = min(len(t_total_overlap_ref_list_finish), int(end_ts / sample_interval_ms))
        
        if start_idx < end_idx:
            t_flop_ref_list_slice = t_flop_ref_list[start_idx:end_idx]
            t_dram_ref_list_slice = t_dram_ref_list[start_idx:end_idx]
            t_roofline_overlap_ref_list_slice = t_roofline_overlap_ref_list_finish[start_idx:end_idx]
            t_roofline_sequential_ref_list_slice = t_roofline_sequential_ref_list_finish[start_idx:end_idx]
            t_otherGPU_overlap_ref_list_slice = t_otherGPU_overlap_ref_list_finish[start_idx:end_idx]
            t_otherGPU_sequential_ref_list_slice = t_otherGPU_sequential_ref_list_finish[start_idx:end_idx]
            t_otherNode_ref_list_slice = t_otherNode_ref_list_finish[start_idx:end_idx]
            t_total_overlap_ref_list_slice = t_total_overlap_ref_list_finish[start_idx:end_idx]
            t_total_sequential_ref_list_slice = t_total_sequential_ref_list_finish[start_idx:end_idx]
        else:
            t_flop_ref_list_slice = []
            t_dram_ref_list_slice = []
            t_roofline_overlap_ref_list_slice = []
            t_roofline_sequential_ref_list_slice = []
            t_otherGPU_overlap_ref_list_slice = []
            t_otherGPU_sequential_ref_list_slice = []
            t_otherNode_ref_list_slice = []
            t_total_overlap_ref_list_slice = []
            t_total_sequential_ref_list_slice = []
            raise ValueError("End Timestamp is earlier than Start Timestamp")
        
        flop = np.mean(t_flop_ref_list_slice) / sample_intv * ref_gpu_spec[prec_ref_mappings[precision]]
        dram = np.mean(t_dram_ref_list_slice) / sample_intv * ref_gpu_spec["ref_mem_bw"]
        print("============ Reference Hardware ============")
        print(f"Estimate TFLOPS on Reference Hardware: {flop:0.2f}")
        print(f"Estimate GPU Memory Bandwidth on Reference Hardware: {dram:0.2f}")
        print(f"Estimate Roofline Time On Reference Hardware [Overlap Scenario]: {sum(t_roofline_overlap_ref_list_slice):0.2f}")
        print(f"Estimate Roofline Time On Reference Hardware [Sequential Scenario]: {sum(t_roofline_sequential_ref_list_slice):0.2f}")
        print(f"Estimate otherGPU Time On Reference Hardware [Overlap Scenario]: {sum(t_otherGPU_overlap_ref_list_slice):0.2f}")
        print(f"Estimate otherGPU Time On Reference Hardware [Sequential Scenario]: {sum(t_otherGPU_sequential_ref_list_slice):0.2f}")
        print(f"Estimate otherNode Time On Reference Hardware: {sum(t_otherNode_ref_list_slice):0.2f}")
        print(f"Estimate Total Runtime On Reference Hardware [Overlap Scenario]: {sum(t_total_overlap_ref_list_slice):0.2f}")
        print(f"Estimate Total Runtime On Reference Hardware [Sequential Scenario]: {sum(t_total_sequential_ref_list_slice):0.2f}")
        print("\n")
        return 

    flop = np.mean(t_flop_ref_list_finish) / sample_intv * ref_gpu_spec[prec_ref_mappings[precision]]
    dram = np.mean(t_dram_ref_list_finish) / sample_intv * ref_gpu_spec["ref_mem_bw"]
    print("============ Reference Hardware ============")
    print(f"Estimate TFLOPS on Reference Hardware: {flop:0.2f}")
    print(f"Estimate GPU Memory Bandwidth on Reference Hardware: {dram:0.2f}")
    print(f"Estimate Roofline Time On Reference Hardware [Overlap Scenario]: {sum(t_roofline_overlap_ref_list_finish):0.2f}")
    print(f"Estimate Roofline Time On Reference Hardware [Sequential Scenario]: {sum(t_roofline_sequential_ref_list_finish):0.2f}")
    print(f"Estimate otherGPU Time On Reference Hardware [Overlap Scenario]: {sum(t_otherGPU_overlap_ref_list_finish):0.2f}")
    print(f"Estimate otherGPU Time On Reference Hardware [Sequential Scenario]: {sum(t_otherGPU_sequential_ref_list_finish):0.2f}")
    print(f"Estimate otherNode Time On Reference Hardware: {sum(t_otherNode_ref_list_finish):0.2f}")
    print(f"Estimate Total Runtime On Reference Hardware [Overlap Scenario]: {sum(t_total_overlap_ref_list_finish):0.2f}")
    print(f"Estimate Total Runtime On Reference Hardware [Sequential Scenario]: {sum(t_total_sequential_ref_list_finish):0.2f}")
    print("\n")
    return


def check_bound_switch(ref_gpu_spec, target_gpu_spec, t_flop_ref, t_dram_ref, t_flop_target, t_dram_target):
    balance_ref = ref_gpu_spec['ref_fp64'] * 1000 / ref_gpu_spec['ref_mem_bw']

    balance_target = target_gpu_spec['target_fp64'] * 1000 / target_gpu_spec['target_mem_bw']

    if t_dram_ref != 0:
        t_intensity_ref = t_flop_ref * ref_gpu_spec['ref_fp64'] * 1000 / (t_dram_ref * ref_gpu_spec['ref_mem_bw'])
    else:
        t_intensity_ref = float('-inf')

    if t_dram_target != 0:
        t_intensity_target = t_flop_target * target_gpu_spec['target_fp64'] * 1000 / (t_dram_target * target_gpu_spec['target_mem_bw'])
    else:
        t_intensity_target = float('-inf')
    
    bound_ref = "compute" if t_intensity_ref > balance_ref else "memory"
    bound_target = "compute" if t_intensity_target > balance_target else "memory"
    
    return bound_ref, bound_target


def perf_predict(gpu_dfs, metrics, overall_runtime_ms_ref, sample_interval_ms, start_ts, end_ts, ref_gpu_arch, target_gpu_arch, precision, flop_util_bound_switch, mem_util_bound_switch):
    sample_intv = sample_interval_ms / 1000

    # Get specifications for both reference and target GPUs
    ref_gpu_spec = get_gpu_specs(ref_gpu_arch, "ref")
    target_gpu_spec = get_gpu_specs(target_gpu_arch, "target")
    
    t_roofline_overlap_target_list = list()
    t_roofline_sequential_target_list = list()
    t_otherGPU_overlap_target_list = list()
    t_otherGPU_sequential_target_list = list()
    t_otherNode_target_list = list()
    t_total_overlap_target_list = list()
    t_total_sequential_target_list = list()
    t_total_switch_target_list = list()

    drama_ref_list = list()
    tensor_ref_list = list()
    fp64a_ref_list = list()
    fp32a_ref_list = list()
    fp16a_ref_list = list()
    
    for row in gpu_dfs.itertuples(index=False, name='MetricRow'):
        # row is a namedtuple, you can access columns via row.<colname>
        # For example, if your metric_names are ["GPUTL", "SMACT", "TENSO"]
        # you can access row.GPUTL, row.SMACT, row.TENSO, etc.
        metric_values = list(getattr(row, metric) for metric in metrics)
        
        # Find the largest value among FLOP metrics and get its name
        flop_metrics = ['TENSO', 'FP64A', 'FP32A', 'FP16A']
        max_metric_name = max(flop_metrics, key=lambda x: metric_values[metrics.index(x)])
        max_flop_value = metric_values[metrics.index(max_metric_name)]

        t_flop_ref = sample_intv * (metric_values[metrics.index('TENSO')] + 
                                    metric_values[metrics.index('FP64A')] + 
                                    metric_values[metrics.index('FP32A')] + 
                                    metric_values[metrics.index('FP16A')])  

        tensor_ref_list.append(metric_values[metrics.index('TENSO')])
        fp64a_ref_list.append(metric_values[metrics.index('FP64A')])
        fp32a_ref_list.append(metric_values[metrics.index('FP32A')])
        fp16a_ref_list.append(metric_values[metrics.index('FP16A')])
        
        t_dram_ref = sample_intv * metric_values[metrics.index('DRAMA')]
        drama_ref_list.append(metric_values[metrics.index('DRAMA')])

        t_roofline_ref_overlap = max(t_flop_ref, t_dram_ref)
        t_roofline_ref_sequential = t_flop_ref + t_dram_ref
        t_otherGPU_ref_overlap = max(0, sample_intv * metric_values[metrics.index('GRACT')] - t_roofline_ref_overlap)
        t_otherGPU_ref_sequential = max(0, sample_intv * metric_values[metrics.index('GRACT')] - t_roofline_ref_sequential)
        t_pcie_ref = (metric_values[metrics.index('PCITX')] + metric_values[metrics.index('PCIRX')]) * sample_intv / (1e9 * ref_gpu_spec["ref_pcie_bw"]) 
        t_nvlink_ref = (metric_values[metrics.index('NVLTX')] + metric_values[metrics.index('NVLRX')]) * sample_intv / (1e9 * ref_gpu_spec["ref_nvlink_bw"])
        t_otherNode_ref = max(0, sample_intv * (1 - metric_values[metrics.index('GRACT')]) - t_pcie_ref - t_nvlink_ref)
        t_flop_target = sample_intv * (metric_values[metrics.index('TENSO')] * (ref_gpu_spec[prec_ref_mappings[precision]] / target_gpu_spec[prec_target_mappings[precision]]) +
                                       metric_values[metrics.index('FP64A')] * (ref_gpu_spec["ref_fp64"] / target_gpu_spec["target_fp64"]) + 
                                       metric_values[metrics.index('FP32A')] * (ref_gpu_spec["ref_fp32"] / target_gpu_spec["target_fp32"]) + 
                                       metric_values[metrics.index('FP16A')] * (ref_gpu_spec["ref_fp16"] / target_gpu_spec["target_fp16"]))
        
        t_dram_target = sample_intv * metric_values[metrics.index('DRAMA')] * (ref_gpu_spec["ref_mem_bw"] / target_gpu_spec["target_mem_bw"])

        t_roofline_overlap_target = max(t_flop_target, t_dram_target)
        t_roofline_overlap_target_list.append(t_roofline_overlap_target)
        t_roofline_sequential_target = t_flop_target + t_dram_target
        t_roofline_sequential_target_list.append(t_roofline_sequential_target)

        bound_ref, bound_target = check_bound_switch(ref_gpu_spec, target_gpu_spec, t_flop_ref, t_dram_ref, t_flop_target, t_dram_target)

        if bound_ref == bound_target:
            pass
        elif bound_ref != bound_target and bound_target == "memory":
            print("compute-bound switch to memory-bound")
            # t_dram_target = t_dram_target * mem_util_bound_switch
            t_dram_target = sample_intv * mem_util_bound_switch * (ref_gpu_spec["ref_mem_bw"] / target_gpu_spec["target_mem_bw"])
        elif bound_ref != bound_target and bound_target == "compute":
            print("memory-bound switch to compute-bound")
            # t_flop_target = t_flop_target * flop_util_bound_switch
            t_flop_target = sample_intv * flop_util_bound_switch * (ref_gpu_spec[metric_ref_mappings[max_metric_name]] / target_gpu_spec[metric_target_mappings[max_metric_name]])
        else:
            raise ValueError("Impossible Error")

        t_roofline_switch_target = max(t_flop_target, t_dram_target)

        t_otherGPU_overlap_target = t_otherGPU_ref_overlap * (ref_gpu_spec["ref_base_clock"] / target_gpu_spec["target_base_clock"])  * (ref_gpu_spec["ref_num_streams"] / target_gpu_spec["target_num_streams"])
        t_otherGPU_overlap_target_list.append(t_otherGPU_overlap_target)

        t_otherGPU_target_sequential = t_otherGPU_ref_sequential * (ref_gpu_spec["ref_base_clock"] / target_gpu_spec["target_base_clock"])  * (ref_gpu_spec["ref_num_streams"] / target_gpu_spec["target_num_streams"])
        t_otherGPU_sequential_target_list.append(t_otherGPU_target_sequential)
        t_pcie_target = t_pcie_ref * (ref_gpu_spec["ref_pcie_bw"] / target_gpu_spec["target_pcie_bw"])
        t_nvlink_target = t_nvlink_ref * (ref_gpu_spec["ref_nvlink_bw"] / target_gpu_spec["target_nvlink_bw"])
        t_otherNode_target = t_otherNode_ref
        t_otherNode_target_list.append(t_otherNode_target)

        t_total_overlap_target = t_roofline_overlap_target + t_otherGPU_overlap_target + t_pcie_target + t_nvlink_target + t_otherNode_target
        t_total_overlap_target_list.append(t_total_overlap_target)
        t_total_sequential_target = t_roofline_sequential_target + t_otherGPU_target_sequential + t_pcie_target + t_nvlink_target + t_otherNode_target
        t_total_sequential_target_list.append(t_total_sequential_target)    
        t_total_switch_target = t_roofline_switch_target + t_otherGPU_overlap_target + t_pcie_target + t_nvlink_target + t_otherNode_target
        t_total_switch_target_list.append(t_total_switch_target)
        
    finish_idx = int(overall_runtime_ms_ref / sample_interval_ms)
    
    if finish_idx < len(t_total_overlap_target_list):
        drama_ref_list_finish = drama_ref_list[:finish_idx]
        tensor_ref_list_finish = tensor_ref_list[:finish_idx]
        fp64a_ref_list_finish = fp64a_ref_list[:finish_idx]
        fp32a_ref_list_finish = fp32a_ref_list[:finish_idx]
        fp16a_ref_list_finish = fp16a_ref_list[:finish_idx]

        t_roofline_overlap_target_list_finish = t_roofline_overlap_target_list[:finish_idx]
        t_roofline_sequential_target_list_finish = t_roofline_sequential_target_list[:finish_idx]
        t_otherGPU_overlap_target_list_finish = t_otherGPU_overlap_target_list[:finish_idx]
        t_otherGPU_sequential_target_list_finish = t_otherGPU_sequential_target_list[:finish_idx]
        t_otherNode_target_list_finish = t_otherNode_target_list[:finish_idx]
        t_total_overlap_target_list_finish = t_total_overlap_target_list[:finish_idx]
        t_total_sequential_target_list_finish = t_total_sequential_target_list[:finish_idx]
        t_total_switch_target_list_finish = t_total_switch_target_list[:finish_idx]
    else:
        drama_ref_list_finish = drama_ref_list
        tensor_ref_list_finish = tensor_ref_list
        fp64a_ref_list_finish = fp64a_ref_list
        fp32a_ref_list_finish = fp32a_ref_list
        fp16a_ref_list_finish = fp16a_ref_list

        t_roofline_overlap_target_list_finish = t_roofline_overlap_target_list
        t_roofline_sequential_target_list_finish = t_roofline_sequential_target_list
        t_otherGPU_overlap_target_list_finish = t_otherGPU_overlap_target_list
        t_otherGPU_sequential_target_list_finish = t_otherGPU_sequential_target_list
        t_otherNode_target_list_finish = t_otherNode_target_list
        t_total_overlap_target_list_finish = t_total_overlap_target_list
        t_total_sequential_target_list_finish = t_total_sequential_target_list
        t_total_switch_target_list_finish = t_total_switch_target_list

    if start_ts is not None or end_ts is not None:
        start_idx = 0
        end_idx = len(t_total_overlap_target_list_finish)

        if start_ts is not None:
            start_idx = max(0, int(start_ts / sample_interval_ms))
        
        if end_ts is not None:
            end_idx = min(len(t_total_overlap_target_list_finish), int(end_ts / sample_interval_ms))
        
        if start_idx < end_idx:
            drama_ref_list_slice = drama_ref_list_finish[start_idx:end_idx]
            tensor_ref_list_slice = tensor_ref_list_finish[start_idx:end_idx]
            fp64a_ref_list_slice = fp64a_ref_list_finish[start_idx:end_idx]
            fp32a_ref_list_slice = fp32a_ref_list_finish[start_idx:end_idx]
            fp16a_ref_list_slice = fp16a_ref_list_finish[start_idx:end_idx]

            t_roofline_overlap_target_list_slice = t_roofline_overlap_target_list_finish[start_idx:end_idx]
            t_roofline_sequential_target_list_slice = t_roofline_sequential_target_list_finish[start_idx:end_idx]
            t_otherGPU_overlap_target_list_slice = t_otherGPU_overlap_target_list_finish[start_idx:end_idx]
            t_otherGPU_sequential_target_list_slice = t_otherGPU_sequential_target_list_finish[start_idx:end_idx]
            t_otherNode_target_list_slice = t_otherNode_target_list_finish[start_idx:end_idx]
            t_total_overlap_target_list_slice = t_total_overlap_target_list_finish[start_idx:end_idx]
            t_total_sequential_target_list_slice = t_total_sequential_target_list_finish[start_idx:end_idx]
            t_total_switch_target_list_slice = t_total_switch_target_list_finish[start_idx:end_idx]
            
        else:
            drama_ref_list_slice = []
            tensor_ref_list_slice = []
            fp64a_ref_list_slice = []
            fp32a_ref_list_slice = []
            fp16a_ref_list_slice = []

            t_roofline_overlap_target_list_slice = []
            t_roofline_sequential_target_list_slice = []
            t_otherGPU_overlap_target_list_slice = []
            t_otherGPU_sequential_target_list_slice = []
            t_otherNode_target_list_slice = []
            t_total_overlap_target_list_slice = []
            t_total_sequential_target_list_slice = []
            t_total_switch_target_list_slice = []
            raise ValueError("End Timestamp is earlier than Start Timestamp")
        
        est_mem_bw = np.mean(drama_ref_list_slice) * target_gpu_spec["target_mem_bw"]
        est_flops = (np.mean(tensor_ref_list_slice) * target_gpu_spec[prec_target_mappings[precision]] +
                     np.mean(fp64a_ref_list_slice) * target_gpu_spec["target_fp64"] + 
                     np.mean(fp32a_ref_list_slice) * target_gpu_spec["target_fp32"] +
                     np.mean(fp16a_ref_list_slice) * target_gpu_spec["target_fp16"])
        
        print("============ Target Hardware ============")
        print(f"Estimate FLOPS On Target Hardware: {est_flops:0.2f}")
        print(f"Estimate Memory BandWidth On Target Hardware: {est_mem_bw:0.2f}")
        print(f"Estimate Roofline Time On Target Hardware [Overlap Scenario]: {sum(t_roofline_overlap_target_list_slice):0.2f}")
        print(f"Estimate Roofline Time On Target Hardware [Sequential Scenario]: {sum(t_roofline_sequential_target_list_slice):0.2f}")
        print(f"Estimate otherGPU Time On Target Hardware [Overlap Scenario]: {sum(t_otherGPU_overlap_target_list_slice):0.2f}")
        print(f"Estimate otherGPU Time On Target Hardware [Sequential Scenario]: {sum(t_otherGPU_sequential_target_list_slice):0.2f}")
        print(f"Estimate otherNode Time On Target Hardware: {sum(t_otherNode_target_list_slice):0.2f}")
        print(f"Estimate Total Runtime On Target Hardware [Overlap Scenario]: {sum(t_total_overlap_target_list_slice):0.2f}")
        print(f"Estimate Total Runtime On Target Hardware [Sequential Scenario]: {sum(t_total_sequential_target_list_slice):0.2f}")
        #print(f"Estimate Runtime of Analysis Window On Target Hardware [Switch Scenario]: {sum(t_total_switch_target_list_slice):0.2f}")
        return 
    
    est_mem_bw = np.mean(drama_ref_list_finish) * target_gpu_spec["target_mem_bw"]
    est_flops = (np.mean(tensor_ref_list_finish) * target_gpu_spec[prec_target_mappings[precision]] +
                 np.mean(fp64a_ref_list_finish) * target_gpu_spec["target_fp64"] + 
                 np.mean(fp32a_ref_list_finish) * target_gpu_spec["target_fp32"] +
                 np.mean(fp16a_ref_list_finish) * target_gpu_spec["target_fp16"])
    
    print("============ Target Hardware ============")
    print(f"Estimate FLOPS On Target Hardware: {est_flops:0.2f}")
    print(f"Estimate Memory BandWidth On Target Hardware: {est_mem_bw:0.2f}")

    print(f"Estimate Roofline Time On Target Hardware [Overlap Scenario]: {sum(t_roofline_overlap_target_list_finish):0.2f}")
    print(f"Estimate Roofline Time On Target Hardware [Sequential Scenario]: {sum(t_roofline_sequential_target_list_finish):0.2f}")
    print(f"Estimate otherGPU Time On Target Hardware [Overlap Scenario]: {sum(t_otherGPU_overlap_target_list_finish):0.2f}")
    print(f"Estimate otherGPU Time On Target Hardware [Sequential Scenario]: {sum(t_otherGPU_sequential_target_list_finish):0.2f}")
    print(f"Estimate otherNode Time On Target Hardware: {sum(t_otherNode_target_list_finish):0.2f}")
    print(f"Estimate Total Runtime On Target Hardware [Overlap Scenario]: {sum(t_total_overlap_target_list_finish):0.2f}")
    print(f"Estimate Total Runtime On Target Hardware [Sequential Scenario]: {sum(t_total_sequential_target_list_finish):0.2f}")
    #print(f"Estimate Runtime On Target Hardware [Switch Scenario]: {sum(t_total_switch_target_list_finish):0.2f}")


def main():
    ###################################
    # get all parameters
    ###################################
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-f', '--dcgm_file', action='store', type=str, required=True,
                        help='indicate the dcgm output file')
    parser.add_argument('-d', '--sample_interval_ms', action='store', type=int, required=True,
                        help='indicate the sample interval in milliseconds')
    parser.add_argument('-st', '--start_timestamp', action='store', type=int, required=False, default=None,
                        help='Start timestamp for analysis window (in milliseconds)')
    parser.add_argument('-et', '--end_timestamp', action='store', type=int, required=False, default=None,
                        help='End timestamp for analysis window (in milliseconds)')
    parser.add_argument('-o', '--overall_runtime_ms', action='store', type=int, required=True,
                        help='indicate the timestamp for overall runtime in milliseconds')
    parser.add_argument('-rg', '--ref_gpu_architect', action='store', type=str, required=True, 
                        choices=['A100-40', 'A100-80', 'H100-NVL', 'H100-SXM'], help='indicate the reference gpu architecture')
    parser.add_argument('-tg', '--target_gpu_architect', action='store', type=str, default=None, 
                        choices=['A100-40', 'A100-80', 'A40', 'H100', 'R100', 'R100-UNI', 
                                 'GPU-M-IO-A-H14', 'GPU-F-IO-A-H14', 'GPU-M-IO-A-H22', 'GPU-F-IO-A-H22', 'GPU-M-IO-A-H24', 'GPU-F-IO-A-H24'], 
                        help='indicate the target gpu architecture')
    parser.add_argument('--metrics', type=list_of_strings, required=True, 
                        help='List of metrics, basically the not-none col names')
    parser.add_argument('-tp', '--tensor_precision', type=str, required=True, choices=['tf64', 'tf32', 'tf16'],
                        help='Specify the tensor precision type: TF64 (FP64 Tensor), TF32 (FP32 Tensor), TF16 (FP16 Tensor)')
    parser.add_argument('-fu', '--flop_util', action='store', type=float, default=1.0,
                        help='indicate the estimated flops utlization when bound swtich')
    parser.add_argument('-mu', '--mem_util', action='store', type=float, default=1.0,
                        help='indicate the estimated memory utlization when bound swtich')
    args = parser.parse_args()

    dcgm_metric_file = args.dcgm_file
    sample_interval_ms = args.sample_interval_ms
    overall_runtime_ms = args.overall_runtime_ms
    metrics = args.metrics
    start_ts = args.start_timestamp
    end_ts = args.end_timestamp
    ref_gpu_arch = args.ref_gpu_architect
    target_gpu_arch = args.target_gpu_architect
    flop_util = args.flop_util 
    mem_util = args.mem_util 
    tensor_precision = args.tensor_precision

    profiled_df = process_file(dcgm_metric_file, metrics)
    perf_modeling(profiled_df, metrics, overall_runtime_ms, sample_interval_ms, start_ts, end_ts, ref_gpu_arch, tensor_precision)
    
    if target_gpu_arch is not None:
        perf_predict(profiled_df, metrics, 
                     overall_runtime_ms, sample_interval_ms, start_ts, end_ts, 
                     ref_gpu_arch, target_gpu_arch, tensor_precision,
                     flop_util, mem_util)


if __name__=="__main__":
    main()