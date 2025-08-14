import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse
import re
from collections import Counter


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
    "R100": {
        "fp64": 9.7*3.0, "fp64_tensor": 19.5*3.0, "fp32": 19.5*6.0, "tf32_tensor": 156*6.0, "fp16": 78*3.0, "fp16_tensor": 312*3.0, 
        "mem_bw": 1555*8.0, "pcie_bw": 64*25.0, "nvlink_bw": 600*6.0, "alpha_gpu": 4.0, "alpha_cpu": 3.0,
    },
    "R100-UNI": {
        "fp64": 9.7*4.0, "fp64_tensor": 19.5*4.0, "fp32": 19.5*8.0, "tf32_tensor": 156*8.0, "fp16": 78*4.0, "fp16_tensor": 312*4.0, 
        "mem_bw": 1555*1.5, "pcie_bw": 64*25.0, "nvlink_bw": 600*6.0, "alpha_gpu": 4.0, "alpha_cpu": 3.0,
    },
    "GPU-M-IO-A-H14": {
        "fp64": 9.7*1.0, "fp64_tensor": 19.5*1.0, "fp32": 19.5*1.0, "tf32_tensor": 156*1.0, "fp16": 78*1.0, "fp16_tensor": 312*1.0, 
        "mem_bw": 1555*4.0, "pcie_bw": 64*4.0, "nvlink_bw": 600*4.0, "alpha_gpu": 1.0, "alpha_cpu": 3.0,
    },
    "GPU-F-IO-A-H14": {
        "fp64": 9.7*4.0, "fp64_tensor": 19.5*4.0, "fp32": 19.5*4.0, "tf32_tensor": 156*4.0, "fp16": 78*4.0, "fp16_tensor": 312*4.0, 
        "mem_bw": 1555*1.0, "pcie_bw": 64*4.0, "nvlink_bw": 600*4.0, "alpha_gpu": 4.0, "alpha_cpu": 3.0,
    },
    "GPU-M-IO-A-H22": {
        "fp64": 9.7*2.0, "fp64_tensor": 19.5*2.0, "fp32": 19.5*2.0, "tf32_tensor": 156*2.0, "fp16": 78*2.0, "fp16_tensor": 312*2.0, 
        "mem_bw": 1555*2.0, "pcie_bw": 64*4.0, "nvlink_bw": 600*4.0, "alpha_gpu": 2.0, "alpha_cpu": 3.0,
    },
    "GPU-F-IO-A-H22": {
        "fp64": 9.7*2.0, "fp64_tensor": 19.5*2.0, "fp32": 19.5*2.0, "tf32_tensor": 156*2.0, "fp16": 78*2.0, "fp16_tensor": 312*2.0, 
        "mem_bw": 1555*2.0, "pcie_bw": 64*4.0, "nvlink_bw": 600*4.0, "alpha_gpu": 2.0, "alpha_cpu": 3.0,
    },
    "GPU-M-IO-A-H24": {
        "fp64": 9.7*2.0, "fp64_tensor": 19.5*2.0, "fp32": 19.5*2.0, "tf32_tensor": 156*2.0, "fp16": 78*2.0, "fp16_tensor": 312*2.0, 
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
def process_single_file(num_gpu, file_path, metric_names):
    # Initialize a dict to hold data lists for each GPU
    gpu_data = {i: [] for i in range(num_gpu)}

    column_mapping = {}
    header_parsed = False

    header_pattern = re.compile(r'^#Entity')
    
    # Compile regex patterns for all GPU prefixes
    # gpu_patterns = [re.compile(rf'^GPU {i}\s') for i in range(num_gpu)]

    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Process header information
            if header_pattern.match(line) and not header_parsed:
                header_parts = re.split(r'\s{2,}', line.strip())
                # Remove the '#Entity' part and 'ID' if present
                columns = [col for col in header_parts if col not in ['#Entity', 'ID']]
                
                # Create mapping from metric name to column index
                for i, col_name in enumerate(columns):
                    column_mapping[col_name] = i

                # Verify all requested metrics are present
                missing_metrics = []
                for metric in metric_names:
                    if metric not in column_mapping:
                        missing_metrics.append(metric)

                if missing_metrics:
                    raise ValueError(f"Missing metrics in data file: {missing_metrics}")
                        
                header_parsed = True
                continue
        
            # Process GPU data lines
            if line.startswith('GPU'):
                # Split the line by three or more spaces
                parts = re.split(r'\s{3,}', line.strip())

                # Extract GPU number
                gpu_match = re.search(r'GPU (\d+)', parts[0])
                if not gpu_match:
                    raise ValueError("Cannot extract GPU ID")

                gpu_id = int(gpu_match.group(1))
                if gpu_id >= num_gpu:
                    raise ValueError("The GPU ID exceeds the number of GPUs")

                # Extract data values (skip the GPU identifier)
                data_values = parts[1:]

                # Convert to numeric and extract only requested metrics in specified order
                try:
                    numeric_values = [float(x) for x in data_values]
                    
                    # Extract requested metrics in the order specified by user
                    selected_metrics = []
                    for metric in metric_names:
                        col_idx = column_mapping[metric]
                        if col_idx < len(numeric_values):
                            selected_metrics.append(numeric_values[col_idx])
                        else:
                            raise ValueError(f"Column index {col_idx} for metric {metric} out of range")
                        
                    gpu_data[gpu_id].append(selected_metrics)
                    
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line: {line}")
                    print(f"Error: {e}")
                    continue
    
    # Create DataFrames with the requested metrics as columns
    gpu_dfs = []
    for i in range(num_gpu):
        if gpu_data[i]:
            df = pd.DataFrame(gpu_data[i], columns=metric_names)
            gpu_dfs.append(df)
        else:
            # Create empty DataFrame with correct columns if no data
            gpu_dfs.append(pd.DataFrame(columns=metric_names))
     
    # returns a list of DataFrames, one per GPU
    return gpu_dfs  


def organize_by_file_content(all_files, num_gpu): 
    file_info = []
    
    for file_path in all_files:
        try:
            with open(file_path, 'r') as f:
                # Read first few lines to find GPU information
                content = ""
                for i, line in enumerate(f):
                    content += line
                    if i > 10:  # Only read first 50 lines for efficiency
                        break
                    
            # Count occurrences of different GPU IDs
            gpu_pattern = re.compile(r'GPU (\d+)')
            gpu_matches = gpu_pattern.findall(content)
            
            if gpu_matches:
                # Find the most common GPU ID in this file
                gpu_counter = Counter(gpu_matches)
                most_common_gpu_id = int(gpu_counter.most_common(1)[0][0])
                total_lines = len(gpu_matches)
                
                file_info.append((file_path, most_common_gpu_id, total_lines))
            else:
                print(f"Warning: No GPU data found in {file_path}")
                # Still include the file but with unknown GPU ID
                file_info.append((file_path, -1, 0))
                
        except Exception as e:
            print(f"Warning: Could not read file {file_path}: {e}")
            # Include the file anyway
            file_info.append((file_path, -1, 0))
    
    # Sort files by filename to ensure consistent ordering
    # This gives us a deterministic order regardless of filesystem order
    file_info.sort(key=lambda x: x[0].name)

    if len(file_info) != num_gpu:
        print(f"Content-based matching found {len(file_info)} valid files, expected {num_gpu}.")
        return None
    
    # Sort by node_id, then by gpu_id
    file_info.sort(key=lambda x: (x[1], x[2]))
    
    # Assign logical GPU IDs based on file order
    organized_files = [str(info[0]) for info in file_info]
    
    print("File organization by content analysis (order-based):")
    for logical_gpu_id, (file_path, detected_gpu_id, line_count) in enumerate(file_info):
        if detected_gpu_id >= 0:
            print(f"  Logical GPU {logical_gpu_id}: {file_path.name} (detected GPU {detected_gpu_id}, {line_count} data lines)")
        else:
            print(f"  Logical GPU {logical_gpu_id}: {file_path.name} (GPU ID unknown, {line_count} data lines)")
    
    return organized_files


def process_multiple_files_single_gpu(file_paths, metric_names):
    '''
    multiple files each file contain data of a single gpu
    '''
    gpu_dfs = []
    
    for logical_gpu_id, file_path in enumerate(file_paths):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        gpu_data = []
        column_mapping = {}
        header_parsed = False
        header_pattern = re.compile(r'^#Entity')
        
        print(f"Processing file {file_path} as logical GPU {logical_gpu_id}")
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                # Process header information
                if header_pattern.match(line) and not header_parsed:
                    header_parts = re.split(r'\s{2,}', line.strip())
                    # Remove the '#Entity' part and 'ID' if present
                    columns = [col for col in header_parts if col not in ['#Entity', 'ID']]
                    
                    # Create mapping from metric name to column index
                    for j, col_name in enumerate(columns):
                        column_mapping[col_name] = j

                    # Verify all requested metrics are present
                    missing_metrics = []
                    for metric in metric_names:
                        if metric not in column_mapping:
                            missing_metrics.append(metric)

                    if missing_metrics:
                        raise ValueError(f"Missing metrics in data file {file_path}: {missing_metrics}")
                            
                    header_parsed = True
                    continue
            
                # Process ALL GPU data lines (regardless of the GPU ID mentioned in the line)
                if line.startswith('GPU'):
                    # Split the line by three or more spaces
                    parts = re.split(r'\s{3,}', line.strip())

                    # Extract data values (skip the GPU identifier)
                    # We ignore the actual GPU ID in the line since this file represents one logical GPU
                    data_values = parts[1:]

                    # Convert to numeric and extract only requested metrics in specified order
                    try:
                        numeric_values = [float(x) for x in data_values]
                        
                        # Extract requested metrics in the order specified by user
                        selected_metrics = []
                        for metric in metric_names:
                            col_idx = column_mapping[metric]
                            if col_idx < len(numeric_values):
                                selected_metrics.append(numeric_values[col_idx])
                            else:
                                raise ValueError(f"Column index {col_idx} for metric {metric} out of range")
                            
                        gpu_data.append(selected_metrics)
                        
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse line in {file_path}: {line}")
                        print(f"Error: {e}")
                        continue
        
        # Create DataFrame for this logical GPU
        if gpu_data:
            df = pd.DataFrame(gpu_data, columns=metric_names)
            gpu_dfs.append(df)
            print(f"Logical GPU {logical_gpu_id}: Created DataFrame with {len(gpu_data)} rows")
        else:
            # Create empty DataFrame with correct columns if no data
            gpu_dfs.append(pd.DataFrame(columns=metric_names))
            print(f"Warning: No data found for logical GPU {logical_gpu_id} in file {file_path}")
    
    return gpu_dfs


def scan_and_organize_gpu_files(folder_path, num_gpu):
    """
    Scan a folder for GPU data files and organize them by logical GPU ID.
    
    Args:
        folder_path: Path to folder containing GPU data files
        num_gpu: Expected total number of GPUs
    
    Returns:
        List of file paths ordered by logical GPU ID
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Find all potential GPU data files (adjust extensions as needed)
    file_patterns = ['*.out']
    all_files = []
    for pattern in file_patterns:
        all_files.extend(folder_path.glob(pattern))
    
    if not all_files:
        raise FileNotFoundError(f"No data files found in {folder_path}")
    
    print(f"Found {len(all_files)} potential GPU data files in {folder_path}")

    organized_files = organize_by_file_content(all_files, num_gpu)
    if organized_files:
        return organized_files


# Update the wrapper function to handle folder input
def process_files(num_gpu, dcgm_input, metric_names):
    if os.path.isdir(dcgm_input):
        print(f"Processing folder: {dcgm_input}")
        file_paths = scan_and_organize_gpu_files(dcgm_input, num_gpu)
        profiled_df = process_multiple_files_single_gpu(file_paths, metric_names)
    elif os.path.isfile(dcgm_input):
        print(f"Processing single file with {num_gpu} GPUs: {dcgm_input}")
        profiled_df = process_single_file(num_gpu, dcgm_input, metric_names)
    else:
        raise ValueError(f"Input path '{dcgm_input}' is neither a valid file nor a directory")

    return profiled_df


def perf_modeling_per_gpu(df, metrics, finish_idx, sample_interval_ms, start_ts, end_ts, hw_pcie_gb, hw_nvlink_gb):
    t_total_list = list()
    sample_intv = sample_interval_ms / 1000

    for row in df.itertuples(index=False, name='MetricRow'):
        # row is a namedtuple, you can access columns via row.<colname>
        # For example, if your metric_names are ["GPUTL", "SMACT", "TENSO"]
        # you can access row.GPUTL, row.SMACT, row.TENSO, etc.
        metric_values = list(getattr(row, metric) for metric in metrics)
        
        t_flop = sample_intv * (metric_values[metrics.index('TENSO')] + 
                                metric_values[metrics.index('FP64A')] + 
                                metric_values[metrics.index('FP32A')] + 
                                metric_values[metrics.index('FP16A')])  
        t_dram = sample_intv * metric_values[metrics.index('DRAMA')]
        
        t_roofline = max(t_flop, t_dram)
        
        t_otherGPU = max(0, sample_intv * metric_values[metrics.index('GRACT')] - t_roofline)

        t_pcie = (metric_values[metrics.index('PCITX')] + metric_values[metrics.index('PCIRX')]) * sample_intv / (1000 * 1000 * 1000 * hw_pcie_gb) 

        t_nvlink = (metric_values[metrics.index('NVLTX')] + metric_values[metrics.index('NVLRX')]) * sample_intv / (1000 * 1000 * 1000 * hw_nvlink_gb)

        t_otherNode = max(0, sample_intv * (1 - metric_values[metrics.index('GRACT')]) - t_pcie - t_nvlink)

        t_total = t_roofline + t_otherGPU + t_pcie + t_nvlink + t_otherNode

        t_total_list.append(t_total)
    
    if finish_idx < len(t_total_list):
        t_total_list_finish = t_total_list[:finish_idx]
    else:
        t_total_list_finish = t_total_list

    if start_ts is not None or end_ts is not None:
        start_idx = 0
        end_idx = len(t_total_list_finish)

        if start_ts is not None:
            start_idx = max(0, int(start_ts / sample_interval_ms))
        
        if end_ts is not None:
            end_idx = min(len(t_total_list_finish), int(end_ts / sample_interval_ms))
        
        if start_idx < end_idx:
            t_total_list_slice = t_total_list_finish[start_idx:end_idx]
        else:
            t_total_list_slice = []
            raise ValueError("End Timestamp is earlier than Start Timestamp")

        return t_total_list_slice

    return t_total_list_finish


def perf_modeling(gpu_dfs, metrics, overall_runtime_ms, sample_interval_ms, agg_interval_ms, start_ts, end_ts, gpu_arch):
    finish_idx = int(overall_runtime_ms / sample_interval_ms)
    
    if gpu_arch == 'A100-40' or gpu_arch == 'A100-80':
        hw_pcie_gb = 64
        hw_nvlink_gb = 600
    else:
        raise ValueError("Reference GPU arch is not recognized")

    t_total_dict = dict()
    for i, df in enumerate(gpu_dfs):
        if not df.empty:
            t_totals = perf_modeling_per_gpu(df, metrics, finish_idx, sample_interval_ms, start_ts, end_ts, hw_pcie_gb, hw_nvlink_gb)
            t_total_dict[f"GPU{i}"] = t_totals
        else:
            raise ValueError("The total time list is empty")

    # Now compute the max for each row index
    # First, check that all lists are of the same length
    lengths = [len(lst) for lst in t_total_dict.values()]
    if len(set(lengths)) != 1:
        raise ValueError("Not all GPU t_total lists are of the same length!")
    
    num_rows = lengths[0]
    # Aggregate every `agg_samples` samples
    # When agg_interval_ms == sample_interval_ms, aggregation is on a row basis
    agg_samples = agg_interval_ms // sample_interval_ms

    # Transpose the lists and take max of every `agg_samples` samples
    max_list = []
    
    for start in range(0, num_rows, agg_samples):
        end = min(start + agg_samples, num_rows)
        # For each row in this window, find the max across GPUs, then find the max in the window
        agg_time_gpus = {
            gpu: sum(t_total_dict[gpu][row_idx] for row_idx in range(start, end))
            for gpu in t_total_dict
        }
        window_max = max(agg_time_gpus.values())
        max_list.append(window_max)

    print(f"Estimate Runtime On Reference Hardware: {sum(max_list):.2f}")


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


def pref_predict_per_gpu(df, metrics, finish_idx, sample_interval_ms, start_ts, end_ts, ref_gpu_spec, target_gpu_spec, precision, flop_util_bound_switch, mem_util_bound_switch):    
    
    t_total_overlap_target_list = list()
    t_total_sequential_target_list = list()
    t_total_switch_target_list = list()

    drama_ref_list = list()
    tensor_ref_list = list()
    fp64a_ref_list = list()
    fp32a_ref_list = list()
    fp16a_ref_list = list()

    sample_intv = sample_interval_ms / 1000

    for row in df.itertuples(index=False, name='MetricRow'):
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
        
        t_pcie_ref = (metric_values[metrics.index('PCITX')] + metric_values[metrics.index('PCIRX')]) * sample_intv / (1000 * 1000 * 1000 * ref_gpu_spec["ref_pcie_bw"]) 
        
        t_nvlink_ref = (metric_values[metrics.index('NVLTX')] + metric_values[metrics.index('NVLRX')]) * sample_intv / (1000 * 1000 * 1000 * ref_gpu_spec["ref_nvlink_bw"])
        
        t_otherNode_ref = max(0, sample_intv * (1 - metric_values[metrics.index('GRACT')]) - t_pcie_ref - t_nvlink_ref)

        t_flop_target = sample_intv * (metric_values[metrics.index('TENSO')] * (ref_gpu_spec[prec_ref_mappings[precision]] / target_gpu_spec[prec_target_mappings[precision]]) +
                                       metric_values[metrics.index('FP64A')] * (ref_gpu_spec["ref_fp64"] / target_gpu_spec["target_fp64"]) + 
                                       metric_values[metrics.index('FP32A')] * (ref_gpu_spec["ref_fp32"] / target_gpu_spec["target_fp32"]) + 
                                       metric_values[metrics.index('FP16A')] * (ref_gpu_spec["ref_fp16"] / target_gpu_spec["target_fp16"]))
        t_dram_target = sample_intv * metric_values[metrics.index('DRAMA')] * (ref_gpu_spec["ref_mem_bw"] / target_gpu_spec["target_mem_bw"])

        t_roofline_target_overlap = max(t_flop_target, t_dram_target)
        
        t_roofline_target_sequential = t_flop_target + t_dram_target
        
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
        
        t_roofline_target_switch = max(t_flop_target, t_dram_target)
        
        if "target_alpha_gpu" in target_gpu_spec:
            t_otherGPU_target_overlap = t_otherGPU_ref_overlap * (1 / target_gpu_spec["target_alpha_gpu"])
            t_otherGPU_target_sequential = t_otherGPU_ref_sequential * (1 / target_gpu_spec["target_alpha_gpu"])
        else:
            t_otherGPU_target_overlap = t_otherGPU_ref_overlap * (ref_gpu_spec["ref_base_clock"] / target_gpu_spec["target_base_clock"]) * (ref_gpu_spec["ref_num_streams"] / target_gpu_spec["target_num_streams"])
            t_otherGPU_target_sequential = t_otherGPU_ref_sequential * (ref_gpu_spec["ref_base_clock"] / target_gpu_spec["target_base_clock"]) * (ref_gpu_spec["ref_num_streams"] / target_gpu_spec["target_num_streams"])

        t_pcie_target = t_pcie_ref * (ref_gpu_spec["ref_pcie_bw"] / target_gpu_spec["target_pcie_bw"])
        
        t_nvlink_target = t_nvlink_ref * (ref_gpu_spec["ref_nvlink_bw"] / target_gpu_spec["target_nvlink_bw"])
        
        if "target_alpha_gpu" in target_gpu_spec:
            t_otherNode_target = t_otherNode_ref * (1 / target_gpu_spec["target_alpha_cpu"])
        else:
            t_otherNode_target = t_otherNode_ref

        t_total_target_overlap = t_roofline_target_overlap + t_otherGPU_target_overlap + t_pcie_target + t_nvlink_target + t_otherNode_target

        t_total_target_sequential = t_roofline_target_sequential + t_otherGPU_target_sequential + t_pcie_target + t_nvlink_target + t_otherNode_target

        t_total_target_switch = t_roofline_target_switch + t_otherGPU_target_overlap + t_pcie_target + t_nvlink_target + t_otherNode_target

        t_total_overlap_target_list.append(t_total_target_overlap) 

        t_total_sequential_target_list.append(t_total_target_sequential)    

        t_total_switch_target_list.append(t_total_target_switch)

    if finish_idx < len(t_total_overlap_target_list):
        t_total_overlap_target_list_finish = t_total_overlap_target_list[:finish_idx]
        t_total_sequential_target_list_finish = t_total_sequential_target_list[:finish_idx]
        t_total_switch_target_list_finish = t_total_switch_target_list[:finish_idx]
        drama_ref_list_finish = drama_ref_list[:finish_idx]
        tensor_ref_list_finish = tensor_ref_list[:finish_idx]
        fp64a_ref_list_finish = fp64a_ref_list[:finish_idx]
        fp32a_ref_list_finish = fp32a_ref_list[:finish_idx]
        fp16a_ref_list_finish = fp16a_ref_list[:finish_idx]
    else:
        t_total_overlap_target_list_finish = t_total_overlap_target_list
        t_total_sequential_target_list_finish = t_total_sequential_target_list
        t_total_switch_target_list_finish = t_total_switch_target_list
        drama_ref_list_finish = drama_ref_list
        tensor_ref_list_finish = tensor_ref_list
        fp64a_ref_list_finish = fp64a_ref_list
        fp32a_ref_list_finish = fp32a_ref_list   
        fp16a_ref_list_finish = fp16a_ref_list

    if start_ts is not None or end_ts is not None:
        start_idx = 0
        end_idx = len(t_total_overlap_target_list_finish)

        if start_ts is not None:
            start_idx = max(0, int(start_ts / sample_interval_ms))
        
        if end_ts is not None:
            end_idx = min(len(t_total_overlap_target_list_finish), int(end_ts / sample_interval_ms))
        
        if start_idx < end_idx:
            t_total_overlap_target_list_slice = t_total_overlap_target_list_finish[start_idx:end_idx]
            t_total_sequential_target_list_slice = t_total_sequential_target_list_finish[start_idx:end_idx]
            t_total_switch_target_list_slice = t_total_switch_target_list_finish[start_idx:end_idx]
            drama_ref_list_slice = drama_ref_list_finish[start_idx:end_idx]
            tensor_ref_list_slice = tensor_ref_list_finish[start_idx:end_idx]
            fp64a_ref_list_slice = fp64a_ref_list_finish[start_idx:end_idx]
            fp32a_ref_list_slice = fp32a_ref_list_finish[start_idx:end_idx]
            fp16a_ref_list_slice = fp16a_ref_list_finish[start_idx:end_idx]
        else:
            t_total_overlap_target_list_slice = []
            t_total_sequential_target_list_slice = []
            t_total_switch_target_list_slice = []
            drama_ref_list_slice = []
            tensor_ref_list_slice = []
            fp64a_ref_list_slice = []
            fp32a_ref_list_slice = []
            fp16a_ref_list_slice = []
            raise ValueError("End Timestamp is earlier than Start Timestamp")
            
        est_mem_bw = np.mean(drama_ref_list_slice) * target_gpu_spec["target_mem_bw"]
        
        est_flops = (np.mean(tensor_ref_list_slice) * target_gpu_spec[prec_target_mappings[precision]] +
                     np.mean(fp64a_ref_list_slice) * target_gpu_spec["target_fp64"] + 
                     np.mean(fp32a_ref_list_slice) * target_gpu_spec["target_fp32"] +
                     np.mean(fp16a_ref_list_slice) * target_gpu_spec["target_fp16"])
        print(f"Estimate FLOPS On Target Hardware: {est_flops:0.2f}")
        print(f"Estimate Memory BandWidth On Target Hardware: {est_mem_bw:0.2f}")
        return t_total_overlap_target_list_slice, t_total_sequential_target_list_slice, t_total_switch_target_list_slice
    
    return t_total_overlap_target_list_finish, t_total_sequential_target_list_finish, t_total_switch_target_list_finish


def perf_predict(gpu_dfs, metrics, overall_runtime_ms_ref, sample_interval_ms, agg_interval_ms, start_ts, end_ts, ref_gpu_arch, target_gpu_arch, precision, flop_util, mem_util):
    finish_idx = int(overall_runtime_ms_ref / sample_interval_ms)

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

    t_total_overlap_dict = dict()
    t_total_sequential_dict = dict()
    t_total_switch_dict = dict()

    for i, df in enumerate(gpu_dfs):
        if not df.empty:
            t_total_overlap, t_total_sequential, t_total_switch = pref_predict_per_gpu(df, metrics, 
                                                                                       finish_idx, sample_interval_ms, start_ts, end_ts, 
                                                                                       ref_gpu_spec, target_gpu_spec, precision, 
                                                                                       flop_util, mem_util)
            t_total_overlap_dict[f"GPU{i}"] = t_total_overlap
            t_total_sequential_dict[f"GPU{i}"] = t_total_sequential
            t_total_switch_dict[f"GPU{i}"] = t_total_switch
        else:
            raise ValueError("The total time list is empty")

    # Now compute the max for each row index
    # First, check that all lists are of the same length
    lengths = [len(lst) for lst in t_total_overlap_dict.values()]
    if len(set(lengths)) != 1:
        raise ValueError("Not all GPU t_total lists are of the same length!")
    
    num_rows = lengths[0]
    # Aggregate every `agg_samples` samples
    # When agg_interval_ms == sample_interval_ms, aggregation is on a row basis
    agg_samples = agg_interval_ms // sample_interval_ms

    # Transpose the lists and take max of every `agg_samples` samples
    max_value_overlap_list = []
    max_value_sequential_list = []
    max_value_switch_list = []

    for start in range(0, num_rows, agg_samples):
        end = min(start + agg_samples, num_rows)
        # For each row in this window, find the max across GPUs, then find the max in the window
        agg_time_gpus = {
            gpu: sum(t_total_overlap_dict[gpu][row_idx] for row_idx in range(start, end))
            for gpu in t_total_overlap_dict
        }

        # window_max = max(agg_time_gpus.values())
        max_index, max_value = max(enumerate(agg_time_gpus.values()), key=lambda x: x[1])
        max_value_overlap_list.append(max_value)
    
    for start in range(0, num_rows, agg_samples):
        end = min(start + agg_samples, num_rows)
        # For each row in this window, find the max across GPUs, then find the max in the window
        agg_time_gpus = {
            gpu: sum(t_total_sequential_dict[gpu][row_idx] for row_idx in range(start, end))
            for gpu in t_total_sequential_dict
        }

        max_index, max_value = max(enumerate(agg_time_gpus.values()), key=lambda x: x[1])
        max_value_sequential_list.append(max_value)

    for start in range(0, num_rows, agg_samples):
        end = min(start + agg_samples, num_rows)
        # For each row in this window, find the max across GPUs, then find the max in the window
        agg_time_gpus = {
            gpu: sum(t_total_switch_dict[gpu][row_idx] for row_idx in range(start, end))
            for gpu in t_total_switch_dict
        }

        max_index, max_value = max(enumerate(agg_time_gpus.values()), key=lambda x: x[1])
        max_value_switch_list.append(max_value)

    print(f"Estimate Runtime On Target Hardware [Overlap Mode]: {sum(max_value_overlap_list):.2f}")
    print(f"Estimate Runtime On Target Hardware [Sequential Mode]: {sum(max_value_sequential_list):.2f}")
    # print(f"Estimate Runtime On Target Hardware [Switch Mode]: {sum(max_value_switch_list):.2f}")


def main():
    ###################################
    # get all parameters
    ###################################
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-f', '--dcgm_input', action='store', type=str, required=True,
                        help='DCGM input: either a single file (containing multiple GPU data) or a folder path (containing multiple files with one GPU data each)')
    parser.add_argument('-n', '--num_gpu', action='store', type=int, required=True,
                        help='indicate number of gpus used for computation')    
    parser.add_argument('-o', '--overall_runtime_ms', action='store', type=int, required=True,
                        help='indicate the timestamp for overall runtime in milliseconds')
    parser.add_argument('-st', '--start_timestamp', action='store', type=int, required=False, default=None,
                        help='Start timestamp for analysis window (in milliseconds)')
    parser.add_argument('-et', '--end_timestamp', action='store', type=int, required=False, default=None,
                        help='End timestamp for analysis window (in milliseconds)')
    parser.add_argument('-s', '--sample_interval_ms', action='store', type=int, required=True,
                        help='indicate the sample interval in milliseconds')
    parser.add_argument('-a', '--aggregate_interval_ms', action='store', type=int, required=True,
                        help='indicate the time interval for aggregation in milliseconds') 
    parser.add_argument('-rg', '--ref_gpu_architect', action='store', type=str, required=True, 
                        choices=['A100-40', 'A100-80'], help='indicate the reference gpu architecture')
    parser.add_argument('-tg', '--target_gpu_architect', action='store', type=str, default=None, 
                        choices=['A100-40', 'A100-80', 'A40', 'H100', 'R100', 'R100-UNI', 'GPU-M-IO-A-H14', 'GPU-F-IO-A-H14'], 
                        help='indicate the target gpu architecture')
    parser.add_argument('--metrics', type=list_of_strings, required=True, 
                        help='List of metrics, basically the not-none col names')
    parser.add_argument('-p', '--precision', type=str, required=False,  default='double', choices=['double', 'single', 'half'],
                        help='Specify the precision type: double (FP64), single (FP32), half (FP16), or tensor (Tensor ops). Default: single')
    parser.add_argument('-fu', '--flop_util', action='store', type=float, default=1.0,
                        help='indicate the estimated flops utlization when bound swtich')
    parser.add_argument('-mu', '--mem_util', action='store', type=float, default=1.0,
                        help='indicate the estimated memory utlization when bound swtich')
    args = parser.parse_args()

    dcgm_input = args.dcgm_input
    num_gpu = args.num_gpu
    overall_runtime_ms = args.overall_runtime_ms
    start_ts = args.start_timestamp
    end_ts = args.end_timestamp
    sample_interval_ms = args.sample_interval_ms
    agg_interval_ms = args.aggregate_interval_ms
    metrics = args.metrics
    ref_gpu_arch = args.ref_gpu_architect
    target_gpu_arch = args.target_gpu_architect
    flop_util = args.flop_util
    mem_util = args.mem_util
    precision = args.precision

    profiled_df = process_files(num_gpu, dcgm_input, metrics)

    perf_modeling(profiled_df, metrics, overall_runtime_ms, sample_interval_ms, agg_interval_ms, start_ts, end_ts, ref_gpu_arch)
    
    if target_gpu_arch is not None:
        perf_predict(profiled_df, metrics, 
                     overall_runtime_ms, sample_interval_ms, agg_interval_ms, start_ts, end_ts, 
                     ref_gpu_arch, target_gpu_arch, precision, 
                     flop_util, mem_util)
    

if __name__=="__main__":
    main()