import pandas as pd
import numpy as np
import argparse
import re


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
def process_file(num_gpu, file_path, metric_names):
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

    
# Function to plot dataframe
def perf_modeling_per_gpu(df, metrics, finish_idx, sample_intv, hw_pcie_gb, hw_nvlink_gb):
    t_total_list = list()

    for row in df.itertuples(index=False, name='MetricRow'):
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
    
    if finish_idx < len(t_total_list):
        t_total_list_finish = t_total_list[:finish_idx]
    else:
        t_total_list_finish = t_total_list

    return t_total_list_finish


def perf_modeling(gpu_dfs, metrics, overall_runtime_ms, sample_interval_ms, agg_interval_ms, gpu_arch):
    sample_interval = sample_interval_ms / 1000
    finish_idx = int(overall_runtime_ms / sample_interval_ms)

    if gpu_arch == 'A100-40' or gpu_arch == 'A100-80':
        hw_pcie_gb = 64
        hw_nvlink_gb = 600
    else:
        raise ValueError("Reference GPU arch is not recognized")

    t_total_dict = dict()
    for i, df in enumerate(gpu_dfs):
        if not df.empty:
            t_totals = perf_modeling_per_gpu(df, metrics, finish_idx, sample_interval, hw_pcie_gb, hw_nvlink_gb)
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


def pref_predict_per_gpu(df, metrics, finish_idx, sample_intv, ref_gpu_spec, target_gpu_spec, flop_util_bound_switch, mem_util_bound_switch):    
    t_total_bursty_target_list = list()
    t_total_interleave_target_list = list()
    t_total_switch_target_list = list()

    for row in df.itertuples(index=False, name='MetricRow'):
        # row is a namedtuple, you can access columns via row.<colname>
        # For example, if your metric_names are ["GPUTL", "SMACT", "TENSO"]
        # you can access row.GPUTL, row.SMACT, row.TENSO, etc.
        metric_values = list(getattr(row, metric) for metric in metrics)
        
        # Find the largest value among FLOP metrics and get its name
        flop_metrics = ['TENSO', 'FP64A', 'FP32A', 'FP16A']
        max_metric_name = max(flop_metrics, key=lambda x: metric_values[metrics.index(x)])
        max_flop_value = metric_values[metrics.index(max_metric_name)]

        # Mapping from metric names to GPU spec keys
        metric_to_spec = {
            'TENSO': 'ref_fp64_tensor',
            'FP64A': 'ref_fp64',
            'FP32A': 'ref_fp32',
            'FP16A': 'ref_fp16'
        }

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

        t_flop_target = (sample_intv * metric_values[metrics.index('TENSO')] * (ref_gpu_spec["ref_fp64_tensor"] / target_gpu_spec["target_fp64_tensor"]) + 
                         sample_intv * metric_values[metrics.index('FP64A')] * (ref_gpu_spec["ref_fp64"] / target_gpu_spec["target_fp64"]) + 
                         sample_intv * metric_values[metrics.index('FP32A')] * (ref_gpu_spec["ref_fp32"] / target_gpu_spec["target_fp32"]) + 
                         sample_intv * metric_values[metrics.index('FP16A')] * (ref_gpu_spec["ref_fp16"] / target_gpu_spec["target_fp16"]))
        t_dram_target = sample_intv * metric_values[metrics.index('DRAMA')] * (ref_gpu_spec["ref_mem_bw"] / target_gpu_spec["target_mem_bw"])
        
        t_roofline_target_interleave = max(t_flop_target, t_dram_target)

        t_roofline_target_bursty = t_flop_target + t_dram_target

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
            t_flop_target = sample_intv * flop_util_bound_switch * (ref_gpu_spec[metric_to_spec[max_metric_name]] / target_gpu_spec[metric_to_spec[max_metric_name]])
        else:
            raise ValueError("Impossible Error")
        
        t_roofline_target_switch = max(t_flop_target, t_dram_target)

        t_otherGPU_target = t_otherGPU_ref

        t_pcie_target = t_pcie_ref * (ref_gpu_spec["ref_pcie_bw"] / target_gpu_spec["target_pcie_bw"])
        
        t_nvlink_target = t_nvlink_ref * (ref_gpu_spec["ref_nvlink_bw"] / target_gpu_spec["target_nvlink_bw"])
        
        t_otherNode_target = t_otherNode_ref

        t_total_target_bursty = t_roofline_target_bursty + t_otherGPU_target + t_pcie_target + t_nvlink_target + t_otherNode_target

        t_total_target_interleave = t_roofline_target_interleave + t_otherGPU_target + t_pcie_target + t_nvlink_target + t_otherNode_target

        t_total_target_switch = t_roofline_target_switch + t_otherGPU_target + t_pcie_target + t_nvlink_target + t_otherNode_target

        t_total_bursty_target_list.append(t_total_target_bursty)    

        t_total_interleave_target_list.append(t_total_target_interleave) 

        t_total_switch_target_list.append(t_total_target_switch)
    
    if finish_idx < len(t_total_interleave_target_list):
        t_total_interleave_target_list_finish = t_total_interleave_target_list[:finish_idx]
        t_total_bursty_target_list_finish = t_total_bursty_target_list[:finish_idx]
        t_total_switch_target_list_finish = t_total_switch_target_list[:finish_idx]
    else:
        t_total_interleave_target_list_finish = t_total_interleave_target_list
        t_total_bursty_target_list_finish = t_total_bursty_target_list
        t_total_switch_target_list_finish = t_total_switch_target_list

    return t_total_interleave_target_list_finish, t_total_bursty_target_list_finish, t_total_switch_target_list_finish


def perf_predict(gpu_dfs, metrics, overall_runtime_ms_ref, sample_interval_ms, agg_interval_ms, ref_gpu_arch, target_gpu_arch, flop_util, mem_util):
    sample_intv = sample_interval_ms / 1000
    finish_idx = int(overall_runtime_ms_ref / sample_interval_ms)

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

    t_total_bursty_dict = dict()
    t_total_interleaved_dict = dict()
    t_total_switch_dict = dict()

    for i, df in enumerate(gpu_dfs):
        if not df.empty:
            t_total_interleaved, t_total_bursty, t_total_switch = pref_predict_per_gpu(df, metrics, finish_idx, sample_intv, ref_gpu_spec, target_gpu_spec, flop_util, mem_util)
            t_total_bursty_dict[f"GPU{i}"] = t_total_bursty
            t_total_interleaved_dict[f"GPU{i}"] = t_total_interleaved
            t_total_switch_dict[f"GPU{i}"] = t_total_switch
        else:
            raise ValueError("The total time list is empty")

    # Now compute the max for each row index
    # First, check that all lists are of the same length
    lengths = [len(lst) for lst in t_total_interleaved_dict.values()]
    if len(set(lengths)) != 1:
        raise ValueError("Not all GPU t_total lists are of the same length!")
    
    num_rows = lengths[0]
    # Aggregate every `agg_samples` samples
    # When agg_interval_ms == sample_interval_ms, aggregation is on a row basis
    agg_samples = agg_interval_ms // sample_interval_ms

    # Transpose the lists and take max of every `agg_samples` samples
    max_value_bursty_list = []
    max_value_interleaved_list = []
    max_value_switch_list = []

    for start in range(0, num_rows, agg_samples):
        end = min(start + agg_samples, num_rows)
        # For each row in this window, find the max across GPUs, then find the max in the window
        agg_time_gpus = {
            gpu: sum(t_total_bursty_dict[gpu][row_idx] for row_idx in range(start, end))
            for gpu in t_total_bursty_dict
        }

        # window_max = max(agg_time_gpus.values())
        max_index, max_value = max(enumerate(agg_time_gpus.values()), key=lambda x: x[1])
        max_value_bursty_list.append(max_value)
    
    for start in range(0, num_rows, agg_samples):
        end = min(start + agg_samples, num_rows)
        # For each row in this window, find the max across GPUs, then find the max in the window
        agg_time_gpus = {
            gpu: sum(t_total_interleaved_dict[gpu][row_idx] for row_idx in range(start, end))
            for gpu in t_total_interleaved_dict
        }

        max_index, max_value = max(enumerate(agg_time_gpus.values()), key=lambda x: x[1])
        max_value_interleaved_list.append(max_value)

    for start in range(0, num_rows, agg_samples):
        end = min(start + agg_samples, num_rows)
        # For each row in this window, find the max across GPUs, then find the max in the window
        agg_time_gpus = {
            gpu: sum(t_total_switch_dict[gpu][row_idx] for row_idx in range(start, end))
            for gpu in t_total_switch_dict
        }

        max_index, max_value = max(enumerate(agg_time_gpus.values()), key=lambda x: x[1])
        max_value_switch_list.append(max_value)

    print(f"Estimate Runtime On Target Hardware [Interleave Mode]: {sum(max_value_interleaved_list):.2f}")
    print(f"Estimate Runtime On Target Hardware [Bursty Mode]: {sum(max_value_bursty_list):.2f}")
    print(f"Estimate Runtime On Target Hardware [Switch Mode]: {sum(max_value_switch_list):.2f}")


def main():
    ###################################
    # get all parameters
    ###################################
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-f', '--dcgm_file', action='store', type=str, required=True,
                        help='indicate the dcgm output file')
    parser.add_argument('-n', '--num_gpu', action='store', type=int, required=True,
                        help='indicate number of gpus used for computation')    
    parser.add_argument('-o', '--overall_runtime_ms', action='store', type=int, required=True,
                        help='indicate the timestamp for overall runtime in milliseconds')
    parser.add_argument('-s', '--sample_interval_ms', action='store', type=int, required=True,
                        help='indicate the sample interval in milliseconds')
    parser.add_argument('-a', '--aggregate_interval_ms', action='store', type=int, required=True,
                        help='indicate the time interval for aggregation in milliseconds') 
    parser.add_argument('-rg', '--ref_gpu_architect', action='store', type=str, required=True, 
                        choices=['A100-40', 'A100-80'], help='indicate the reference gpu architecture')
    parser.add_argument('-tg', '--target_gpu_architect', action='store', type=str, default=None, 
                        choices=['A100-40', 'A100-80', 'A40', 'H100'], help='indicate the target gpu architecture')
    parser.add_argument('--metrics', type=list_of_strings, required=True, 
                        help='List of metrics, basically the not-none col names')
    parser.add_argument('-fu', '--flop_util', action='store', type=float, required=True,
                        help='indicate the estimated flops utlization when bound swtich')
    parser.add_argument('-mu', '--mem_util', action='store', type=float, required=True,
                        help='indicate the estimated memory utlization when bound swtich')
    args = parser.parse_args()

    dcgm_metric_file = args.dcgm_file
    num_gpu = args.num_gpu
    overall_runtime_ms = args.overall_runtime_ms
    sample_interval_ms = args.sample_interval_ms
    agg_interval_ms = args.aggregate_interval_ms
    metrics = args.metrics
    ref_gpu_arch = args.ref_gpu_architect
    target_gpu_arch = args.target_gpu_architect
    flop_util = args.flop_util # such as 0.3
    mem_util = args.mem_util # such as 0.33

    profiled_df = process_file(num_gpu, dcgm_metric_file, metrics)
    
    perf_modeling(profiled_df, metrics, overall_runtime_ms, sample_interval_ms, agg_interval_ms, ref_gpu_arch)

    if target_gpu_arch is not None:
        perf_predict(profiled_df, metrics, overall_runtime_ms, sample_interval_ms, agg_interval_ms, ref_gpu_arch, target_gpu_arch, flop_util, mem_util)


if __name__=="__main__":
    main()