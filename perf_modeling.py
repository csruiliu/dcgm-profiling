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

    # Compile regex patterns for all GPU prefixes
    gpu_patterns = [re.compile(rf'^GPU {i}\s') for i in range(num_gpu)]

    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        for i, pattern in enumerate(gpu_patterns):
            if pattern.match(line):
                # Split the line by three or more spaces
                values = re.split(r'\s{3,}', line.strip())

                # Only get number values, so "GPU x" will be discarded
                numeric_values = [float(value) for value in values if is_number(value)]

                if len(numeric_values) == len(metric_names):
                    gpu_data[i].append(numeric_values)
                else:
                    raise ValueError(f"The number of data columns doesn't match the number of metric names for GPU {i}")
                
                # Only one GPU per line, so stop checking patterns
                break  
     
    gpu_dfs = [pd.DataFrame(gpu_data[i], columns=metric_names) for i in range(num_gpu)]

    # returns a list of DataFrames, one per GPU
    return gpu_dfs  

    
# Function to plot dataframe
def perf_modeling_per_gpu(df, metrics, sample_intv, hw_pcie_gb, hw_nvlink_gb):
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

        t_pcie = (metric_values[metrics.index('PCITX')] + metric_values[metrics.index('PCIRX')]) * sample_intv / (1000 * 1000 * hw_pcie_gb) 

        t_nvlink = (metric_values[metrics.index('NVLTX')] + metric_values[metrics.index('NVLRX')]) * sample_intv / (1000 * 1000 * hw_nvlink_gb)

        t_otherNode = max(0, sample_intv * (1 - metric_values[metrics.index('GRACT')]) - t_pcie - t_nvlink)

        t_total = t_roofline + t_otherGPU + t_pcie + t_nvlink + t_otherNode

        t_total_list.append(t_total)

    return t_total_list


def perf_modeling(gpu_dfs, metrics, sample_interval_ms, agg_interval_ms, hw_pcie_gb, hw_nvlink_gb):
    sample_interval = sample_interval_ms / 1000
    t_total_dict = dict()
    for i, df in enumerate(gpu_dfs):
        if not df.empty:
            t_totals = perf_modeling_per_gpu(df, metrics, sample_interval, hw_pcie_gb, hw_nvlink_gb)
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

    print(sum(max_list))


def main():
    ###################################
    # get all parameters
    ###################################
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-f', '--dcgm_file', action='store', type=str, required=True,
                        help='indicate the dcgm output file')
    parser.add_argument('-n', '--num_gpu', action='store', type=int, required=True,
                        help='indicate number of gpus used for computation')
    parser.add_argument('-d', '--sample_interval_ms', action='store', type=int, required=True,
                        help='indicate the sample interval in milliseconds')
    parser.add_argument('-a', '--aggregate_interval_ms', action='store', type=int, required=True,
                        help='indicate the time interval for aggregation in milliseconds') 
    parser.add_argument('--metrics', type=list_of_strings, required=True, 
                        help='List of metrics, basically the not-none col names')
    parser.add_argument('-h', '--help', action='help',
                        help='Example: python3 dcgm_analyze.py -f gpu_util/results/xx.100.out -o ./gpu_util/results -d 100 --metrics GRACT,PCITX,PCIRX')
    args = parser.parse_args()

    dcgm_metric_file = args.dcgm_file
    num_gpu = args.num_gpu
    sample_interval_ms = args.sample_interval_ms
    agg_interval_ms = args.aggregate_interval_ms
    metric_names = args.metrics
    
    hw_pcie_gb = 64
    hw_nvlink_gb = 600 

    profiled_results_df = process_file(num_gpu, dcgm_metric_file, metric_names)
    
    perf_modeling(profiled_results_df, metric_names, sample_interval_ms, agg_interval_ms, hw_pcie_gb, hw_nvlink_gb)


if __name__=="__main__":
    main()