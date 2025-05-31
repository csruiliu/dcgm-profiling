import pandas as pd
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

def detect_sleep_period(numbers, n=5, exclude_trailing=True):
    """
    Find all non-overlapping sequences of n consecutive zeros 
    that appear after the first non-zero number.
    
    Args:
        numbers: List of numbers to search
        n: Number of consecutive zeros to detect (default: 5)
        exclude_trailing: If True, exclude zero sequences that extend to the end
    
    Returns:
        List of tuples (start_index, end_index) for each sequence found
    """
        # Handle pandas Series
    if hasattr(numbers, 'iloc'):
        numbers_list = numbers.tolist()
    else:
        numbers_list = numbers

    results = []
    
    # Convert n to integer if needed
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    elif not isinstance(n, int) or n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    
    # Find the first non-zero number
    first_nonzero_index = None
    for i, num in enumerate(numbers_list):
        if num != 0:
            first_nonzero_index = i
            break
    
    # If no non-zero number found, return empty list
    if first_nonzero_index is None:
        return results
    
    # Find the last non-zero number
    last_nonzero_index = None
    for i in range(len(numbers_list) - 1, -1, -1):
        if numbers_list[i] != 0:
            last_nonzero_index = i
            break
    
    # Start searching for n consecutive zeros after the first non-zero
    i = first_nonzero_index + 1
    
    while i <= len(numbers_list) - n:
        # Check if we have n consecutive zeros starting at index i
        if all(numbers_list[i+j] == 0 for j in range(n)):
            end_index = i + n - 1
            
            # Check if this is a trailing sequence
            is_trailing = exclude_trailing and all(numbers_list[j] == 0 for j in range(end_index + 1, len(numbers_list)))
            
            if not is_trailing:
                results.append((i, end_index))
            
            i += n  # Jump past this sequence
        else:
            i += 1
    
    return results

def get_non_sleep_sums_from_sleep_periods(numbers, sleep_periods):
    """
    Calculate sums of non-sleep periods based on detected sleep periods.
    
    Args:
        numbers: List or pandas Series of numbers
        sleep_periods: List of tuples (start_index, end_index) for sleep periods
    
    Returns:
        List of sums for each non-sleep period
    """
    if len(numbers) == 0:
        return []
    
    # Handle pandas Series
    if hasattr(numbers, 'iloc'):
        get_sum = lambda start, end: numbers.iloc[start:end+1].sum()
    else:
        get_sum = lambda start, end: sum(numbers[start:end+1])
    
    # If no sleep periods, sum the entire list
    if not sleep_periods:
        return [get_sum(0, len(numbers) - 1)]
    
    # Sort sleep periods by start index
    sleep_periods = sorted(sleep_periods, key=lambda x: x[0])
    
    sums = []
    
    # Sum before first sleep period (if exists)
    if sleep_periods[0][0] > 0:
        sums.append(get_sum(0, sleep_periods[0][0] - 1))
    
    # Sums between sleep periods
    for i in range(len(sleep_periods) - 1):
        start = sleep_periods[i][1] + 1
        end = sleep_periods[i + 1][0] - 1
        if start <= end:
            sums.append(get_sum(start, end))
    
    # Sum after last sleep period (if exists)
    if sleep_periods[-1][1] < len(numbers) - 1:
        sums.append(get_sum(sleep_periods[-1][1] + 1, len(numbers) - 1))
    
    return sums

# Function to plot dataframe
def perf_modeling(gpu_dfs, metrics, sample_interval_ms, sleep_threshold_ms, hw_pcie_gb, hw_nvlink_gb):
    sample_intv = sample_interval_ms / 1000
    
    t_total_list = list()
    
    for row in gpu_dfs.itertuples(index=False, name='MetricRow'):
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

    sleep_period = int(sleep_threshold_ms / sample_interval_ms)

    sleep_marks = detect_sleep_period(gpu_dfs['DRAMA'], sleep_period)

    sums = get_non_sleep_sums_from_sleep_periods(t_total_list, sleep_marks)
    
    print(sums)

    # print(sum(t_total))

    


def main():
    ###################################
    # get all parameters
    ###################################
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-f', '--dcgm_file', action='store', type=str, required=True,
                        help='indicate the dcgm output file')
    parser.add_argument('-g', '--gpu_architect', action='store', type=str, required=True, choices=['A100', 'A40', 'H100'],
                        help='indicate the gpu architecture')
    parser.add_argument('-d', '--sample_interval_ms', action='store', type=int, required=True,
                        help='indicate the sample interval in milliseconds')
    parser.add_argument('-s', '--sleep_period_ms', action='store', type=int, required=True,
                        help='indicate the sleep period during GPU execution in milliseconds')  
    parser.add_argument('--metrics', type=list_of_strings, required=True, 
                        help='List of metrics, basically the not-none col names')
    parser.add_argument('-h', '--help', action='help',
                        help='Example: python3 dcgm_analyze.py -f gpu_util/results/xx.100.out -o ./gpu_util/results -d 100 --metrics GRACT,PCITX,PCIRX')
    args = parser.parse_args()

    dcgm_metric_file = args.dcgm_file
    gpu_arch = args.gpu_architect
    sample_interval_ms = args.sample_interval_ms
    sleep_threshold_ms = args.sleep_period_ms
    metric_names = args.metrics
    
    if gpu_arch == 'A100':
        hw_pcie_gb = 64
        hw_nvlink_gb = 600
    elif gpu_arch == 'A40':
        hw_pcie_gb = 64
        hw_nvlink_gb = 112.5
    else:
        hw_pcie_gb = 128
        hw_nvlink_gb = 900

    profiled_results_df = process_file(dcgm_metric_file, metric_names)
    
    perf_modeling(profiled_results_df, metric_names, sample_interval_ms, sleep_threshold_ms, hw_pcie_gb, hw_nvlink_gb)


if __name__=="__main__":
    main()