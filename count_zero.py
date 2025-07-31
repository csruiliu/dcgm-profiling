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
    

def count_zero(profiled_df, metrics):
    total_samples = 0
    zero_samples = 0

    for row in profiled_df.itertuples(index=False):
        # row is a namedtuple, you can access columns via row.<colname>
        # For example, if your metric_names are ["GPUTL", "SMACT", "TENSO"]
        # you can access row.GPUTL, row.SMACT, row.TENSO, etc.
        metric_values = list(getattr(row, metric) for metric in metrics)
        
        if metric_values[metrics.index('GRACT')] > 0.9:
            total_samples += 1

        if metric_values[metrics.index('TENSO')] < 0.01 and metric_values[metrics.index('GRACT')] > 0.9:
            zero_samples += 1

    print(f"Total Samples: {total_samples}, Zero Samples: {zero_samples}")    
    

def main():
    ###################################
    # get all parameters
    ###################################
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-f', '--dcgm_file', action='store', type=str, required=True,
                        help='indicate the dcgm output file')
    parser.add_argument('--metrics', type=list_of_strings, required=True, 
                        help='List of metrics, basically the not-none col names')
    args = parser.parse_args()

    dcgm_metric_file = args.dcgm_file
    metrics = args.metrics
    
    profiled_df = process_file(dcgm_metric_file, metrics)
    
    count_zero(profiled_df, metrics)
    

if __name__=="__main__":
    main()