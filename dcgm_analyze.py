import pandas as pd
import numpy as np
import argparse
import re
import matplotlib.pyplot as plt


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
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = list()
    for line in lines:
        # Split the line by three or more spaces
        values = re.split(r'\s{3,}', line.strip())
        numeric_values = [float(value) for value in values if is_number(value)]
        
        if len(numeric_values) != 0:
            data.append(numeric_values)
  
    if len(data[0]) == len(metric_names):
        df = pd.DataFrame(data, columns=metric_names)
        return df
    else:
        raise ValueError("The number of data columns doesn't match the number of metric names")
    

# Function to plot dataframe
def plot(df, metric_names, output, dcgm_delay):
    print(df.columns)
    for metric in metric_names:
        plt.figure(figsize=(15, 4)) 
        
        # Add title and labels to the plot
        plt.title(metric)
        plt.xlabel('Time Index')
        plt.ylabel('Value')
        if metric == "GPUTL":
            col_values = df[metric]
            plt.ylim(0, 110)
            plt.yticks([0, 20, 40, 60, 80, 100], ["0", "20%", "40%", "60%", "80%", "100%"], fontsize=18)
        elif metric == "MCUTL":
            col_values = df[metric]
            plt.ylim(0, 110)
            plt.yticks([0, 20, 40, 60, 80, 100], ["0", "20%", "40%", "60%", "80%", "100%"], fontsize=18)
        elif metric == "PCITX":
            col_values = [element / (1024 * 1024 * 1024) for element in df[metric]]
            plt.ylim(0, max(col_values) * 1.1)
            plt.ylabel('Rate of Data Transmitted over PCIe (GiB/s)')
        elif metric == "PCIRX":
            col_values = [element / (1024 * 1024 * 1024) for element in df[metric]]
            plt.ylim(0, max(col_values) * 1.1)
            plt.ylabel('Rate of Data Recevied over PCIe (GiB/s)')
        elif metric == "POWER":
            col_values = df[metric]
            plt.ylim(0, col_values.max() * 1.1)
            plt.ylabel('Watts (W)')
            plt.xlabel('Time Index (Seconds)', fontsize=12)
        elif metric == "TMPTR":
            plt.ylim(0, 110) 
            plt.ylabel('Celsius (°C)')
            col_values = df[metric]
        elif metric == "TOTEC":
            col_values = [element / (1000 * 1000) for element in df[metric]]
            plt.ylim(0, max(col_values) * 1.1) 
            plt.ylabel('Kilojoule (KJ)')
        elif metric == "FP64A" or metric == "FP32A" or metric == "FP16A" or metric == "TENSO":
            plt.ylim(0, 1.1)
            plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ["0", "20%", "40%", "60%", "80%", "100%"], fontsize=12)
            col_values = df[metric]
            plt.ylabel('Ratio of cycles the tensor core is active', fontsize=12)
            plt.xlabel('Time Index (Seconds)', fontsize=12)
        else:
            plt.ylim(0, 1.1)
            plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ["0", "20%", "40%", "60%", "80%", "100%"], fontsize=18)
            col_values = df[metric]

        plt.xlabel(f'Time Interval: {dcgm_delay} ms', fontsize=12)
        
        # Show the plot
        plt.plot(col_values, marker='o', markersize=0.7, linestyle='-', color='royalblue', linewidth=0.5)
        plt.savefig(output + "/" + metric + '.png', dpi=300, bbox_inches='tight', pad_inches=0.0)
        


def main():
    ###################################
    # get all parameters
    ###################################
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-f', '--dcgm_file', action='store', type=str,
                        help='indicate the dcgm output file')
    parser.add_argument('-o', '--output_path', action='store', type=str,
                        help='indicate the output figures path')
    parser.add_argument('-d', '--dcgm_delay', action='store', type=int, choices=[1, 10, 100, 1000, 10000],
                        help='indicate the sample rate used for plotting') 
    parser.add_argument('--metrics', type=list_of_strings, help='List of metrics, basically the not-none col names')
    parser.add_argument('-h', '--help', action='help',
                        help='Example: python3 dcgm_analyze.py -f gpu_util/results/xx.100.out -o ./gpu_util/results -d 100 --metrics GRACT,PCITX,PCIRX')
    # metric_cols = ["GPUTL", "SMACT", "TENSO", "DRAMA", "FP64A", "FP32A", "FP16A", "TIMMA", "THMMA"]
    # metric_cols = ["GPUTL", "MCUTL", "GRACT", "PCITX", "PCIRX"]
    # metric_cols = ["TMPTR", "POWER", "TOTEC", "GPUTL", "SMACT", "TENSO", "DRAMA", "FP64A", "FP32A", "FP16A", "TIMMA", "THMMA"]
    args = parser.parse_args()

    dcgm_metric_file = args.dcgm_file
    output_file = args.output_path
    dcgm_delay = args.dcgm_delay
    metric_cols = args.metrics
    
    df_metrics = process_file(dcgm_metric_file, metric_cols)
    
    plot(df_metrics, df_metrics, output_file, dcgm_delay)
    
    # Find the minimum value of each column
    min_values = df_metrics.min()

    # Find the maximum value of each column
    max_values = df_metrics.max()

    print("Minimum values of each column:")
    print(min_values)
    print("\nMaximum values of each column:")
    print(max_values)


if __name__=="__main__":
    main()