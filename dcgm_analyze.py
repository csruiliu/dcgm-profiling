import pandas as pd
import numpy as np
import argparse
import re
import matplotlib.pyplot as plt


def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


# Function to read the file and process the data
def process_file(file_path):
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
        
    # Transpose the data to get columns
    lowest_values = list()
    highest_values = list()
    has_zero_values = list()

    metrics_name = ["GPUTL", "SMACT", "TENSO", "DRAMA", "FP64A", "FP32A", "FP16A", "TIMMA", "THMMA"]
    for idx, column in enumerate(zip(*data)):
        col_values = list(column)
        plt.figure(figsize=(60, 12)) 
        plt.plot(col_values, marker='o', markersize=1, linestyle='-', color='royalblue', linewidth=0.1)
        
        # Add title and labels to the plot
        plt.title(metrics_name[idx])
        plt.xlabel('Index')
        plt.ylabel('Value')
        if metrics_name[idx] == "GPUTL":
            plt.ylim(0, 110)
            plt.yticks([0, 20, 40, 60, 80, 100], ["0", "20%", "40%", "60%", "80%", "100%"], fontsize=18)
        else:
            plt.ylim(0, 1.1)
            plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ["0", "20%", "40%", "60%", "80%", "100%"], fontsize=18)
        # Show the plot
        plt.savefig(metrics_name[idx] + '.png', dpi=300)

        zeros_exist = any(num == 0 for num in col_values)
        col_values_filtered = [num for num in col_values if num != 0]

        if len(col_values_filtered) == 0:
            col_values_filtered = [0]

        has_zero_values.append(zeros_exist)

        lowest_values.append(min(col_values_filtered))
        highest_values.append(max(col_values_filtered))
    
    return lowest_values, highest_values, has_zero_values


def main():
    ###################################
    # get all parameters
    ###################################
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--dcgm_file', action='store', type=str,
                        help='indicate the dcgm output file')
    args = parser.parse_args()

    output_file = args.dcgm_file

    # Get the minimum and maximum values for each column
    min_values, max_values, zero_values = process_file(output_file)

    print(min_values, max_values, zero_values)


if __name__=="__main__":
    main()