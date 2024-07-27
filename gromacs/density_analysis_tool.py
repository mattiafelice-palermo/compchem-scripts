#!/usr/bin/env python3

"""
================================================================================
Density Analysis Tool
Version: 1.0
Date: July 27, 2024
Author: Mattia Felice Palermo
Description: This tool analyzes density data from molecular dynamics simulations,
identifying stationary points and visualizing key metrics in density fluctuations.
Usage: Run with a GROMACS EDR file to compute rolling statistics, detect stationary
points, and generate a plot highlighting density variations and averages.
================================================================================
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' for non-interactive plotting
import matplotlib.pyplot as plt
import panedr
from tabulate import tabulate

def read_density_data(file_path):
    df = panedr.edr_to_df(file_path, verbose=True)
    # Ensure Time is correctly set if it's not present in the dataframe
    if 'Time' not in df.columns:
        df['Time'] = df.index
    return df[['Time', 'Density']].reset_index(drop=True)

def calculate_rolling_stats(df, window_size):
    df['Extended Rolling Mean'] = df['Density'].rolling(window=window_size, min_periods=1).mean()
    df['Extended Rolling Std'] = df['Density'].rolling(window=window_size, min_periods=1).std()
    return df

def find_stationary_point(df, window_size, mean_threshold):
    df['Mean Change'] = df['Extended Rolling Mean'].diff().abs()
    if df[df['Mean Change'] < mean_threshold].empty:
        raise ValueError("No stationary point found with the given mean threshold.")
    stationary_time = df[df['Mean Change'] < mean_threshold].index[0]

    stationary_row = df.index.get_loc(stationary_time)
    adjusted_index = max(0, stationary_row - 10)  # Scroll back 10 points before
    return adjusted_index

def calculate_average_density(df, stationary_start):
    stationary_data = df.iloc[stationary_start:]
    return stationary_data['Density'].mean()

def identify_points_within_range(df, average_density, tolerance=0.01):
    lower_bound = average_density * (1 - tolerance)
    upper_bound = average_density * (1 + tolerance)
    filtered_df = df[(df['Density'] >= lower_bound) & (df['Density'] <= upper_bound)]
    return filtered_df  # Reset index to avoid issues in downstream operations

def plot_density(df, stationary_start, average_density, within_range_indices, best_ten_indices, file_dir):
    plt.figure(figsize=(15, 7))
    
    # Basic density and statistics plots
    plt.plot(df['Time'], df['Density'], label='Density', color='blue', marker='o', linestyle='-', markersize=5, zorder=1)
    plt.plot(df['Time'], df['Extended Rolling Mean'], color='orange', label='Extended Rolling Mean', zorder=2)
    plt.plot(df['Time'], df['Extended Rolling Std'], color='red', label='Extended Rolling Std Dev', zorder=3)
    
    plt.axvline(df['Time'].iloc[stationary_start], color='green', linestyle='--', label=f'Stationary Start @ {df["Time"].iloc[stationary_start]} ps', zorder=4)
    plt.axhline(y=average_density, color='purple', linestyle='-', label=f'Average Density: {average_density:.2f}', zorder=5)
    
    # Scatter for general points within the range
    valid_within_range = df.iloc[within_range_indices]
    plt.scatter(valid_within_range['Time'], valid_within_range['Density'], color='cyan', marker='o', label='Within 1% Range', zorder=6)

    # Best ten points with a color gradient from light gray to deep red
    if best_ten_indices is not None and len(best_ten_indices) > 0:
        valid_best_ten = df.iloc[best_ten_indices]
        colors = [plt.cm.Reds(i / len(valid_best_ten)) for i in range(len(valid_best_ten), -1, -1)]
        for idx, (index, row) in enumerate(valid_best_ten.iterrows()):
            label = 'Closest to Avg Density' if idx == 0 else ('Farthest from Avg Density' if idx == len(valid_best_ten) - 1 else '')
            plt.scatter(row['Time'], row['Density'], color=colors[idx], edgecolors='black', marker='*', s=120, label=label, zorder=10)

    plt.fill_between(df['Time'], 0, df['Density'], where=(df.index < stationary_start), color='red', alpha=0.3, label='Discarded Data', zorder=2)
    plt.xlabel('Time (ps)')
    plt.ylabel('Density (kg/m^3)')
    plt.title('Density Analysis with Stationary Point')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(file_dir, 'density_analysis.png'))
    plt.close()

def main(args):
    df = read_density_data(args.file)
    df = calculate_rolling_stats(df, args.window_size)
    try:
        stationary_start = find_stationary_point(df, args.window_size, args.mean_threshold)
    except ValueError as e:
        print(e)
        return

    average_density = calculate_average_density(df, stationary_start)
    within_range = identify_points_within_range(df.iloc[stationary_start:], average_density, args.tolerance)
    
    if not within_range.empty:
        within_range['Delta to Average'] = abs(within_range['Density'] - average_density)
        sorted_within_range = within_range.sort_values(by='Time')

        # Determine top 10 best points based on delta to average density
        best_ten = within_range.nsmallest(10, 'Delta to Average')
        points_to_show = within_range.nsmallest(args.show_points, 'Delta to Average').sort_values(by='Time')

        
        # Display points with special formatting for the best ten
        print("Points within 1% of the average density:")
        for idx, row in points_to_show.iterrows():
            if idx in best_ten.index:
                rank = list(best_ten.index).index(idx) + 1
                # Set color from green (best/closest) to red (worst/farthest)
                green = int(255 - (255 * rank / 10))
                red = int(255 * rank / 10)
                color = f"\033[38;2;{red};{green};0m"  # RGB coloring from green to red
                delta_display = f"{row['Delta to Average']:.4f} (BEST: #{rank})"
            else:
                color = "\033[0m"  # Reset to default
                delta_display = f"{row['Delta to Average']:.4f}"

            print(f"{color}Index: {idx}, Time: {row['Time']}, Density: {row['Density']:.4f}, Delta to Average: {delta_display}\033[0m")

        selected_index = int(input("Enter the index of the point you want to extract (from the list above): "))
        selected_time = within_range.loc[selected_index, 'Time']

        # Derive the paths from args.file
        file_dir = os.path.dirname(os.path.abspath(args.file))
        base_name = os.path.splitext(os.path.basename(args.file))[0]
        topol_file = os.path.join(file_dir, base_name + '.tpr')
        traj_file = os.path.join(file_dir, base_name + '.trr')
        output_file = os.path.join(file_dir, f"frame_{selected_index}.gro")

        # Build the gmx trjconv command
        command = f"echo 0 | gmx trjconv -s {topol_file} -f {traj_file} -o {output_file} -dump {selected_time}"
        print(f"Running command: {command}")

        # Uncomment to actually run the command, ensure gmx is in your PATH or specify full path to gmx
        os.system(command)
        
        # Plot adjustments to highlight the best ten points
        plot_density(df, stationary_start, average_density, within_range.index, best_ten.index, file_dir)

    else:
        print("No points found within the specified range.")

if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser(description='Analyze density data to find stationary points.')
    parser.add_argument('file', type=str, help='Path to the EDR file.')
    parser.add_argument('--window_size', type=int, default=10, help='Window size for rolling calculations.')
    parser.add_argument('--mean_threshold', type=float, default=0.1, help='Threshold for changes in the rolling mean to identify stationarity.')
    parser.add_argument('--tolerance', type=float, default=0.01, help='Tolerance for identifying points within the average density range.')
    parser.add_argument('--show_points', type=int, default=20, help='Number of points to show. Set to -1 to show all.')
    args = parser.parse_args()
    main(args)
