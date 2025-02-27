"""
QIDLearningLib

Library Description (QIDLearningLib):
QIDLearningLib is a Python library designed to provide a vast set of metrics for quasi-identifiers recognition processes.
The library includes metrics for assessing data privacy, data utility, and the performance of quasi-identifiers recognition algorithms.

Module Description (util.time):
This module in QIDLearningLib provides primitives for time efficiency benchmarking.

Year: 2023/2024
Institution: University of Coimbra
Department: Department of Informatics Engineering
Program: Master's in Informatics Engineering - Intelligent Systems
Author: Sancho Amaral Simões
Student No: 2019217590
Emails: sanchoamaralsimoes@gmail.com (Personal)| uc2019217590@student.uc.pt | sanchosimoes@student.dei.uc.pt
Version: v0.01

License:
This open-source software is released under the terms of the GNU General Public License, version 3 (GPL-3.0).
For more details, see https://www.gnu.org/licenses/gpl-3.0.html

"""


import time
import numpy as np
import pandas as pd


def measure_time(func, df, quasi_identifiers, sensitive_attributes, num_runs=50):
    times = []
    for i in range(num_runs):
        print(f"Run {i + 1}/{num_runs}")
        start_time = time.time()
        if sensitive_attributes:
            func(df, quasi_identifiers, sensitive_attributes)
        else:
            func(df, quasi_identifiers)

        times.append(time.time() - start_time)
    return np.mean(times), np.std(times)


def compare_results(original_metric, new_metric):
    # Ensure both metrics are numpy arrays for comparison
    original_values = np.array(original_metric.values)
    new_values = np.array(new_metric.values)

    # Check if both arrays have the same shape
    if original_values.shape != new_values.shape:
        raise ValueError("The shapes of the two metrics do not match.")

    # Calculate the number of differences
    num_differences = np.sum(original_values != new_values)
    diff_sum = np.sum(original_values - new_values)

    return num_differences, diff_sum


def run_tests(test_cases, df, num_runs):
    # List to store comparison results
    comparisons = []

    # Extract DataFrame size
    df_size = len(df)

    # Filename with parameters
    filename = f'test_results_num_runs_{num_runs}_df_size_{df_size}.csv'

    for data in test_cases:
        original_func = data[0]
        new_func = data[1]
        quasi_identifiers = data[2]
        sensitive_attributes = data[3] if len(data) == 4 else None

        # Measure performance
        original_mean, original_std = measure_time(original_func, df, quasi_identifiers, sensitive_attributes, num_runs)
        new_mean, new_std = measure_time(new_func, df, quasi_identifiers, sensitive_attributes, num_runs)

        # Get results
        if sensitive_attributes:
            original_metric = original_func(df, quasi_identifiers, sensitive_attributes)
            new_metric = new_func(df, quasi_identifiers, sensitive_attributes)
        else:
            original_metric = original_func(df, quasi_identifiers)
            new_metric = new_func(df, quasi_identifiers)

        # Compare results
        results_diff, results_diff_sum = compare_results(original_metric, new_metric)

        # Store comparison results
        comparisons.append({
            'Original Function': original_func.__name__,
            'New Function': new_func.__name__,
            'Original Mean Time': original_mean,
            'Original Std Dev Time': original_std,
            'New Mean Time': new_mean,
            'New Std Dev Time': new_std,
            'Speedup': original_mean / new_mean,
            'Number of Differences': results_diff,
            'Sum of Differences': results_diff_sum
        })

        # Print performance results
        print(f"{original_func.__name__} vs {new_func.__name__}")
        print(f"Original function duration: {original_mean:.6f} seconds (±{original_std:.6f})")
        print(f"New function duration: {new_mean:.6f} seconds (±{new_std:.6f})")
        print(f"Speedup: {original_mean / new_mean:.2f}x")
        print(f"Results differences|sum of diff: {results_diff}|{results_diff_sum}")
        print("-" * 40)

    # Save results to CSV
    df_comparisons = pd.DataFrame(comparisons)
    df_comparisons.to_csv(filename, index=False)

    print(f"Results saved to {filename}")