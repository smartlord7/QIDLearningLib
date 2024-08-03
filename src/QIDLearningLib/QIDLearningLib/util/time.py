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


def measure_time(func, df, quasi_identifiers, sensitive_attributes, num_runs=30):
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


def compare_results(original_metric, numpy_metric):
    if np.array_equal(original_metric.values, numpy_metric.values):
        return True
    else:
        return False


# Function to run tests
def run_tests(test_cases, df):
    for data in test_cases:
        original_func = data[0]
        new_func = data[1]
        quasi_identifiers = data[2]

        sensitive_attributes = None

        if len(data) == 4:
            sensitive_attributes = data[3]

        # Measure performance
        original_mean, original_std = measure_time(original_func, df, quasi_identifiers, sensitive_attributes)
        new_mean, new_std = measure_time(new_func, df, quasi_identifiers, sensitive_attributes)

        # Get results

        if sensitive_attributes:
            original_metric = original_func(df, quasi_identifiers, sensitive_attributes)
            new_metric = new_func(df, quasi_identifiers, sensitive_attributes)
        else:
            original_metric = original_func(df, quasi_identifiers)
            new_metric = new_func(df, quasi_identifiers)

        original_metric.plot_all()
        new_metric.plot_all()

        # Compare results
        results_match = compare_results(original_metric, new_metric)

        # Print performance results
        print(f"{original_func.__name__} vs {new_func.__name__}")
        print(f"Original function duration: {original_mean:.6f} seconds (±{original_std:.6f})")
        print(f"New function duration: {new_mean:.6f} seconds (±{new_std:.6f})")
        print(f"Speedup: {original_mean / new_mean:.2f}x")
        print(f"Results match: {results_match}")
        print("-" * 40)