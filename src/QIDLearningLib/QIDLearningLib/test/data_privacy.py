"""
QIDLearningLib

Library Description (QIDLearningLib):
QIDLearningLib is a Python library designed to provide a vast set of metrics for quasi-identifiers recognition processes.
The library includes metrics for assessing data privacy, data utility, and the performance of quasi-identifiers recognition algorithms.

Module Description (test.data_privacy):
This module in QIDLearningLib allows to test all at once the metrics present in the module metrics.data_privacy.

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

from metrics.data_privacy import *
from util.data import generate_synthetic_dataset
from util.time import measure_time


def analyze_privacy_metrics(df, quasi_identifiers, sensitive_attribute):
    # Measure performance
    original_mean, original_std = measure_time(k_anonymity, df, quasi_identifiers)
    numpy_mean, numpy_std = measure_time(k_anonymity_numpy, df, quasi_identifiers)

    # Print results
    print(f"Original function duration: {original_mean:.6f} seconds (±{original_std:.6f})")
    print(f"NumPy-based function duration: {numpy_mean:.6f} seconds (±{numpy_std:.6f})")
    print(f"Speedup: {original_mean / numpy_mean:.2f}x")

    # Print k-Anonymity results
    print("k-Anonymity:")
    k_anonymity_metric = k_anonymity(df, quasi_identifiers)
    print(repr(k_anonymity_metric))
    k_anonymity_metric.plot_all()

    print("k-Anonymity numpy:")
    k_anonymity_metric = k_anonymity_numpy(df, quasi_identifiers)
    print(repr(k_anonymity_metric))
    k_anonymity_metric.plot_all()



def main():
    # Generate synthetic dataset
    df = generate_synthetic_dataset()

    # Example usage
    quasi_identifiers = ['Age', 'Gender', 'Country', 'Education']
    sensitive_attribute = 'Disease'

    # Analyze privacy metrics
    analyze_privacy_metrics(df, quasi_identifiers, sensitive_attribute)


if __name__ == '__main__':
    main()
