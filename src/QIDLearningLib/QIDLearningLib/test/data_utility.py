"""
QIDLearningLib

Library Description (QIDLearningLib):
QIDLearningLib is a Python library designed to provide a vast set of metrics for quasi-identifiers recognition processes.
The library includes metrics for assessing data privacy, data utility, and the performance of quasi-identifiers recognition algorithms.

Module Description (test.data_utility):
This module in QIDLearningLib allows to test all at once the metrics present in the module metrics.data_utility.

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

import pandas as pd

from QIDLearningLib.metrics.data_utility import (
    mean_squared_err,
    accuracy,
    utility_score,
    range_utility,
    distinct_values_utility,
    completeness_utility
)
from QIDLearningLib.util.data import generate_synthetic_dataset, calculate_core
from typing import List


def analyze_data_utility_metrics(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    target_attribute: str,
    true_values: pd.Series
) -> None:
    """
    Analyze data utility metrics for a given DataFrame and attributes.

    Synopse:
    This function prints and plots various data utility metrics, including Mean Squared Error, Accuracy, Utility Score (Sum), Range Utility, Distinct Values Utility, Completeness Utility, and Core, for a given DataFrame and specified attributes.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - quasi_identifiers (List[str]): List of column names representing the quasi-identifiers.
    - target_attribute (str): The column name representing the target attribute.
    - true_values (pd.Series): True values of the target attribute.

    Return:
    None: The function prints and plots data utility metrics.

    Example:
    >>> df = generate_synthetic_dataset()
    >>> quasi_identifiers = ['Age', 'Gender', 'Country', 'Education']
    >>> target_attribute = 'Income'
    >>> true_values = df[target_attribute]
    >>> analyze_data_utility_metrics(df, quasi_identifiers, target_attribute, true_values)

    """

    print("Mean Squared Error:")
    mse_metric = mean_squared_err(df, quasi_identifiers, target_attribute, true_values)
    print(repr(mse_metric))
    mse_metric.plot_all()

    print("\nAccuracy:")
    accuracy_metric = accuracy(df, quasi_identifiers, target_attribute, true_values)
    print(repr(accuracy_metric))
    accuracy_metric.plot_all()

    print("\nUtility Score (Sum):")
    utility_metric_sum = utility_score(df, quasi_identifiers, target_attribute, lambda x: x.sum())
    print(repr(utility_metric_sum))
    utility_metric_sum.plot_all()

    print("\nRange Utility:")
    range_utility_metric = range_utility(df, quasi_identifiers, target_attribute)
    print(repr(range_utility_metric))
    range_utility_metric.plot_all()

    print("\nDistinct Values Utility:")
    distinct_values_utility_metric = distinct_values_utility(df, quasi_identifiers, target_attribute)
    print(repr(distinct_values_utility_metric))
    distinct_values_utility_metric.plot_all()

    print("\nCompleteness Utility:")
    completeness_utility_metric = completeness_utility(df, quasi_identifiers, target_attribute)
    print(repr(completeness_utility_metric))
    completeness_utility_metric.plot_all()

    print("\nCore:")
    print(calculate_core(df))

def main() -> None:
    """
    Main function to demonstrate the usage of data utility metrics analysis.

    Synopse:
    This function generates a synthetic dataset and performs data utility metrics analysis using the 'analyze_data_utility_metrics' function.

    Parameters:
    None

    Return:
    None

    Example:
    >>> main()
    """
    # Generate synthetic dataset
    df = generate_synthetic_dataset()

    # Example usage
    quasi_identifiers = ['Age', 'Gender', 'Country', 'Education']
    target_attribute = 'Income'
    true_values = df[target_attribute]

    # Analyze data utility metrics
    analyze_data_utility_metrics(df, quasi_identifiers, target_attribute, true_values)


if __name__ == '__main__':
    main()
