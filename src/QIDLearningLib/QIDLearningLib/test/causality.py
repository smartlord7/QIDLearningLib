"""
QIDLearningLib

Library Description (QIDLearningLib):
QIDLearningLib is a Python library designed to provide a comprehensive set of metrics for quasi-identifiers recognition processes.
The library encompasses metrics for assessing data privacy, data utility, and the performance of quasi-identifiers recognition algorithms.

Module Description (test.causality):
This module in QIDLearningLib allows to test all at once the metrics present in the module metrics.causality.

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

import matplotlib.pyplot as plt
from pandas import DataFrame
from typing import List

from QIDLearningLib.metrics.causality import balance_test, covariate_shift, propensity_score_overlap
from QIDLearningLib.util.data import generate_synthetic_dataset


def analyze_causality_metrics(
    df: DataFrame,
    quasi_identifiers: List[str],
    treatment_col: str,
    treatment_value: str
) -> None:
    """
    Analyze causality metrics for a given DataFrame and causal attributes.

    Synopse:
    This function analyzes various causality metrics, including Covariate Shift, Balance Test, and Propensity Score Overlap, for a given DataFrame and specified causal attributes.

    Details:
    The function prints and plots the results of each causality metric, providing insights into the distributional differences, balance, and propensity score overlap between treated and control groups.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - quasi_identifiers (List[str]): List of column names representing the quasi-identifiers.
    - treatment_col (str): The column name representing the treatment indicator.
    - treatment_value (str): The value indicating the treated group.

    Return:
    None: The function prints and plots causality metrics.

    Example:
    >>> df = generate_synthetic_dataset()
    >>> quasi_identifiers = ['Age', 'Gender', 'Country', 'Education']
    >>> treatment_col = 'Disease'  # Assuming there's a column indicating treatment/control groups
    >>> treatment_value = 'A'  # Specify the treatment value
    >>> analyze_causality_metrics(df, quasi_identifiers, treatment_col, treatment_value)
    plt.show()  # Display the causality metric plots

    """

    print("\nCovariate Shift Metrics:")
    causality_metrics_covariate_shift = covariate_shift(df, quasi_identifiers, treatment_col, treatment_value)
    print(repr(causality_metrics_covariate_shift))
    causality_metrics_covariate_shift.plot_all()

    print("\nBalance Test Metrics:")
    causality_metrics_balance_test = balance_test(df, quasi_identifiers, treatment_col, treatment_value)
    print(repr(causality_metrics_balance_test))
    causality_metrics_balance_test.plot_all()

    print("\nPropensity Score Overlap Metric:")
    causality_metrics_propensity_overlap = propensity_score_overlap(df, quasi_identifiers, treatment_col, treatment_value)
    print(repr(causality_metrics_propensity_overlap))
    causality_metrics_propensity_overlap.plot_all()


def main() -> None:
    """
    Main function to demonstrate the usage of causality metrics analysis.

    Synopse:
    This function generates a synthetic dataset and performs causality metrics analysis.

    Details:
    The function generates a synthetic dataset using the 'generate_synthetic_dataset' function and analyzes causality metrics using the 'analyze_causality_metrics' function. It displays the causality metric plots using matplotlib.

    Parameters:
    None

    Return:
    None

    Example:
    >>> main()
    """
    # Generate synthetic dataset
    df = generate_synthetic_dataset()

    # Example usage for causality metrics
    quasi_identifiers = ['Age', 'Gender', 'Country', 'Education']
    treatment_col = 'Disease'  # Assuming there's a column indicating treatment/control groups
    treatment_value = 'A'  # Specify the treatment value

    # Analyze causality metrics
    analyze_causality_metrics(df, quasi_identifiers, treatment_col, treatment_value)

    plt.show()


if __name__ == '__main__':
    main()
