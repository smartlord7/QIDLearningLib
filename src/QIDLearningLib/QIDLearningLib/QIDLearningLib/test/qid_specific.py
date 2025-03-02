"""
QIDLearningLib

Library Description (QIDLearningLib):
QIDLearningLib is a Python library designed to provide a vast set of metrics for quasi-identifiers recognition processes.
The library includes metrics for assessing data privacy, data utility, and the performance of quasi-identifiers recognition algorithms.

Module Description (test.qid_specific):
This module in QIDLearningLib allows to test all at once the metrics present in the module metrics.qid_specific.

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

import pandas as pd  # Make sure to import pandas if not already done
from QIDLearningLib.metrics.qid_specific import separation, distinction
from QIDLearningLib.util.data import generate_synthetic_dataset


def test_qid_metrics(df: pd.DataFrame, attributes: list) -> None:
    """
    Test quasi-identifiers metrics for a given DataFrame and attributes.

    Synopse:
    This function tests various quasi-identifiers metrics, including Separation and Distinction, for a given DataFrame and specified attributes.

    Details:
    The function prints the results of each quasi-identifiers metric, providing insights into the separation and distinctness of instances based on the specified attributes.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - attributes (list): List of column names representing the attributes.

    Return:
    None: The function prints quasi-identifiers metrics.

    Example:
    >>> df = pd.DataFrame({'A': [1, 2, 1, 2], 'B': ['X', 'Y', 'X', 'Y']})
    >>> attributes = ['A', 'B']
    >>> test_qid_metrics(df, attributes)

    """

    print("\nSeparation Metric:")
    separation_value = separation(df, attributes)
    print(f"Separation Value: {separation_value}%")

    print("\nDistinction Metric:")
    distinction_value = distinction(df, attributes)
    print(f"Distinction Value: {distinction_value}%")


def main() -> None:
    """
    Main function to demonstrate the usage of quasi-identifiers metrics.

    Synopse:
    This function generates a synthetic dataset and performs quasi-identifiers metrics analysis.

    Details:
    The function generates a synthetic dataset using the 'generate_synthetic_dataset' function and analyzes quasi-identifiers metrics using the 'test_qid_metrics' function.

    Parameters:
    None

    Return:
    None

    Example:
    >>> main()
    """

    # Generate synthetic dataset
    df = generate_synthetic_dataset()

    # Example usage for quasi-identifiers metrics
    attributes = ['Age', 'Gender', 'Country', 'Education']

    # Test quasi-identifiers metrics
    test_qid_metrics(df, attributes)


if __name__ == '__main__':
    main()
