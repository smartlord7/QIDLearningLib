"""
QIDLearningLib

Description:
A Python library designed to offer a comprehensive set of metrics for quasi-identification recognition processes.
The library includes metrics that assess data privacy, data utility, and the performance of quasi-identification recognition algorithms.

Year: 2023/2024
Institution: University of Coimbra
Department: Department of Informatics Engineering
Program: Master's in Informatics Engineering - Intelligent Systems
Author: Sancho Amaral Simões
Student No: 2019217590
Email: sanchoamaralsimoes@gmail.com
Version: v0.01

License:
This open-source software is released under the terms of the GNU General Public License, version 3 (GPL-3.0).
For more details, see https://www.gnu.org/licenses/gpl-3.0.html

"""

import numpy as np
import pandas as pd  # Make sure to import pandas if not already done


def separation(df: pd.DataFrame, attributes: list) -> float:
    """
    Calculate the Separation metric for a given DataFrame and attributes.

    Synopse:
    Separation metric measures the degree of separation among instances based on specified attributes.

    Details:
    The metric is computed by counting the number of unique attribute combinations and their occurrences in the DataFrame.
    It subtracts the number of occurrences minus one for each unique combination and calculates the percentage of separation.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - attributes (list): List of column names representing the attributes.

    Return:
    float: Separation metric value.

    Example:
    >>> df = pd.DataFrame({'A': [1, 2, 1, 2], 'B': ['X', 'Y', 'X', 'Y']})
    >>> separation_value = separation(df, ['A', 'B'])
    >>> print(separation_value)
    50.0

    See Also:
    - distinction: Calculates the Distinction metric.

    """

    # Extract the subset of the DataFrame with specified attributes
    subset_df = df[attributes]

    # Count the occurrences of each unique attribute combination
    _, counts = np.unique(subset_df, axis=0, return_counts=True)

    # Calculate the total to subtract for separation metric
    to_subtract = np.sum((counts - 1) * counts // 2)

    # Calculate the total possible combinations
    total = subset_df.shape[0] * (subset_df.shape[0] - 1) / 2

    # Calculate and return the separation metric
    separation_value = (total - to_subtract) * 100 / total

    return separation_value


def distinction(df: pd.DataFrame, attributes: list) -> float:
    """
    Calculate the Distinction metric for a given DataFrame and attributes.

    Synopse:
    Distinction metric measures the distinctness of instances based on specified attributes.

    Details:
    The metric is computed by counting the number of unique attribute combinations and calculating the percentage of distinction.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - attributes (list): List of column names representing the attributes.

    Return:
    float: Distinction metric value.

    Example:
    >>> df = pd.DataFrame({'A': [1, 2, 1, 2], 'B': ['X', 'Y', 'X', 'Y']})
    >>> distinction_value = distinction(df, ['A', 'B'])
    >>> print(distinction_value)
    100.0

    See Also:
    - separation: Calculates the Separation metric.
    """

    # Extract the subset of the DataFrame with specified attributes
    subset_df = df[attributes]

    # Count the number of unique attribute combinations
    unique = np.unique(subset_df, axis=0) if len(subset_df.shape) == 2 else np.unique(subset_df)

    # Calculate and return the distinction metric
    dist = (len(unique) / subset_df.shape[0]) * 100

    return dist