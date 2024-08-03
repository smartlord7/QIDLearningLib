"""
QIDLearningLib

Library Description (QIDLearningLib):
QIDLearningLib is a Python library designed to provide a comprehensive set of metrics for quasi-identifiers recognition processes.
The library encompasses metrics for assessing data privacy, data utility, and the performance of quasi-identifiers recognition algorithms.

Module Description (metrics.qid_specific):
This module in QIDLearningLib includes functions to calculate various metrics related to quasi-identifiers recognition.
These metrics measure aspects such as data separation, distinction.

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

    # Convert DataFrame to NumPy array with a specific dtype
    array_values = df[attributes].to_numpy(dtype=str)

    # Count the occurrences of each unique attribute combination
    _, counts = np.unique(array_values, axis=0, return_counts=True)

    # Calculate the total to subtract for separation metric
    to_subtract = np.sum((counts - 1) * counts // 2)

    # Calculate the total possible combinations
    total = array_values.shape[0] * (array_values.shape[0] - 1) / 2

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

    # Convert DataFrame to NumPy array with a specific dtype
    array_values = df[attributes].to_numpy(dtype=str)

    # Count the number of unique attribute combinations
    unique = np.unique(array_values, axis=0) if len(array_values.shape) == 2 else np.unique(array_values)

    # Calculate and return the distinction metric
    dist = (len(unique) / array_values.shape[0]) * 100

    return dist
