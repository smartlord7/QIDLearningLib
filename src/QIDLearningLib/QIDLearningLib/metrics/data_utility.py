"""
QIDLearningLib

Library Description:
QIDLearningLib is a Python library designed to provide a vast set of metrics for quasi-identifiers recognition processes.
The library includes metrics for assessing data privacy, data utility, and the performance of quasi-identifiers recognition algorithms.

Module Description (metrics.data_utility):
This module in QIDLearningLib includes functions to calculate various metrics related to the data utility regarding the quasi identifiers.
These metrics measure aspects such as the predictive power of each quasi identifier group

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
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score

from QIDLearningLib.structure.grouped_metric import GroupedMetric


def mean_squared_err(df: pd.DataFrame, quasi_identifiers: list, target_attribute: str, true_values: pd.Series) -> GroupedMetric:
    """
    Calculate the Mean Squared Error metric for a given DataFrame and quasi-identifiers.

    Synopse:
    Mean Squared Error measures the average squared difference between true and predicted values.

    Details:
    The metric is computed by grouping the DataFrame based on quasi-identifiers, calculating the mean of the target
    attribute within each group, and then calculating the mean squared error for each group.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - quasi_identifiers (list): List of column names representing quasi-identifiers.
    - target_attribute (str): The target attribute for which MSE is calculated.
    - true_values (pd.Series): Series containing true values corresponding to the target attribute.

    Return:
    GroupedMetric: Mean Squared Error metric for each group.

    Example:
    >>> mse_metric = mean_squared_err(df, ['Age', 'Gender'], 'Income', true_income_values)
    >>> print(repr(mse_metric))

    See Also:
    - accuracy: Calculates the Accuracy metric.

    """

    # Group the DataFrame based on quasi_identifiers
    grouped = df.groupby(quasi_identifiers)

    # Initialize an empty list to store mean squared error values for each group
    mse_values = []

    # Initialize an empty list to store group labels
    group_labels = []

    # Iterate over each group in the grouped DataFrame
    for group_name, group_df in grouped:
        # Generate a list of predicted values with the mean of the target attribute for the current group
        predicted_values = [group_df[target_attribute].mean()] * len(group_df)

        # Calculate mean squared error for the current group
        mse = mean_squared_error(true_values.loc[group_df.index], predicted_values)

        # Append the calculated mean squared error to the list
        mse_values.append(mse)

        # Append the group_name (quasi_identifiers values) to the group_labels list
        group_labels.append(group_name)

    # Create a GroupedMetric object with the calculated mean squared error values, group labels, and a name
    return GroupedMetric(np.array(mse_values), group_labels, name='Mean Squared Error')


def accuracy(df: pd.DataFrame, quasi_identifiers: list, target_attribute: str, true_values: pd.Series) -> GroupedMetric:
    """
    Calculate the Accuracy metric for a given DataFrame and quasi-identifiers.

    Synopse:
    Accuracy measures the proportion of correctly predicted instances.

    Details:
    The metric is computed by grouping the DataFrame based on quasi-identifiers, determining the mode of the target
    attribute within each group, and then calculating the accuracy using the mode as the predicted value.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - quasi_identifiers (list): List of column names representing quasi-identifiers.
    - target_attribute (str): The target attribute for which accuracy is calculated.
    - true_values (pd.Series): Series containing true values corresponding to the target attribute.

    Return:
    GroupedMetric: Accuracy metric for each group.

    Example:
    >>> accuracy_metric = accuracy(df, ['Age', 'Gender'], 'Income', true_income_values)
    >>> print(repr(accuracy_metric))

    See Also:
    - mean_squared_err: Calculates the Mean Squared Error metric.

    """

    # Group the DataFrame based on quasi_identifiers
    grouped = df.groupby(quasi_identifiers)

    # Initialize an empty list to store accuracy values for each group
    accuracy_values = []

    # Initialize an empty list to store group labels
    group_labels = []

    # Iterate over each group in the grouped DataFrame
    for group_name, group_df in grouped:
        # Generate a list of predicted values with the mode of the target attribute for the current group
        predicted_values = [group_df[target_attribute].mode().iloc[0]] * len(group_df)

        # Extract true values corresponding to the current group
        true_values_group = true_values.loc[group_df.index]

        # Calculate accuracy for the current group
        accuracy = accuracy_score(true_values_group, predicted_values)

        # Append the calculated accuracy to the list
        accuracy_values.append(accuracy)

        # Append the group_name (quasi_identifiers values) to the group_labels list
        group_labels.append(group_name)

    # Create a GroupedMetric object with the calculated accuracy values, group labels, and a name
    return GroupedMetric(np.array(accuracy_values), group_labels, name='Accuracy')


def utility_score(df: pd.DataFrame, quasi_identifiers: list, target_attribute: str, scoring_function: callable) -> GroupedMetric:
    """
    Calculate the Utility Score metric for a given DataFrame and quasi-identifiers using a custom scoring function.

    Synopse:
    Utility Score measures the utility of the data based on a user-defined scoring function.

    Details:
    The metric is computed by grouping the DataFrame based on quasi-identifiers and applying the user-defined scoring
    function to the target attribute within each group.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - quasi_identifiers (list): List of column names representing quasi-identifiers.
    - target_attribute (str): The target attribute for which utility is calculated.
    - scoring_function (callable): A scoring function that takes a pandas Series as input and returns a utility score.

    Return:
    GroupedMetric: Utility Score metric for each group.

    Example:
    >>> utility_metric = utility_score(df, ['Age', 'Gender'], 'Income', lambda x: x.sum())
    >>> print(repr(utility_metric))

    """

    # Group the DataFrame based on quasi_identifiers
    grouped = df.groupby(quasi_identifiers)

    # Initialize an empty list to store utility values for each group
    utility_values = []

    # Initialize an empty list to store group labels
    group_labels = []

    # Iterate over each group in the grouped DataFrame
    for group_name, group_df in grouped:
        # Calculate the utility for the current group using the specified scoring function
        group_utility = scoring_function(group_df[target_attribute])

        # Append the calculated utility to the list
        utility_values.append(group_utility)

        # Append the group_name (quasi_identifiers values) to the group_labels list
        group_labels.append(group_name)

    # Create a GroupedMetric object with the calculated utility values, group labels, and a name
    return GroupedMetric(np.array(utility_values), group_labels, name='Utility Score')


def range_utility(df: pd.DataFrame, quasi_identifiers: list, target_attribute: str) -> GroupedMetric:
    """
    Calculate the Range Utility metric for a given DataFrame and quasi-identifiers.

    Synopse:
    Range Utility measures the range (difference between max and min) of the target attribute.

    Details:
    The metric is computed by grouping the DataFrame based on quasi-identifiers and calculating the range of the target
    attribute within each group.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - quasi_identifiers (list): List of column names representing quasi-identifiers.
    - target_attribute (str): The target attribute for which range utility is calculated.

    Return:
    GroupedMetric: Range Utility metric for each group.

    Example:
    >>> range_utility_metric = range_utility(df, ['Age', 'Gender'], 'Income')
    >>> print(repr(range_utility_metric))

    See Also:
    - utility_score: Calculates the Utility Score metric.

    """

    # Group the DataFrame based on quasi_identifiers
    grouped = df.groupby(quasi_identifiers)

    # Initialize an empty list to store range utility values for each group
    range_values = []

    # Initialize an empty list to store group labels
    group_labels = []

    # Iterate over each group in the grouped DataFrame
    for group_name, group_df in grouped:
        # Calculate the range utility for the current group
        group_range = group_df[target_attribute].max() - group_df[target_attribute].min()

        # Append the calculated range utility to the list
        range_values.append(group_range)

        # Append the group_name (quasi_identifiers values) to the group_labels list
        group_labels.append(group_name)

    # Create a GroupedMetric object with the calculated range utility values, group labels, and a name
    return GroupedMetric(np.array(range_values), group_labels, name='Range Utility')


def distinct_values_utility(df: pd.DataFrame, quasi_identifiers: list, target_attribute: str) -> GroupedMetric:
    """
    Calculate the Distinct Values Utility metric for a given DataFrame and quasi-identifiers.

    Synopse:
    Distinct Values Utility measures the number of unique values in the target attribute.

    Details:
    The metric is computed by grouping the DataFrame based on quasi-identifiers and calculating the number of distinct
    values in the target attribute within each group.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - quasi_identifiers (list): List of column names representing quasi-identifiers.
    - target_attribute (str): The target attribute for which distinct values utility is calculated.

    Return:
    GroupedMetric: Distinct Values Utility metric for each group.

    Example:
    >>> distinct_values_metric = distinct_values_utility(df, ['Age', 'Gender'], 'Income')
    >>> print(repr(distinct_values_metric))

    See Also:
    - utility_score: Calculates the Utility Score metric.

    """

    # Group the DataFrame based on quasi_identifiers
    grouped = df.groupby(quasi_identifiers)

    # Initialize an empty list to store distinct values utility for each group
    distinct_values_values = []

    # Initialize an empty list to store group labels
    group_labels = []

    # Iterate over each group in the grouped DataFrame
    for group_name, group_df in grouped:
        # Calculate the number of distinct values in the target attribute for the current group
        distinct_values = group_df[target_attribute].nunique()

        # Append the calculated distinct values utility to the list
        distinct_values_values.append(distinct_values)

        # Append the group_name (quasi_identifiers values) to the group_labels list
        group_labels.append(group_name)

    # Create a GroupedMetric object with the calculated distinct values utility, group labels, and a name
    return GroupedMetric(np.array(distinct_values_values), group_labels, name='Distinct Values Utility')


def completeness_utility(df: pd.DataFrame, quasi_identifiers: list, target_attribute: str) -> GroupedMetric:
    """
    Calculate the Completeness Utility metric for a given DataFrame and quasi-identifiers.

    Synopse:
    Completeness Utility measures the proportion of instances with non-null values in the target attribute.

    Details:
    The metric is computed by grouping the DataFrame based on quasi-identifiers and calculating the completeness (ratio
    of non-null values) of the target attribute within each group.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - quasi_identifiers (list): List of column names representing quasi-identifiers.
    - target_attribute (str): The target attribute for which completeness utility is calculated.

    Return:
    GroupedMetric: Completeness Utility metric for each group.

    Example:
    >>> completeness_metric = completeness_utility(df, ['Age', 'Gender'], 'Income')
    >>> print(repr(completeness_metric))

    """

    # Group the DataFrame based on quasi_identifiers
    grouped = df.groupby(quasi_identifiers)

    # Initialize an empty list to store completeness utility values for each group
    completeness_values = []

    # Initialize an empty list to store group labels
    group_labels = []

    # Iterate over each group in the grouped DataFrame
    for group_name, group_df in grouped:
        # Calculate completeness (ratio of non-null values) of the target attribute for the current group
        completeness = group_df[target_attribute].count() / len(group_df)

        # Append the calculated completeness utility to the list
        completeness_values.append(completeness)

        # Append the group_name (quasi_identifiers values) to the group_labels list
        group_labels.append(group_name)

    # Create a GroupedMetric object with the calculated completeness utility values, group labels, and a name
    return GroupedMetric(np.array(completeness_values), group_labels, name='Completeness Utility')


def attr_length_penalty(quasi_identifiers: list, attributes: list):
    num_attributes = len(quasi_identifiers)
    proportion = num_attributes / len(attributes)

    return (1 - proportion) ** 2 + proportion ** 2
