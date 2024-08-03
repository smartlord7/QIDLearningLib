"""
QIDLearningLib

Library Description:
QIDLearningLib is a Python library designed to provide a vast set of metrics for quasi-identifiers recognition processes.
The library includes metrics for assessing data privacy, data utility, and the performance of quasi-identifiers recognition algorithms.

Module Description (metrics.data_privacy):
This module in QIDLearningLib includes functions to calculate various metrics related to the data privacy regarding the assumed quasi identifiers and/or sensitive attributes.

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
from typing import Iterable, Any
from scipy.special import kl_div
from scipy.stats import wasserstein_distance
from typing_extensions import deprecated

from structure.grouped_metric import GroupedMetric


def k_anonymity(df: pd.DataFrame, quasi_identifiers: Iterable[str]) -> GroupedMetric:
    """
        Calculate the k-Anonymity metric for a given DataFrame and quasi-identifiers.

        Synopsis:
        Measures the group sizes of quasi-identifiers in a dataset. In an anonymization context it confirms that each group has at least
        k identical instances.

        Details:
        The metric is computed by grouping the DataFrame based on quasi-identifiers and determining the size of each group.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - quasi_identifiers (iterable): Iterable of column names representing quasi-identifiers.

        Return:
        GroupedMetric: k-Anonymity metric for each group.

        Example:
        >>> k_anonymity_metric = k_anonymity(df, ['Age', 'Gender'])
        >>> print(repr(k_anonymity_metric))
        """
    structured_array = df[quasi_identifiers].astype(str).to_records(index=False)
    unique_groups, counts = np.unique(structured_array, return_counts=True)

    return GroupedMetric(counts, unique_groups.tolist(), name='k-Anonymity')


def l_diversity(df: pd.DataFrame, quasi_identifiers: Iterable[str],
                sensitive_attributes: Iterable[str]) -> GroupedMetric:
    """
    Calculate the l-Diversity metric for a given DataFrame, quasi-identifiers, and sensitive attribute.

    Synopsis:
    l-Diversity measures the minimum number of unique values in the sensitive attribute within each group defined by 
    quasi-identifiers.

    Details:
    The metric is computed by grouping the DataFrame based on quasi-identifiers and counting the unique values in the
    sensitive attribute for each group.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - quasi_identifiers (iterable): Iterable of column names representing quasi-identifiers.
    - sensitive_attributes (iterable): The sensitive attribute for which l-Diversity is calculated.

    Return:
    GroupedMetric: l-Diversity metric for each group.

    Example:
    >>> l_diversity_metric = l_diversity(df, ['Age', 'Gender'], ['Disease'])
    >>> print(repr(l_diversity_metric))
    """

    # Group the DataFrame based on quasi_identifiers
    grouped = df.groupby(quasi_identifiers)

    # Initialize an empty list to store l-Diversity values for each group
    unique_values = []

    # Get group labels
    group_labels = [group_name for group_name, _ in grouped]

    # Iterate over each group in the grouped DataFrame
    for group_name, group_df in grouped:
        # Count the unique values in each sensitive attribute for the current group
        unique_values.extend(group_df[sensitive_attributes].nunique().values)

    # Create a GroupedMetric object with the calculated l-Diversity values, group labels, and a name
    return GroupedMetric(np.array(unique_values), group_labels, name='l-Diversity')


def closeness_centrality(df: pd.DataFrame, quasi_identifiers: Iterable[str],
                         sensitive_attributes: Iterable[str]) -> GroupedMetric:
    """
        Calculate the Closeness Centrality metric for a given DataFrame, quasi-identifiers, and sensitive attribute.

        Synopsis:
        Closeness Centrality measures the Wasserstein distance between the overall distribution of the sensitive attribute
        and the distribution within each group defined by quasi-identifiers.

        Details:
        The metric is computed by grouping the DataFrame based on quasi-identifiers and calculating the Wasserstein distance
        between the overall distribution and the distribution within each group.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - quasi_identifiers (iterable): Iterable of column names representing quasi-identifiers.
        - sensitive_attributes (iterable): The sensitive attribute for which Closeness Centrality is calculated.

        Return:
        GroupedMetric: Closeness Centrality metric for each group.

        Example:
        >>> closeness_centrality_metric = closeness_centrality(df, ['Age', 'Gender'], ['Income'])
        >>> print(repr(closeness_centrality_metric))
        """

    # Calculate the overall distribution
    overall_counts = df[sensitive_attributes].value_counts(normalize=True)
    overall_values = overall_counts.values
    overall_index = overall_counts.index

    # Create a mapping from overall index to numeric values if needed
    if not all(isinstance(x, (int, float)) for x in overall_index):
        category_mapping = {val: i for i, val in enumerate(overall_index)}
        overall_index_numeric = np.array([category_mapping[val] for val in overall_index], dtype=float)
    else:
        overall_index_numeric = np.array(overall_index, dtype=float)

    # Convert category mapping to NumPy array
    category_mapping = {val: i for i, val in enumerate(overall_index)}

    # Group the DataFrame based on quasi-identifiers
    grouped = df.groupby(quasi_identifiers)

    # Get group labels
    group_labels = [group_name for group_name, _ in grouped]

    # List to hold all closeness values
    closeness_values = []

    for group_name, group_df in grouped:
        # Compute the group's distribution
        group_counts = group_df[sensitive_attributes].value_counts(normalize=True)
        group_values = group_counts.values
        group_index = group_counts.index

        # Convert group index to numeric using the same mapping
        if not all(isinstance(val, (int, float)) for val in group_index):
            group_index_numeric = np.array([category_mapping.get(val, np.nan) for val in group_index], dtype=float)
        else:
            group_index_numeric = np.array(group_index, dtype=float)

        # Compute Wasserstein distance
        closeness = wasserstein_distance(
            overall_index_numeric, group_index_numeric, overall_values, group_values
        )

        # Append results
        closeness_values.append(closeness)

    return GroupedMetric(np.array(closeness_values), group_labels, name='Closeness Centrality')


def delta_presence(df: pd.DataFrame, quasi_identifiers: Iterable[str], values: Iterable[Any]) -> GroupedMetric:
    """
    Calculate the Delta Presence metric for a given DataFrame, quasi-identifier, and value.

    Synopsis:
    Delta Presence measures the absolute difference in the presence of a specific value in the quasi-identifier across
    the entire dataset and within each group.

    Details:
    The metric is computed by comparing the presence of a specific value in the quasi-identifier across the entire dataset
    and within each group defined by the quasi-identifier.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - quasi_identifier (str): The quasi-identifier for which Delta Presence is calculated.
    - value: The value for which Delta Presence is calculated.

    Return:
    GroupedMetric: Delta Presence metric for each group.

    Example:
    >>> delta_presence_metric = delta_presence(df, ['Age'], [25, ])
    >>> print(repr(delta_presence_metric))
    """

    # Convert the quasi_identifiers and values to a tuple for consistency
    quasi_identifiers = quasi_identifiers
    values = tuple(values)

    # Create a tuple version of the quasi_identifiers for comparison
    df_tuples = df[quasi_identifiers].apply(tuple, axis=1)

    # Calculate overall presence of the values in the quasi_identifiers
    overall_presence = np.mean(df_tuples == values)

    # Group the DataFrame by the quasi_identifiers
    grouped = df.groupby(list(quasi_identifiers))

    # Calculate group-wise presence
    group_presences = grouped.apply(lambda group: np.mean(df_tuples[group.index] == values))

    # Calculate Delta Presence for each group
    delta_presences = np.abs(overall_presence - group_presences)

    # Get group labels
    group_labels = [group_name for group_name, _ in grouped]

    # Create a GroupedMetric object with the calculated Delta Presence values, group labels, and a name
    return GroupedMetric(np.array(delta_presences), group_labels, name='Delta Presence')


def t_closeness(df: pd.DataFrame, quasi_identifiers: Iterable[str],
                sensitive_attributes: Iterable[str]) -> GroupedMetric:
    """
    Calculate the t-Closeness metric for a given DataFrame, quasi-identifiers, and sensitive attribute.

    Synopsis:
    t-Closeness measures the Kullback-Leibler (KL) divergence between the distribution of the sensitive attribute within
    each group defined by quasi-identifiers and the overall distribution.

    Details:
    The metric is computed by grouping the DataFrame based on quasi-identifiers and calculating the KL divergence for
    each group.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - quasi_identifiers (iterable): Iterable of column names representing quasi-identifiers.
    - sensitive_attributes (iterable): The sensitive attribute for which t-Closeness is calculated.

    Return:
    GroupedMetric: t-Closeness metric for each group.

    Example:
    >>> t_closeness_metric = t_closeness(df, ['Age', 'Gender'], ['Income'])
    >>> print(repr(t_closeness_metric))
    """

    # Get overall distribution of the sensitive attribute
    overall_counts = df[sensitive_attributes].apply(lambda x: tuple(x)).value_counts(normalize=True)
    overall_index = overall_counts.index
    overall_distribution = overall_counts.reindex(overall_index, fill_value=0).values

    grouped = df.groupby(quasi_identifiers)

    # Create a DataFrame with quasi-identifiers as index and sensitive attributes as columns
    df_grouped = grouped.apply(
        lambda g: g[sensitive_attributes].apply(lambda x: tuple(x)).value_counts(normalize=True)
    ).unstack(fill_value=0)

    # Reindex to ensure all sensitive attribute tuples are present in all groups
    df_grouped = df_grouped.reindex(columns=overall_index, fill_value=0).fillna(0)

    # Convert to numpy arrays for efficient computation
    group_distributions = df_grouped.to_numpy()
    overall_distribution = np.expand_dims(overall_distribution, axis=0)  # Make it 2D for broadcasting

    # Calculate KL divergence for each group
    kl_divergences = np.sum(kl_div(group_distributions, overall_distribution), axis=1)

    # Get group labels
    group_labels = [group_name for group_name, _ in grouped]

    return GroupedMetric(np.array(kl_divergences), group_labels, name='t-Closeness')


def generalization_ratio(df: pd.DataFrame, quasi_identifiers: Iterable[str],
                         sensitive_attributes: Iterable[str]) -> GroupedMetric:
    """
    Calculate the Generalization Ratio metric for a given DataFrame and quasi-identifiers.

    Synopse:
    Generalization Ratio measures the overall difference in the distribution of sensitive attributes between the entire
    dataset and its grouped subsets based on quasi-identifiers.

    Details:
    The metric is computed by grouping the DataFrame based on quasi-identifiers and calculating the sum of absolute
    differences in the distribution of sensitive attributes between the overall dataset and each group.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - quasi_identifiers (list): List of column names representing quasi-identifiers.
    - sensitive_attributes (list): List of column names representing sensitive attributes.

    Return:
    GroupedMetric: Generalization Ratio metric for each group.

    Example:
    >>> generalization_ratio_metric = t_closeness(df, ['Age', 'Gender'], ['Income'])
    >>> print(repr(generalization_ratio_metric))
    """

    # Calculate the overall distribution of sensitive attributes
    overall_distribution = df[sensitive_attributes].value_counts(normalize=True).sort_index()

    # Prepare for grouping
    grouped = df.groupby(quasi_identifiers)

    # Prepare a DataFrame to store distributions for all groups
    distributions = []

    # Create a dictionary to map group names to their distributions
    group_dict = {}

    for group_name, group_df in grouped:
        group_distribution = group_df[sensitive_attributes].value_counts(normalize=True).sort_index()
        group_dict[group_name] = group_distribution

    # Align distributions with the overall distribution
    index = overall_distribution.index
    aligned_distributions = {}

    for group_name, group_distribution in group_dict.items():
        aligned_group_distribution = group_distribution.reindex(index, fill_value=0)
        aligned_distributions[group_name] = aligned_group_distribution

    # Convert the distributions to a DataFrame
    distributions_df = pd.DataFrame(aligned_distributions).T.fillna(0).reindex(columns=index, fill_value=0)

    # Calculate the absolute differences and sum them up
    overall_distribution_array = overall_distribution.values
    diffs = np.abs(distributions_df.values - overall_distribution_array).sum(axis=1)

    # Prepare results for GroupedMetric
    group_labels = [group_name for group_name, _ in grouped]

    return GroupedMetric(np.array(diffs), group_labels, name='Generalization Ratio')


def reciprocal_rank(df: pd.DataFrame, quasi_identifiers: Iterable[str],
                    sensitive_attributes: Iterable[str]) -> GroupedMetric:
    """
    Calculate the Reciprocal Rank metric for a given DataFrame and quasi-identifiers.

    Synopse:
    Reciprocal Rank measures the quality of rankings based on the inverse of the rank position of the correct sensitive
    attribute value within each group.

    Details:
    The metric is computed by grouping the DataFrame based on quasi-identifiers, ranking the sensitive attribute values
    in descending order within each group, and then calculating the reciprocal of the rank of the correct value.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - quasi_identifiers (iterable): Iterable of column names representing quasi-identifiers.
    - sensitive_attributes (iterable): Iterable of column names representing sensitive attributes.

    Return:
    GroupedMetric: Reciprocal Rank metric for each group.

    >>> reciprocal_rank_metric = t_closeness(df, ['Age', 'Gender'], ['Income'])
    >>> print(repr(reciprocal_rank_metric))
    """

    grouped = df.groupby(quasi_identifiers)

    # Rank the sensitive attributes in descending order within each group
    ranks = grouped[sensitive_attributes].rank(ascending=False)

    # Calculate the reciprocal ranks
    reciprocal_ranks = 1 / ranks
    mrr_values = reciprocal_ranks.values.reshape(-1)

    group_labels = [group_name for group_name, _ in grouped]

    # Create a GroupedMetric object with the calculated Reciprocal Rank values, group labels, and a name
    return GroupedMetric(mrr_values, group_labels, name='Reciprocal Rank')


@deprecated("Deprecated")
def k_anonymity_old(df, quasi_identifiers):
    """
    Calculate the k-Anonymity metric for a given DataFrame and quasi-identifiers.

    Synopsis:
    k-Anonymity measures the minimum group size of quasi-identifiers in a dataset, ensuring that each group has at least
    k identical instances.

    Details:
    The metric is computed by grouping the DataFrame based on quasi-identifiers and determining the size of each group.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - quasi_identifiers (list): List of column names representing quasi-identifiers.

    Return:
    GroupedMetric: k-Anonymity metric for each group.

    Example:
    >>> k_anonymity_metric = k_anonymity(df, ['Age', 'Gender'])
    >>> print(repr(k_anonymity_metric))
    """

    # Group the DataFrame based on quasi_identifiers
    grouped = df.groupby(quasi_identifiers)

    # Extract group labels from the grouped DataFrame
    group_labels = [group_name for group_name, _ in grouped]

    # Create a GroupedMetric object with the calculated k-Anonymity values, group labels, and a name
    return GroupedMetric(np.array(grouped.size()), group_labels, name='k-Anonymity', lazy_calc=False)


@deprecated("Deprecated")
def l_diversity_old(df: pd.DataFrame, quasi_identifiers: list, sensitive_attributes: list) -> GroupedMetric:
    """
    Calculate the l-Diversity metric for a given DataFrame, quasi-identifiers, and sensitive attribute.

    Synopsis:
    l-Diversity measures the minimum number of unique values in the sensitive attribute within each group defined by
    quasi-identifiers.

    Details:
    The metric is computed by grouping the DataFrame based on quasi-identifiers and counting the unique values in the
    sensitive attribute for each group.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - quasi_identifiers (list): List of column names representing quasi-identifiers.
    - sensitive_attributes (str): The sensitive attribute for which l-Diversity is calculated.

    Return:
    GroupedMetric: l-Diversity metric for each group.

    Example:
    >>> l_diversity_metric = l_diversity(df, ['Age', 'Gender'], ['Disease'])
    >>> print(repr(l_diversity_metric))
    """

    # Group the DataFrame based on quasi_identifiers
    grouped = df.groupby(quasi_identifiers)

    # Initialize an empty list to store l-Diversity values for each group
    unique_values = []

    # Get group labels
    group_labels = [group_name for group_name, _ in grouped]

    # Iterate over each group in the grouped DataFrame
    for group_name, group_df in grouped:
        # Count the unique values in each sensitive attribute for the current group
        unique_values.extend(group_df[sensitive_attributes].nunique().values)

    # Create a GroupedMetric object with the calculated l-Diversity values, group labels, and a name
    return GroupedMetric(np.array(unique_values), group_labels, name='l-Diversity', lazy_calc=False)


@deprecated("Deprecated")
def closeness_centrality_old(df: pd.DataFrame, quasi_identifiers: list, sensitive_attributes: list):
    """
    Calculate the Closeness Centrality metric for a given DataFrame, quasi-identifiers, and sensitive attribute.

    Synopsis:
    Closeness Centrality measures the Wasserstein distance between the overall distribution of the sensitive attribute
    and the distribution within each group defined by quasi-identifiers.

    Details:
    The metric is computed by grouping the DataFrame based on quasi-identifiers and calculating the Wasserstein distance
    between the overall distribution and the distribution within each group.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - quasi_identifiers (list): List of column names representing quasi-identifiers.
    - sensitive_attributes (str): The sensitive attribute for which Closeness Centrality is calculated.

    Return:
    GroupedMetric: Closeness Centrality metric for each group.

    Example:
    >>> closeness_centrality_metric = closeness_centrality(df, ['Age', 'Gender'], ['Income'])
    >>> print(repr(closeness_centrality_metric))
    """

    # Calculate the overall distribution of the sensitive attribute
    overall_distribution = df[sensitive_attributes].value_counts(normalize=True)

    # Group the DataFrame based on quasi_identifiers
    grouped = df.groupby(quasi_identifiers)

    # Initialize an empty list to store Closeness Centrality values for each group
    closeness_values = []

    # Get group labels
    group_labels = [group_name for group_name, _ in grouped]

    # Iterate over each group in the grouped DataFrame
    for group_name, group_df in grouped:
        # Calculate the distribution of the sensitive attribute within the current group
        group_distribution = group_df[sensitive_attributes].value_counts(normalize=True)

        # Check if the values in group_distribution are numeric
        if all(isinstance(val, (int, float)) for val in group_distribution.index):
            closeness = wasserstein_distance(overall_distribution.index.astype(float),
                                             group_distribution.index.astype(float), overall_distribution.values,
                                             group_distribution.values)
        else:
            # Convert categorical values to numerical indices
            category_mapping = {val: i for i, val in enumerate(overall_distribution.index)}
            overall_distribution_numeric = overall_distribution.index.map(category_mapping)
            group_distribution_numeric = group_distribution.index.map(category_mapping)
            closeness = wasserstein_distance(overall_distribution_numeric, group_distribution_numeric,
                                             overall_distribution.values, group_distribution.values)

        closeness_values.append(closeness)

    # Create a GroupedMetric object with the calculated Closeness Centrality values, group labels, and a name
    return GroupedMetric(np.array(closeness_values), group_labels, name='Closeness Centrality', lazy_calc=False)


@deprecated("Deprecated")
def delta_presence_old(df: pd.DataFrame, quasi_identifiers: list, values: list) -> GroupedMetric:
    """
    Calculate the Delta Presence metric for a given DataFrame, quasi-identifier, and value.

    Synopsis:
    Delta Presence measures the absolute difference in the presence of a specific value in the quasi-identifier across
    the entire dataset and within each group.

    Details:
    The metric is computed by comparing the presence of a specific value in the quasi-identifier across the entire dataset
    and within each group defined by the quasi-identifier.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - quasi_identifier (str): The quasi-identifier for which Delta Presence is calculated.
    - value: The value for which Delta Presence is calculated.

    Return:
    GroupedMetric: Delta Presence metric for each group.

    Example:
    >>> delta_presence_metric = delta_presence(df, 'Age', 25)
    >>> print(repr(delta_presence_metric))
    """

    # Calculate the overall presence of the values in the quasi_identifiers across the entire dataset
    overall_presence = df[quasi_identifiers].value_counts(normalize=True).get(tuple(values), 0)

    # Group the DataFrame based on quasi_identifier
    grouped = df.groupby(quasi_identifiers)

    # Initialize an empty list to store Delta Presence values for each group
    delta_presence_values = []

    # Get group labels
    group_labels = [group_name for group_name, _ in grouped]

    # Iterate over each group in the grouped DataFrame
    for group_name, group_df in grouped:
        # Calculate the presence of the value in the quasi_identifier within the current group
        group_presence = group_df[quasi_identifiers].value_counts(normalize=True).get(tuple(values), 0)

        # Calculate Delta Presence for the current group
        delta_presence = abs(overall_presence - group_presence)
        delta_presence_values.append(delta_presence)

    # Create a GroupedMetric object with the calculated Delta Presence values, group labels, and a name
    return GroupedMetric(np.array(delta_presence_values), group_labels, name='Delta Presence', lazy_calc=False)


@deprecated("Deprecated")
def t_closeness_old(df: pd.DataFrame, quasi_identifiers: list, sensitive_attributes: list):
    """
    Calculate the t-Closeness metric for a given DataFrame, quasi-identifiers, and sensitive attribute.

    Synopsis:
    t-Closeness measures the Kullback-Leibler (KL) divergence between the distribution of the sensitive attribute within
    each group defined by quasi-identifiers and the overall distribution.

    Details:
    The metric is computed by grouping the DataFrame based on quasi-identifiers and calculating the KL divergence for
    each group.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - quasi_identifiers (list): List of column names representing quasi-identifiers.
    - sensitive_attributes (str): The sensitive attribute for which t-Closeness is calculated.

    Return:
    GroupedMetric: t-Closeness metric for each group.

    Example:
    >>> t_closeness_metric = t_closeness(df, ['Age', 'Gender'], ['Income'])
    >>> print(repr(t_closeness_metric))
    """

    # Calculate the overall distribution of the sensitive attribute
    overall_distribution = df[sensitive_attributes].apply(lambda x: tuple(x)).value_counts(normalize=True)

    # Group the DataFrame based on quasi_identifiers
    grouped = df.groupby(quasi_identifiers)

    # Initialize an empty list to store t-Closeness values for each group
    t_closeness_values = []

    # Iterate over each group in the grouped DataFrame
    for group_name, group_df in grouped:
        # Calculate the distribution of the sensitive attribute within the current group
        group_distribution = group_df[sensitive_attributes].apply(lambda x: tuple(x)).value_counts(normalize=True)

        # Compute KL divergence between the overall distribution and the distribution within the group
        kl_divergence = sum(
            overall_distribution.get(value, 0) * np.log(overall_distribution.get(value, 1e-10) /
                                                        (group_distribution.get(value, 1e-10)))
            for value in overall_distribution.index
        )
        t_closeness_values.append(kl_divergence)

    group_labels = [group_name for group_name, _ in grouped]

    # Create a GroupedMetric object with the calculated t-Closeness values, group labels, and a name
    return GroupedMetric(np.array(t_closeness_values), group_labels, name='t-Closeness', lazy_calc=False)


@deprecated("Deprecated")
def generalization_ratio_old(df: pd.DataFrame, quasi_identifiers: Iterable[str],
                             sensitive_attributes: Iterable[str]) -> GroupedMetric:
    """
    Calculate the Generalization Ratio metric for a given DataFrame and quasi-identifiers.

    Synopse:
    Generalization Ratio measures the overall difference in the distribution of sensitive attributes between the entire
    dataset and its grouped subsets based on quasi-identifiers.

    Details:
    The metric is computed by grouping the DataFrame based on quasi-identifiers and calculating the sum of absolute
    differences in the distribution of sensitive attributes between the overall dataset and each group.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - quasi_identifiers (list): List of column names representing quasi-identifiers.
    - sensitive_attributes (list): List of column names representing sensitive attributes.

    Return:
    GroupedMetric: Generalization Ratio metric for each group.

    Example:
    >>> generalization_ratio_metric = t_closeness(df, ['Age', 'Gender'], ['Income'])
    >>> print(repr(generalization_ratio_metric))
    """

    # Calculate the overall distribution of sensitive attributes
    overall_distribution = df[sensitive_attributes].value_counts(normalize=True).sort_index()

    # Prepare for grouping
    grouped = df.groupby(quasi_identifiers)

    # Prepare a DataFrame to store distributions for all groups
    distributions = []

    # Create a dictionary to map group names to their distributions
    group_dict = {}

    for group_name, group_df in grouped:
        group_distribution = group_df[sensitive_attributes].value_counts(normalize=True).sort_index()
        group_dict[group_name] = group_distribution

    # Align distributions with the overall distribution
    index = overall_distribution.index
    aligned_distributions = {}

    for group_name, group_distribution in group_dict.items():
        aligned_group_distribution = group_distribution.reindex(index, fill_value=0)
        aligned_distributions[group_name] = aligned_group_distribution

    # Convert the distributions to a DataFrame
    distributions_df = pd.DataFrame(aligned_distributions).T.fillna(0).reindex(columns=index, fill_value=0)

    # Calculate the absolute differences and sum them up
    overall_distribution_array = overall_distribution.values
    diffs = np.abs(distributions_df.values - overall_distribution_array).sum(axis=1)

    # Prepare results for GroupedMetric
    group_labels = [group_name for group_name, _ in grouped]

    return GroupedMetric(np.array(diffs), group_labels, name='Generalization Ratio', lazy_calc=False)
