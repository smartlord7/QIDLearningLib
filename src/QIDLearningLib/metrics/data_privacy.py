import numpy as np
from scipy.stats import wasserstein_distance
from scipy.stats import entropy as scipy_entropy

from structure.GroupedMetric import GroupedMetric


def k_anonymity(df, quasi_identifiers):
    grouped = df.groupby(quasi_identifiers)
    group_labels = [group_name for group_name, _ in grouped]

    return GroupedMetric(np.array(grouped.size()), group_labels, name='k-Anonymity')

def l_diversity(df, quasi_identifiers, sensitive_attribute):
    grouped = df.groupby(quasi_identifiers)
    unique_values = []

    for group_name, group_df in grouped:
        unique_values.append(group_df[sensitive_attribute].nunique())

    group_labels = [group_name for group_name, _ in grouped]

    return GroupedMetric(np.array(unique_values), group_labels, name='l-Diversity')

def closeness_centrality(df, quasi_identifiers, sensitive_attribute):
    overall_distribution = df[sensitive_attribute].value_counts(normalize=True)
    grouped = df.groupby(quasi_identifiers)

    closeness_values = []
    for group_name, group_df in grouped:
        group_distribution = group_df[sensitive_attribute].value_counts(normalize=True)

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

    group_labels = [group_name for group_name, _ in grouped]

    return GroupedMetric(np.array(closeness_values), group_labels, name='Closeness Centrality')

def delta_presence(df, quasi_identifier, value):
    overall_presence = df[quasi_identifier].value_counts(normalize=True).get(value, 0)
    grouped = df.groupby(quasi_identifier)

    delta_presence_values = []
    for group_name, group_df in grouped:
        group_presence = group_df[quasi_identifier].value_counts(normalize=True).get(value, 0)
        delta_presence = abs(overall_presence - group_presence)
        delta_presence_values.append(delta_presence)

    group_labels = [group_name for group_name, _ in grouped]

    return GroupedMetric(np.array(delta_presence_values), group_labels, name='Delta Presence')

def t_closeness(df, quasi_identifiers, sensitive_attribute):
    overall_distribution = df[sensitive_attribute].value_counts(normalize=True)
    grouped = df.groupby(quasi_identifiers)

    t_closeness_values = []
    for group_name, group_df in grouped:
        group_distribution = group_df[sensitive_attribute].value_counts(normalize=True)

        # Create a union of unique values from both distributions
        all_values = set(overall_distribution.index).union(group_distribution.index)

        # Fill missing values with zero probability
        overall_distribution = overall_distribution.reindex(all_values, fill_value=0)
        group_distribution = group_distribution.reindex(all_values, fill_value=0)

        # Calculate KL divergence using the entropy function
        kl_divergence = scipy_entropy(group_distribution, qk=overall_distribution)

        t_closeness_values.append(kl_divergence)

    group_labels = [group_name for group_name, _ in grouped]

    return GroupedMetric(np.array(t_closeness_values), group_labels, name='t-Closeness')

def generalization_ratio(df, quasi_identifiers, sensitive_attribute):
    overall_distribution = df[sensitive_attribute].value_counts(normalize=True)
    grouped = df.groupby(quasi_identifiers)

    generalization_ratio_values = []
    for group_name, group_df in grouped:
        group_distribution = group_df[sensitive_attribute].value_counts(normalize=True)
        generalization_ratio_values.append(np.sum(np.abs(overall_distribution - group_distribution)))

    group_labels = [group_name for group_name, _ in grouped]

    return GroupedMetric(np.array(generalization_ratio_values), group_labels, name='Generalization Ratio')

def reciprocal_rank(df, quasi_identifiers, sensitive_attribute):
    ranks = df.groupby(quasi_identifiers)[sensitive_attribute].rank(ascending=False)
    reciprocal_ranks = 1 / ranks
    mrr_values = reciprocal_ranks.values

    group_labels = [group_name for group_name, _ in df.groupby(quasi_identifiers)]

    return GroupedMetric(mrr_values, group_labels, name='Reciprocal Rank')
