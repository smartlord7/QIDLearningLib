import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

from structure.GroupedMetric import GroupedMetric


def mean_squared_err(df, quasi_identifiers, target_attribute, true_values):
    grouped = df.groupby(quasi_identifiers)
    mse_values = []

    for _, group_df in grouped:
        predicted_values = [group_df[target_attribute].mean()] * len(group_df)
        mse = mean_squared_error(true_values.loc[group_df.index], predicted_values)
        mse_values.append(mse)

    group_labels = [tuple(group) for group in grouped.indices.values()]

    return GroupedMetric(np.array(mse_values), group_labels, name='Mean Squared Error')


def accuracy(df, quasi_identifiers, target_attribute, true_values):
    grouped = df.groupby(quasi_identifiers)
    accuracy_values = []

    for _, group_df in grouped:
        predicted_values = [group_df[target_attribute].mode().iloc[0]] * len(group_df)
        true_values_group = true_values.loc[group_df.index]
        accuracy = accuracy_score(true_values_group, predicted_values)
        accuracy_values.append(accuracy)

    group_labels = [tuple(group) for group in grouped.indices.values()]

    return GroupedMetric(np.array(accuracy_values), group_labels, name='Accuracy')

def utility_score(df, quasi_identifiers, target_attribute, scoring_function):
    grouped = df.groupby(quasi_identifiers)
    utility_values = []
    group_labels = []

    for group_name, group_df in grouped:
        group_utility = scoring_function(group_df[target_attribute])
        utility_values.append(group_utility)
        group_labels.append(group_name)

    return GroupedMetric(np.array(utility_values), group_labels, name='Utility Score')


def range_utility(df, quasi_identifiers, target_attribute):
    grouped = df.groupby(quasi_identifiers)
    range_values = []

    for _, group_df in grouped:
        group_range = group_df[target_attribute].max() - group_df[target_attribute].min()
        range_values.append(group_range)

    group_labels = [tuple(group) for group in grouped.indices.values()]

    return GroupedMetric(np.array(range_values), group_labels, name='Range Utility')

def distinct_values_utility(df, quasi_identifiers, target_attribute):
    grouped = df.groupby(quasi_identifiers)
    distinct_values_values = []

    for _, group_df in grouped:
        distinct_values = group_df[target_attribute].nunique()
        distinct_values_values.append(distinct_values)

    group_labels = [tuple(group) for group in grouped.indices.values()]

    return GroupedMetric(np.array(distinct_values_values), group_labels, name='Distinct Values Utility')

def completeness_utility(df, quasi_identifiers, target_attribute):
    grouped = df.groupby(quasi_identifiers)
    completeness_values = []

    for _, group_df in grouped:
        completeness = group_df[target_attribute].count() / len(group_df)
        completeness_values.append(completeness)

    group_labels = [tuple(group) for group in grouped.indices.values()]

    return GroupedMetric(np.array(completeness_values), group_labels, name='Completeness Utility')