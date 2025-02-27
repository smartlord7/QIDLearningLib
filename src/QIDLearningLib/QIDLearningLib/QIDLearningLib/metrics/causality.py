"""
QIDLearningLib

Library Description (QIDLearningLib):
QIDLearningLib is a Python library designed to provide a vast set of metrics for quasi-identifiers recognition processes.
The library includes metrics for assessing data privacy, data utility, and the performance of quasi-identifiers recognition algorithms.

Module Description (metrics.causality):
This module in QIDLearningLib includes functions to calculate various causality metrics in order to study the causal effect
between the assumed quasi identifiers and the remaining attributes.
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
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from QIDLearningLib.structure.grouped_metric import GroupedMetric
from QIDLearningLib.util.data import encode_categorical
from QIDLearningLib.util.stats import t_test, ks_test



def covariate_shift(df, quasi_identifiers, treatment_col, treatment_value):
    """
    Calculate covariate shift metric for a given DataFrame and attributes.

    Synopse:
    Covariate shift metric measures the difference in distribution between treated and control groups based on specified attributes.

    Details:
    The metric is computed by comparing the distribution of attribute values between treated and control groups using the Kolmogorov-Smirnov statistic. Additionally, it considers the overall distribution difference across the entire DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - quasi_identifiers (list): List of column names representing the quasi-identifiers.
    - treatment_col (str): The column name representing the treatment indicator.
    - treatment_value: The value indicating the treated group.

    Return:
    GroupedMetric: An object containing covariate shift metric values and corresponding group labels.

    Example:
    >>> df = pd.DataFrame({'A': [1, 2, 1, 2], 'B': ['X', 'Y', 'X', 'Y'], 'Treatment': [1, 0, 1, 0]})
    >>> metric_result = covariate_shift(df, ['A', 'B'], 'Treatment', 1)
    >>> print(metric_result.values)
    [0.1, 0.2, ...]  # Covariate shift metric values
    >>> print(metric_result.group_labels)
    ['Group 1', 'Group 2', ...]  # Corresponding group labels

    """
    treated = df[df[treatment_col] == treatment_value]
    control = df[df[treatment_col] != treatment_value]

    overall_distribution = df[quasi_identifiers].value_counts(normalize=True)

    grouped_treated = treated.groupby(quasi_identifiers)
    grouped_control = control.groupby(quasi_identifiers)

    shift_metrics = []
    group_labels = []

    # Flatten data and encode categorical columns
    df_flat_encoded = encode_categorical(df[quasi_identifiers].reset_index(drop=True), quasi_identifiers)

    for group_name, group_df_treated in grouped_treated:
        # Check if the control group also has data for the same group_name
        if group_name in grouped_control.groups:
            group_df_control = grouped_control.get_group(group_name)

            # Flatten data and encode categorical columns in treated and control groups
            group_df_treated_flat_encoded = encode_categorical(group_df_treated.reset_index(drop=True), quasi_identifiers)
            group_df_control_flat_encoded = encode_categorical(group_df_control.reset_index(drop=True), quasi_identifiers)

            # Check if the groups have data
            if not group_df_treated_flat_encoded.empty and not group_df_control_flat_encoded.empty:
                # Calculate overall distribution difference
                overall_diff = np.sum(
                    np.abs(overall_distribution - group_df_treated_flat_encoded.value_counts(normalize=True, sort=False)))

                # Calculate covariate shift metric
                ks_statistic, _ = ks_test(
                    df_flat_encoded.loc[group_df_treated_flat_encoded.index].values.flatten(),
                    df_flat_encoded.loc[group_df_control_flat_encoded.index].values.flatten()
                )

                shift_metrics.append(ks_statistic + overall_diff)
                group_labels.append(group_name)

    values = np.array(shift_metrics)
    return GroupedMetric(values, group_labels, name='Covariate Shift')


def balance_test(df, quasi_identifiers, treatment_col, treatment_value):
    """
    Perform balance test for a given DataFrame and attributes.

    Synopse:
    Balance test measures the balance between treated and control groups based on specified quasi-identifiers.

    Details:
    The metric is computed by applying a t-test to each group defined by the quasi-identifiers, comparing the distribution of values between treated and control groups.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - quasi_identifiers (list): List of column names representing the quasi-identifiers.
    - treatment_col (str): The column name representing the treatment indicator.
    - treatment_value: The value indicating the treated group.

    Return:
    GroupedMetric: An object containing balance test metric values and corresponding group labels.

    Example:
    >>> df = pd.DataFrame({'A': [1, 2, 1, 2], 'B': ['X', 'Y', 'X', 'Y'], 'Treatment': [1, 0, 1, 0]})
    >>> metric_result = balance_test(df, ['A', 'B'], 'Treatment', 1)
    >>> print(metric_result.values)
    [0.1, 0.2, ...]  # Balance test metric values
    >>> print(metric_result.group_labels)
    ['Group 1', 'Group 2', ...]  # Corresponding group labels

    """
    treated = df[df[treatment_col] == treatment_value]
    control = df[df[treatment_col] != treatment_value]

    grouped_treated = treated.groupby(quasi_identifiers)
    grouped_control = control.groupby(quasi_identifiers)

    balance_metrics = []
    group_labels = []

    # Flatten data and encode categorical columns
    df_flat_encoded = encode_categorical(df[quasi_identifiers].reset_index(drop=True), quasi_identifiers)

    for group_name, group_df_treated in grouped_treated:
        # Check if the control group also has data for the same group_name
        if group_name in grouped_control.groups:
            group_df_control = grouped_control.get_group(group_name)

            # Flatten data and encode categorical columns in treated and control groups
            group_df_treated_flat_encoded = encode_categorical(group_df_treated.reset_index(drop=True), quasi_identifiers)
            group_df_control_flat_encoded = encode_categorical(group_df_control.reset_index(drop=True), quasi_identifiers)

            # Check if the groups have data
            if not group_df_treated_flat_encoded.empty and not group_df_control_flat_encoded.empty:
                # Calculate balance metric
                t_statistic, _ = t_test(
                    df_flat_encoded.loc[group_df_treated_flat_encoded.index].values.flatten(),
                    df_flat_encoded.loc[group_df_control_flat_encoded.index].values.flatten()
                )

                balance_metrics.append(t_statistic)
                group_labels.append(group_name)

    values = np.array(balance_metrics)

    return GroupedMetric(values, group_labels, name='Balance Test')


def propensity_score_overlap(df, quasi_identifiers, treatment_col, treatment_value):
    """
    Calculate propensity score overlap metric for a given DataFrame and attributes.

    Synopse:
    Propensity score overlap metric measures the degree of overlap in propensity scores between treated and control groups based on specified quasi-identifiers.

    Details:
    The metric is computed by fitting a logistic regression model to predict the treatment indicator based on quasi-identifiers. Propensity scores are then extracted, and the mean overlap is calculated.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - quasi_identifiers (list): List of column names representing the quasi-identifiers.
    - treatment_col (str): The column name representing the treatment indicator.
    - treatment_value: The value indicating the treated group.

    Return:
    GroupedMetric: An object containing propensity score overlap metric values and corresponding group labels.

    Example:
    >>> df = pd.DataFrame({'A': [1, 2, 1, 2], 'B': ['X', 'Y', 'X', 'Y'], 'Treatment': [1, 0, 1, 0]})
    >>> metric_result = propensity_score_overlap(df, ['A', 'B'], 'Treatment', 1)
    >>> print(metric_result.values)
    [0.1, 0.2, ...]  # Propensity score overlap metric values
    >>> print(metric_result.group_labels)
    ['Treatment 1', 'Treatment 2', ...]  # Corresponding group labels

    """
    X = df[quasi_identifiers]
    y = df[treatment_col]

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Create a ColumnTransformer to apply one-hot encoding to categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ],
        remainder='drop'
    )

    # Create a pipeline with the preprocessor and logistic regression model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    # Fit the entire pipeline
    model.fit(X, y)

    # Transform the data
    X_encoded = model.named_steps['preprocessor'].transform(X)

    # Extract propensity scores for all treated samples against all control samples
    propensity_scores_treated = model.named_steps['classifier'].predict_proba(X_encoded[y == treatment_value])[:, 1]
    propensity_scores_control = model.named_steps['classifier'].predict_proba(X_encoded[y != treatment_value])[:, 1]

    # Calculate the mean propensity score overlap for each treated sample
    overlap_metrics = []

    for treated_score in propensity_scores_treated:
        overlap_metric = np.mean(np.abs(treated_score - propensity_scores_control))
        overlap_metrics.append(overlap_metric)

    # Use treatment labels as group labels
    group_labels = [f'Treatment {i+1}' for i in range(len(overlap_metrics))]

    values = np.array(overlap_metrics)

    return GroupedMetric(values, group_labels=group_labels, name='Propensity Score Overlap')
