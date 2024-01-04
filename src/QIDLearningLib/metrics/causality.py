import numpy as np
from scipy.stats import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from structure.GroupedMetric import GroupedMetric
from util.data import encode_categorical


def covariate_shift(df, quasi_identifiers, treatment_col, treatment_value):
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




# Utility functions for statistical tests
def ks_test(sample1, sample2):
    # Kolmogorov-Smirnov test
    ks_statistic, _ = stats.ks_2samp(sample1, sample2)
    return ks_statistic, _


def t_test(sample1, sample2):
    # Independent t-test
    t_statistic, _ = stats.ttest_ind(sample1, sample2)
    return t_statistic, _