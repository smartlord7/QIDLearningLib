# Calculate the correlation between multiple metrics
def calculate_metric_correlations(df: pd.DataFrame, metrics: Dict[str, Callable],
                                  quasi_identifiers_combinations: List[List[str]]) -> pd.DataFrame:
    metric_values = []
    metric_names = metrics.keys()

    # Loop over combinations of quasi-identifiers and compute metrics
    for combo in quasi_identifiers_combinations:
        metric_row = []
        for metric_name, metric_func in metrics.items():
            value = metric_func(df, combo)
            metric_row.append(value)
        metric_values.append(metric_row)

    # Create DataFrame with metric values for each combination
    metrics_df = pd.DataFrame(metric_values, columns=[metric_name for metric_name in metric_names])

    # Compute correlation matrix
    correlation_matrix = metrics_df.corr()

    return correlation_matrix