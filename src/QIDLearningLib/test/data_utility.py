from metrics.data_utility import *
from util.data import generate_synthetic_dataset, calculate_core


def analyze_data_utility_metrics(df, quasi_identifiers, target_attribute, true_values):
    print("Mean Squared Error:")
    mse_metric = mean_squared_err(df, quasi_identifiers, target_attribute, true_values)
    print(repr(mse_metric))
    mse_metric.plot_all()

    print("\nAccuracy:")
    accuracy_metric = accuracy(df, quasi_identifiers, target_attribute, true_values)
    print(repr(accuracy_metric))
    accuracy_metric.plot_all()

    print("\nUtility Score (Sum):")
    utility_metric_sum = utility_score(df, quasi_identifiers, target_attribute, lambda x: x.sum())
    print(repr(utility_metric_sum))
    utility_metric_sum.plot_all()

    print("\nRange Utility:")
    range_utility_metric = range_utility(df, quasi_identifiers, target_attribute)
    print(repr(range_utility_metric))
    range_utility_metric.plot_all()

    print("\nDistinct Values Utility:")
    distinct_values_utility_metric = distinct_values_utility(df, quasi_identifiers, target_attribute)
    print(repr(distinct_values_utility_metric))
    distinct_values_utility_metric.plot_all()

    print("\nCompleteness Utility:")
    completeness_utility_metric = completeness_utility(df, quasi_identifiers, target_attribute)
    print(repr(completeness_utility_metric))
    completeness_utility_metric.plot_all()

    print("\nCore:")
    print(calculate_core(df))

def main():
    # Generate synthetic dataset
    df = generate_synthetic_dataset()

    # Example usage
    quasi_identifiers = ['Age', 'Gender', 'Country', 'Education']
    target_attribute = 'Income'
    true_values = df[target_attribute]

    # Analyze data utility metrics
    analyze_data_utility_metrics(df, quasi_identifiers, target_attribute, true_values)

if __name__ == '__main__':
    main()
