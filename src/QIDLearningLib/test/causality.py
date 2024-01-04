import matplotlib.pyplot as plt
from metrics.causality import balance_test, covariate_shift, propensity_score_overlap
from util.data import generate_synthetic_dataset


def analyze_causality_metrics(df, quasi_identifiers, treatment_col, treatment_value):
    print("\nCovariate Shift Metrics:")
    causality_metrics_covariate_shift = covariate_shift(df, quasi_identifiers, treatment_col, treatment_value)
    print(repr(causality_metrics_covariate_shift))
    causality_metrics_covariate_shift.plot_all()

    print("\nBalance Test Metrics:")
    causality_metrics_balance_test = balance_test(df, quasi_identifiers, treatment_col, treatment_value)
    print(repr(causality_metrics_balance_test))
    causality_metrics_balance_test.plot_all()

    print("\nPropensity Score Overlap Metric:")
    causality_metrics_propensity_overlap = propensity_score_overlap(df, quasi_identifiers, treatment_col, treatment_value)
    print(repr(causality_metrics_propensity_overlap))
    causality_metrics_propensity_overlap.plot_all()

def main():
    # Generate synthetic dataset
    df = generate_synthetic_dataset()

    # Example usage for causality metrics
    quasi_identifiers = ['Age', 'Gender', 'Country', 'Education']
    treatment_col = 'Disease'  # Assuming there's a column indicating treatment/control groups
    treatment_value = 'A'  # Specify the treatment value

    # Analyze causality metrics
    analyze_causality_metrics(df, quasi_identifiers, treatment_col, treatment_value)

    plt.show()

if __name__ == '__main__':
    main()
