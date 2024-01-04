from metrics.data_privacy import *
from util.data import generate_synthetic_dataset


def analyze_privacy_metrics(df, quasi_identifiers, sensitive_attribute):
    print("k-Anonymity:")
    k_anonymity_metric = k_anonymity(df, quasi_identifiers)
    print(repr(k_anonymity_metric))
    k_anonymity_metric.plot_all()

    print("\nl-Diversity:")
    l_diversity_metric = l_diversity(df, quasi_identifiers, sensitive_attribute)
    print(repr(l_diversity_metric))
    l_diversity_metric.plot_all()

    print("\nCloseness Centrality:")
    closeness_centrality_metric = closeness_centrality(df, quasi_identifiers, sensitive_attribute)
    print(repr(closeness_centrality_metric))
    closeness_centrality_metric.plot_all()

    print("\nt-Closeness:")
    t_closeness_metric = t_closeness(df, quasi_identifiers, sensitive_attribute)
    print(repr(t_closeness_metric))
    t_closeness_metric.plot_all()

    print("\nDelta Presence:")
    delta_presence_metric = delta_presence(df, 'Age', 22)
    print(repr(delta_presence_metric))
    delta_presence_metric.plot_all()

    print("\nGeneralization Ratio:")
    generalization_ratio_metric = generalization_ratio(df, quasi_identifiers, sensitive_attribute)
    print(repr(generalization_ratio_metric))
    generalization_ratio_metric.plot_all()

def main():
    # Generate synthetic dataset
    df = generate_synthetic_dataset()

    # Example usage
    quasi_identifiers = ['Age', 'Gender', 'Country', 'Education']
    sensitive_attribute = 'Disease'

    # Analyze privacy metrics
    analyze_privacy_metrics(df, quasi_identifiers, sensitive_attribute)

if __name__ == '__main__':
    main()
