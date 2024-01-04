"""
QIDLearningLib

Library Description (QIDLearningLib):
QIDLearningLib is a Python library designed to provide a comprehensive set of metrics for quasi-identification recognition processes.
The library encompasses metrics for assessing data privacy, data utility, and the performance of quasi-identification recognition algorithms.

Module Description (test.data_privacy):
This module in QIDLearningLib allows to test all at once the metrics present in the module metrics.data_privacy.

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

from QIDLearningLib.metrics.data_privacy import *
from QIDLearningLib.util.data import generate_synthetic_dataset


def analyze_privacy_metrics(df, quasi_identifiers, sensitive_attribute):
    print("k-Anonymity:")
    k_anonymity_metric = k_anonymity(df, quasi_identifiers)
    print(repr(k_anonymity_metric))
    k_anonymity_metric.plot_all()

    print("\nl-Diversity:")
    l_diversity_metric = l_diversity(df, quasi_identifiers, [sensitive_attribute, ])
    print(repr(l_diversity_metric))
    l_diversity_metric.plot_all()

    print("\nCloseness Centrality:")
    closeness_centrality_metric = closeness_centrality(df, quasi_identifiers, [sensitive_attribute, ])
    print(repr(closeness_centrality_metric))
    closeness_centrality_metric.plot_all()

    print("\nt-Closeness:")
    t_closeness_metric = t_closeness(df, quasi_identifiers, [sensitive_attribute, ])
    print(repr(t_closeness_metric))
    t_closeness_metric.plot_all()

    print("\nDelta Presence:")
    delta_presence_metric = delta_presence(df, ['Age', ], [22, ])
    print(repr(delta_presence_metric))
    delta_presence_metric.plot_all()

    print("\nGeneralization Ratio:")
    generalization_ratio_metric = generalization_ratio(df, quasi_identifiers, [sensitive_attribute, ])
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
