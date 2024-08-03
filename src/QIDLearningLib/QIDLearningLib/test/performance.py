"""
QIDLearningLib

Library Description (QIDLearningLib):
QIDLearningLib is a Python library designed to provide a vast set of metrics for quasi-identifiers recognition processes.
The library includes metrics for assessing data privacy, data utility, and the performance of quasi-identifiers recognition algorithms.

Module Description (test.performance):
This module in QIDLearningLib allows to test all at once the metrics present in the module metrics.performance.

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

from metrics.performance import (
    accuracy,
    precision,
    recall,
    f1_score,
    f2_score,
    specificity,
    fpr,
    jaccard_similarity,
    dice_similarity
)
from util.data import generate_random_sets


def test_performance_metrics(
    predicted_qid: set[str],
    actual_qid: set[str]
) -> None:
    """
    Test performance metrics for predicted quasi-identifiers against actual quasi-identifiers.

    Synopse:
    This function prints various performance metrics, including Accuracy, Precision, Recall, F1 Score, F2 Score, Specificity, False Positive Rate (FPR), Jaccard Index, and Dice Similarity, for predicted quasi-identifiers against actual quasi-identifiers.

    Parameters:
    - predicted_qid (List[int]): List of predicted quasi-identifiers.
    - actual_qid (List[int]): List of actual quasi-identifiers.

    Return:
    None: The function prints the performance metrics.

    Example:
    >>> predicted_quasi_identifiers, actual_quasi_identifiers = generate_random_sets(size=100, overlap_ratio=0.9)
    >>> test_performance_metrics(predicted_quasi_identifiers, actual_quasi_identifiers)

    """

    print("Actual QIDs: ", actual_qid)
    print("Predicted QIDs: ", predicted_qid)
    print("Accuracy:", accuracy(predicted_qid, actual_qid))
    print("Precision:", precision(predicted_qid, actual_qid))
    print("Recall:", recall(predicted_qid, actual_qid))
    print("F1 Score:", f1_score(predicted_qid, actual_qid))
    print("F2 Score:", f2_score(predicted_qid, actual_qid))
    print("Specificity:", specificity(predicted_qid, actual_qid))
    print("FPR:", fpr(predicted_qid, actual_qid))
    print("Jaccard Index:", jaccard_similarity(predicted_qid, actual_qid))
    print("Dice Similarity:", dice_similarity(predicted_qid, actual_qid))

def main() -> None:
    """
    Main function to demonstrate the usage of performance metrics testing.

    Synopse:
    This function generates random sets of predicted and actual quasi-identifiers and tests the performance metrics using the 'test_performance_metrics' function.

    Parameters:
    None

    Return:
    None

    Example:
    >>> main()
    """
    # Example usage
    predicted_quasi_identifiers, actual_quasi_identifiers = generate_random_sets(size=100, overlap_ratio=0.9)

    test_performance_metrics(predicted_quasi_identifiers, actual_quasi_identifiers)

if __name__ == '__main__':
    main()
