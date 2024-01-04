import random
from metrics.performance import *
from util.data import generate_random_sets


def test_performance_metrics(predicted_qid, actual_qid):
    print("Actual QIDs: ", actual_qid)
    print("Predicted QIDs: ", predicted_qid)
    print("Accuracy:", accuracy(predicted_qid, actual_qid))
    print("Precision:", precision(predicted_qid, actual_qid))
    print("Recall:", recall(predicted_qid, actual_qid))
    print("F1 Score:", f1_score(predicted_qid, actual_qid))
    print("Specificity:", specificity(predicted_qid, actual_qid))
    print("FPR:", fpr(predicted_qid, actual_qid))
    print("Jaccard Index:", jaccard_similarity(predicted_qid, actual_qid))
    print("Dice Similarity:", dice_similarity(predicted_qid, actual_qid))
    print("Overlap Coefficient:", overlap_coefficient(predicted_qid, actual_qid))

def main():
    # Example usage
    predicted_quasi_identifiers, actual_quasi_identifiers = generate_random_sets(size=10)

    test_performance_metrics(predicted_quasi_identifiers, actual_quasi_identifiers)

if __name__ == '__main__':
    main()