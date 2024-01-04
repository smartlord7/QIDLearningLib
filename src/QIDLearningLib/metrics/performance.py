"""
QIDLearningLib

Library Description (QIDLearningLib):
QIDLearningLib is a Python library designed to provide a comprehensive set of metrics for quasi-identification recognition processes.
The library encompasses metrics for assessing data privacy, data utility, and the performance of quasi-identification recognition algorithms.

Module Description (metrics.performance):
This module provides a set of performance metrics commonly used in the evaluation of classification and prediction models.
The metrics include specificity, false positive rate (fpr), precision, recall, F1 score, Jaccard similarity, Dice similarity,
overlap coefficient, and accuracy. In this case, since the real and predicted quasi-identifiers are sets (the order is not relevant), the mentioned
metrics are therefore adapted for sets and make use of sets operations, such as intersection and union.


Year: 2023/2024
Institution: University of Coimbra
Department: Department of Informatics Engineering
Program: Master's in Informatics Engineering - Intelligent Systems
Author: Sancho Amaral Simões
Student No: 2019217590
Email: sanchoamaralsimoes@gmail.com
Version: v0.01

License:
This open-source software is released under the terms of the GNU General Public License, version 3 (GPL-3.0).
For more details, see https://www.gnu.org/licenses/gpl-3.0.html

"""


def specificity(predicted: set, actual: set) -> float:
    """
    Calculate the Specificity metric for binary classification.

    Synopse:
    Specificity measures the proportion of true negatives among all actual negatives.

    Details:
    It is computed by counting the true negatives (instances correctly predicted as negatives) and
    false positives (instances incorrectly predicted as positives) and then calculating the ratio of true negatives
    to the sum of true negatives and false positives.

    Parameters:
    - predicted (set): Set of predicted positive instances.
    - actual (set): Set of actual positive instances.

    Return:
    float: Specificity metric value.

    Example:
    >>> spec_value = specificity(predicted_set, actual_set)
    >>> print(spec_value)

    """

    # Calculate the true negatives by finding instances that are in the actual set but not in the predicted set
    true_negatives = len(actual - predicted)

    # Calculate the false positives by finding instances that are in the predicted set but not in the actual set
    false_positives = len(predicted - actual)

    # Check if the sum of true negatives and false positives is zero to avoid division by zero
    if true_negatives + false_positives == 0:
        # Return 0.0 if the denominator is zero to handle the edge case
        return 0.0
    else:
        # Return the specificity metric by dividing true negatives by the sum of true negatives and false positives
        return true_negatives / (true_negatives + false_positives)


def fpr(predicted: set, actual: set) -> float:
    """
    Calculate the False Positive Rate (FPR) metric for binary classification.

    Synopse:
    FPR measures the proportion of false positives among all actual negatives.

    Details:
    It is computed by counting the true negatives and false positives and then calculating the ratio of false positives
    to the sum of true negatives and false positives.

    Parameters:
    - predicted (set): Set of predicted positive instances.
    - actual (set): Set of actual positive instances.

    Return:
    float: FPR metric value.

    Example:
    >>> fpr_value = fpr(predicted_set, actual_set)
    >>> print(fpr_value)

    """

    # Calculate the true negatives by finding instances that are in the actual set but not in the predicted set
    true_negatives = len(actual - predicted)

    # Calculate the false positives by finding instances that are in the predicted set but not in the actual set
    false_positives = len(predicted - actual)

    # Check if the sum of true negatives and false positives is zero to avoid division by zero
    if true_negatives + false_positives == 0:
        # Return 0.0 if the denominator is zero to handle the edge case
        return 0.0
    else:
        # Return the false positive rate metric by dividing false positives by the sum of true negatives and false positives
        return false_positives / (true_negatives + false_positives)


def precision(predicted: set, actual: set) -> float:
    """
    Calculate the Precision metric for binary classification.

    Synopse:
    Precision measures the proportion of true positives among all predicted positives.

    Details:
    It is computed by counting the true positives (instances correctly predicted as positives) and false positives
    and then calculating the ratio of true positives to the sum of true positives and false positives.

    Parameters:
    - predicted (set): Set of predicted positive instances.
    - actual (set): Set of actual positive instances.

    Return:
    float: Precision metric value.

    Example:
    >>> precision_value = precision(predicted_set, actual_set)
    >>> print(precision_value)

    """

    # Calculate the true positives by finding instances that are both in the predicted and actual sets
    true_positives = len(predicted & actual)

    # Calculate the false positives by finding instances that are in the predicted set but not in the actual set
    false_positives = len(predicted - actual)

    # Check if the sum of true positives and false positives is zero to avoid division by zero
    if true_positives + false_positives == 0:
        # Return 0.0 if the denominator is zero to handle the edge case
        return 0.0
    else:
        # Return the precision metric by dividing true positives by the sum of true positives and false positives
        return true_positives / (true_positives + false_positives)


def recall(predicted: set, actual: set) -> float:
    """
    Calculate the Recall (Sensitivity) metric for binary classification.

    Synopse:
    Recall measures the proportion of true positives among all actual positives.

    Details:
    It is computed by counting the true positives and false negatives (instances incorrectly predicted as negatives)
    and then calculating the ratio of true positives to the sum of true positives and false negatives.

    Parameters:
    - predicted (set): Set of predicted positive instances.
    - actual (set): Set of actual positive instances.

    Return:
    float: Recall metric value.

    Example:
    >>> recall_value = recall(predicted_set, actual_set)
    >>> print(recall_value)

    """

    # Calculate the true positives by finding instances that are both in the predicted and actual sets
    true_positives = len(predicted & actual)

    # Calculate the false negatives by finding instances that are in the actual set but not in the predicted set
    false_negatives = len(actual - predicted)

    # Check if the sum of true positives and false negatives is zero to avoid division by zero
    if true_positives + false_negatives == 0:
        # Return 0.0 if the denominator is zero to handle the edge case
        return 0.0
    else:
        # Return the recall metric by dividing true positives by the sum of true positives and false negatives
        return true_positives / (true_positives + false_negatives)


def f1_score(predicted: set, actual: set) -> float:
    """
    Calculate the F1 Score metric for binary classification.

    Synopse:
    F1 Score is the harmonic mean of Precision and Recall.

    Details:
    It is computed by calculating Precision and Recall using the provided sets of predicted and actual instances
    and then applying the formula: 2 * (precision * recall) / (precision + recall).

    Parameters:
    - predicted (set): Set of predicted positive instances.
    - actual (set): Set of actual positive instances.

    Return:
    float: F1 Score metric value.

    Example:
    >>> f1_score_value = f1_score(predicted_set, actual_set)
    >>> print(f1_score_value)

    """

    # Calculate precision and recall using the specified functions
    prec = precision(predicted, actual)
    rec = recall(predicted, actual)

    # Check if the sum of precision and recall is zero to avoid division by zero
    if prec + rec == 0:
        # Return 0.0 if the denominator is zero to handle the edge case
        return 0.0
    else:
        # Return the F1 score by applying the formula: 2 * (precision * recall) / (precision + recall)
        return 2 * (prec * rec) / (prec + rec)


def f2_score(predicted: set, actual: set, beta: float = 2.0) -> float:
    """
    Calculate the F2 Score metric for binary classification.

    Synopse:
    F2 Score is the harmonic mean of Precision and Recall with emphasis on recall.

    Details:
    It is computed by calculating Precision and Recall using the provided sets of predicted and actual instances
    and then applying the formula: (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall).

    Parameters:
    - predicted (set): Set of predicted positive instances.
    - actual (set): Set of actual positive instances.
    - beta (float, optional): Controls the trade-off between precision and recall. Defaults to 2.0.

    Return:
    float: F2 Score metric value.

    Example:
    >>> f2_score_value = f2_score(predicted_set, actual_set, beta=2.0)
    >>> print(f2_score_value)

    """

    # Calculate true positives, false positives, and false negatives
    true_positives = len(predicted.intersection(actual))
    false_positives = len(predicted - actual)
    false_negatives = len(actual - predicted)

    # Calculate precision and recall, handling the case where the denominator is zero
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0

    # Check if precision + recall is zero to avoid division by zero
    if precision + recall == 0:
        # Return 0.0 if the denominator is zero to handle the edge case
        return 0.0
    else:
        # Calculate the F-beta score using the specified beta value
        numerator = (1 + beta ** 2) * precision * recall
        denominator = (beta ** 2 * precision) + recall
        return numerator / denominator


def jaccard_similarity(predicted: set, actual: set) -> float:
    """
    Calculate the Jaccard Similarity metric for binary classification.

    Synopse:
    Jaccard Similarity measures the intersection over union of predicted and actual sets.

    Details:
    It is computed by calculating the size of the intersection and the size of the union of predicted and actual sets,
    and then applying the formula: intersection_size / union_size.

    Parameters:
    - predicted (set): Set of predicted positive instances.
    - actual (set): Set of actual positive instances.

    Return:
    float: Jaccard Similarity metric value.

    Example:
    >>> jaccard_value = jaccard_similarity(predicted_set, actual_set)
    >>> print(jaccard_value)

    """

    # Calculate the size of the intersection and union of the sets
    intersection_size = len(predicted & actual)
    union_size = len(predicted | actual)

    # Check if the union size is zero to avoid division by zero
    if union_size == 0:
        # Return 0.0 if the union size is zero to handle the edge case
        return 0.0
    else:
        # Calculate the Jaccard similarity
        return intersection_size / union_size


def dice_similarity(predicted: set, actual: set, alpha: float = 1.0, beta: float = 1.0) -> float:
    """
    Calculate the Dice Similarity Coefficient for binary classification.

    Synopse:
    Dice Similarity Coefficient measures the similarity between predicted and actual sets.

    Details:
    It is computed by calculating the size of the intersection and the total size of predicted and actual sets,
    and then applying the formula: (alpha * intersection_size^beta) / ((alpha * size_of_predicted_set^beta) + (size_of_actual_set^beta)). If alpha and beta are 1
    then is the same as calculating the F1 score.

    Parameters:
    - predicted (set): Set of predicted positive instances.
    - actual (set): Set of actual positive instances.
    - alpha (float, optional): Controls the weight of the intersection. Defaults to 1.5.
    - beta (float, optional): Controls the exponent in the denominator. Defaults to 1.0.

    Return:
    float: Dice Similarity Coefficient value.

    Example:
    >>> dice_value = dice_similarity(predicted_set, actual_set, alpha=1.5, beta=1.0)
    >>> print(dice_value)

    """

    # Calculate the size of the intersection, size of predicted set, and size of actual set
    intersection_size = len(predicted.intersection(actual))
    size_of_predicted_set = len(predicted)
    size_of_actual_set = len(actual)

    # Calculate the numerator and denominator for the Dice similarity coefficient
    numerator = alpha * intersection_size
    denominator = (alpha * size_of_predicted_set ** beta) + (size_of_actual_set ** beta)

    # Check if the denominator is zero to avoid division by zero
    if denominator != 0:
        # Calculate and return the Dice similarity coefficient
        return 2 * numerator / denominator
    else:
        # Return 0.0 if the denominator is zero to handle the edge case
        return 0.0


def accuracy(predicted: set, actual: set) -> float:
    """
    Calculate the Accuracy metric for binary classification.

    Synopse:
    Accuracy measures the proportion of correctly predicted instances.

    Details:
    It is computed by counting the correct predictions (true positives and true negatives)
    and then calculating the ratio of correct predictions to the total number of instances.

    Parameters:
    - predicted (set): Set of predicted positive instances.
    - actual (set): Set of actual positive instances.

    Return:
    float: Accuracy metric value.

    Example:
    >>> accuracy_value = accuracy(predicted_set, actual_set)
    >>> print(accuracy_value)

    """

    # Calculate the number of correct predictions and the total number of instances
    correct_predictions = len(predicted & actual)
    total_instances = len(actual)

    # Check if the total number of instances is zero to avoid division by zero
    if total_instances != 0:
        # Calculate and return the accuracy
        return correct_predictions / total_instances
    else:
        # Return 0.0 if the total number of instances is zero to handle the edge case
        return 0.0