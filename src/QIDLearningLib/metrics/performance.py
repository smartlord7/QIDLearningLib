def specificity(predicted, actual):
    true_negatives = len(set(actual) - set(predicted))
    false_positives = len(set(predicted) - set(actual))

    if true_negatives + false_positives == 0:
        return 0.0
    else:
        return true_negatives / (true_negatives + false_positives)


def fpr(predicted, actual):
    true_negatives = len(set(actual) - set(predicted))
    false_positives = len(set(predicted) - set(actual))

    if true_negatives + false_positives == 0:
        return 0.0
    else:
        return false_positives / (true_negatives + false_positives)


def precision(predicted, actual):
    true_positives = len(set(predicted) & set(actual))
    false_positives = len(set(predicted) - set(actual))

    if true_positives + false_positives == 0:
        return 0.0
    else:
        return true_positives / (true_positives + false_positives)

def recall(predicted, actual):
    true_positives = len(set(predicted) & set(actual))
    false_negatives = len(set(actual) - set(predicted))

    if true_positives + false_negatives == 0:
        return 0.0
    else:
        return true_positives / (true_positives + false_negatives)


def f1_score(predicted, actual):
    prec = precision(predicted, actual)
    rec = recall(predicted, actual)

    if prec + rec == 0:
        return 0.0
    else:
        return 2 * (prec * rec) / (prec + rec)


def jaccard_similarity(predicted, actual):
    intersection_size = len(set(predicted) & set(actual))
    union_size = len(set(predicted) | set(actual))

    if union_size == 0:
        return 0.0
    else:
        return intersection_size / union_size

def dice_similarity(predicted, actual):
    intersection_size = len(predicted.intersection(actual))
    total_size = len(predicted) + len(actual)

    return 2 * intersection_size / total_size if total_size != 0 else 0


def overlap_coefficient(predicted, actual):
    intersection_size = len(predicted.intersection(actual))
    min_size = min(len(predicted), len(actual))

    return intersection_size / min_size if min_size != 0 else 0


def accuracy(predicted, actual):
    correct_predictions = len(set(predicted) & set(actual))
    total_instances = len(set(predicted) | set(actual))

    if total_instances == 0:
        return 0.0
    else:
        return correct_predictions / total_instances