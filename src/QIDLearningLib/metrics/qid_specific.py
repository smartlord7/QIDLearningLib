import numpy as np


def separation(df, attributes):
    subset_df = df[attributes]
    _, counts = np.unique(subset_df, axis=0, return_counts=True)
    to_subtract = np.sum((counts - 1) * counts // 2)
    total = subset_df.shape[0] * (subset_df.shape[0] - 1) / 2
    separation_value = (total - to_subtract) * 100 / total

    return separation_value

def distinction(df, attributes):
    subset_df = df[attributes]
    unique = np.unique(subset_df, axis=0) if len(subset_df.shape) == 2 else np.unique(subset_df)
    dist = (len(unique) / subset_df.shape[0]) * 100

    return dist
