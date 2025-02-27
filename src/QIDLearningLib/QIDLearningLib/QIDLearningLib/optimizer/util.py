import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# =============================================================================
# Data Loading and Preprocessing Functions
# =============================================================================

def load_data_from_folder(folder_path):
    """
    Load datasets from a folder where each subfolder contains files ending with ".data".
    Returns a list of tuples (file_name, dataframe) and a header list.
    """
    logging.info(f"Loading data from folder: {folder_path}")
    datasets = []
    headers = []
    for folder in os.listdir(folder_path):
        folder_full = os.path.join(folder_path, folder)
        if os.path.isdir(folder_full):
            logging.info(f"Processing folder: {folder_full}")
            for file in os.listdir(folder_full):
                if file.endswith(".data"):
                    file_path = os.path.join(folder_full, file)
                    try:
                        df = pd.read_csv(file_path, delimiter=',')
                        headers.append((file, df.columns.tolist()))
                        datasets.append((file, df))
                        logging.info(f"Loaded dataset: {file_path}, shape: {df.shape}")
                    except Exception as e:
                        logging.error(f"Error loading {file_path}: {e}")
    return datasets, headers


def one_hot_encode_data(df):
    """
    One-hot encode a given dataframe.
    """
    logging.info("One-hot encoding data")
    try:
        encoder = OneHotEncoder()
        encoded_array = encoder.fit_transform(df).toarray()
        encoded_df = pd.DataFrame(encoded_array)
        logging.info(f"Data shape after one-hot encoding: {encoded_df.shape}")
        return encoded_df
    except Exception as e:
        logging.error(f"Error during one-hot encoding: {e}")
        raise


