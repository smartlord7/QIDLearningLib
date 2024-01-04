"""
QIDLearningLib

Library Description (QIDLearningLib):
QIDLearningLib is a Python library designed to provide a comprehensive set of metrics for quasi-identification recognition processes.
The library encompasses metrics for assessing data privacy, data utility, and the performance of quasi-identification recognition algorithms.

Module Description (util.data):
This module in QIDLearningLib defines some utility functions related to dataset manipulation and generation.

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

import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Set, Tuple, List


def generate_synthetic_dataset(num_records: int = 10000) -> pd.DataFrame:
    """
    Generate a synthetic dataset with random attributes.

    Synopse:
    This function generates a synthetic dataset with attributes such as Age, Gender, Income, Disease, Country, and Education.

    Parameters:
    - num_records (int): Number of records in the dataset.

    Return:
    pd.DataFrame: The synthetic dataset.

    Example:
    >>> synthetic_df = generate_synthetic_dataset(num_records=500)
    >>> print(synthetic_df.head())

    """
    np.random.seed(42)

    ages = np.random.randint(18, 65, num_records)
    genders = np.random.choice(['Male', 'Female'], num_records)
    incomes = np.random.normal(50000, 15000, num_records).astype(int)
    diseases = np.random.choice(['A', 'B', 'C', 'D'], num_records)

    df = pd.DataFrame({
        'ID': range(1, num_records + 1),
        'Age': ages,
        'Gender': genders,
        'Income': incomes,
        'Disease': diseases
    })

    # Adding more complexity with additional attributes
    countries = np.random.choice(['USA', 'Canada', 'UK', 'Germany'], num_records)
    education_levels = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], num_records)

    df['Country'] = countries
    df['Education'] = education_levels

    return df


def generate_random_sets(size: int = 500, overlap_ratio: float = 0.3) -> Tuple[Set[int], Set[int]]:
    """
    Generate two random sets with a specified size and overlap ratio.

    Synopse:
    This function generates two random sets with a specified size and overlap ratio.

    Parameters:
    - size (int): Size of each set.
    - overlap_ratio (float): Overlap ratio between the two sets.

    Return:
    Tuple[Set[int], Set[int]]: Two random sets.

    Example:
    >>> set1, set2 = generate_random_sets(size=100, overlap_ratio=0.2)
    >>> print(set1, set2)

    """
    size = max(size, 2)
    overlap_size = int(overlap_ratio * size)
    elements = list(range(size * 2))

    # Shuffle the elements to add more randomness
    random.shuffle(elements)

    set1 = set(elements[:size])
    set2 = set(elements[size - overlap_size:])

    return set1, set2


def calculate_core(dataframe: pd.DataFrame) -> Set[str]:
    """
    Calculate the core attributes of a DataFrame.

    Synopse:
    The core attributes are those that, when removed, do not reduce the granularity of the partition.

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame.

    Return:
    Set[str]: Set of core attributes.

    Example:
    >>> core_attributes = calculate_core(dataframe)
    >>> print(core_attributes)

    """
    # Ensure that the input is a pandas DataFrame
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    # Get the columns of the DataFrame
    columns = dataframe.columns

    # Initialize the set of core attributes
    core_attributes = set(columns)

    # Iterate through each attribute
    for attribute in columns:
        # Check if removing the attribute reduces the granularity of the partition
        unique_combinations_before = dataframe.groupby(attribute).size().reset_index(name='count')

        # If removing the attribute reduces granularity, update the core attributes set
        if len(unique_combinations_before) > dataframe[attribute].nunique():
            core_attributes.discard(attribute)

    return core_attributes


def encode_categorical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Encode categorical columns to numeric representation.

    Synopse:
    This function encodes categorical columns of a DataFrame to a numeric representation using Label Encoding.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (List[str]): List of column names representing the categorical columns.

    Return:
    pd.DataFrame: DataFrame with encoded categorical columns.

    Example:
    >>> df_encoded = encode_categorical(df, ['Gender', 'Country'])
    >>> print(df_encoded.head())

    """
    label_encoder = LabelEncoder()
    for column in columns:
        df[column] = label_encoder.fit_transform(df[column])
    return df
