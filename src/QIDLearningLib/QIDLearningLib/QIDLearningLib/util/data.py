"""
QIDLearningLib

Library Description (QIDLearningLib):
QIDLearningLib is a Python library designed to provide a vast set of metrics for quasi-identifiers recognition processes.
The library includes metrics for assessing data privacy, data utility, and the performance of quasi-identifiers recognition algorithms.

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


def generate_synthetic_dataset(
    num_records: int = 10000,
    age_range: tuple = (12, 90),
    income_mean: float = 50000,
    income_std: float = 15000,
    diseases: list = ['A', 'B', 'C', 'D', 'E', 'F'],
    genders: list = ['Male', 'Female', 'Non-binary', 'Other'],
    countries: list = ['USA', 'Canada', 'UK', 'Germany'],
    education_levels: list = ['High School', 'Bachelor', 'Master', 'PhD'],
    marital_status: list = ['Single', 'Married', 'Divorced', 'Widowed'],
    employment_status: list = ['Employed', 'Unemployed', 'Self-employed', 'Student'],
    housing_types: list = ['Owned', 'Rented', 'Mortgaged', 'Living with Parents'],
    credit_score_range: tuple = (300, 850)
) -> pd.DataFrame:
    """
    Generate a synthetic dataset with customizable attributes.

    Synopse:
    This function generates a synthetic dataset with customizable attributes such as Age, Gender, Income,
    Disease, Country, Education, Marital Status, Employment Status, Housing Type, and Credit Score.

    Parameters:
    - num_records (int): Number of records in the dataset.
    - age_range (tuple): The range of ages (min, max).
    - income_mean (float): The mean income.
    - income_std (float): The standard deviation of income.
    - diseases (list): List of possible disease categories.
    - genders (list): List of possible gender options.
    - countries (list): List of possible country options.
    - education_levels (list): List of possible education levels.
    - marital_status (list): List of possible marital status options.
    - employment_status (list): List of possible employment status options.
    - housing_types (list): List of possible housing types.
    - credit_score_range (tuple): The range of credit scores (min, max).

    Return:
    pd.DataFrame: The synthetic dataset.

    Example:
    >>> synthetic_df = generate_synthetic_dataset(num_records=500)
    >>> print(synthetic_df.head())

    """
    np.random.seed(42)

    # Generate Age, Gender, and Income
    ages = np.random.randint(age_range[0], age_range[1], num_records)
    genders = np.random.choice(genders, num_records)
    incomes = np.random.normal(income_mean, income_std, num_records).astype(int)
    diseases = np.random.choice(diseases, num_records)

    # Generate more complex attributes
    countries = np.random.choice(countries, num_records)
    education_levels = np.random.choice(education_levels, num_records)
    marital_statuses = np.random.choice(marital_status, num_records)
    employment_statuses = np.random.choice(employment_status, num_records)
    housing_types = np.random.choice(housing_types, num_records)
    credit_scores = np.random.randint(credit_score_range[0], credit_score_range[1], num_records)

    # Creating the DataFrame
    df = pd.DataFrame({
        'ID': range(1, num_records + 1),
        'Age': ages,
        'Gender': genders,
        'Income': incomes,
        'Disease': diseases,
        'Country': countries,
        'Education': education_levels,
        'Marital Status': marital_statuses,
        'Employment Status': employment_statuses,
        'Housing Type': housing_types,
        'Credit Score': credit_scores
    })

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
