import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def generate_synthetic_dataset(num_records=10000):
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


def generate_random_sets(size=500, overlap_ratio=0.3):
    size = max(size, 2)
    overlap_size = int(overlap_ratio * size)
    elements = list(range(size * 2))

    # Shuffle the elements to add more randomness
    random.shuffle(elements)

    set1 = set(elements[:size])
    set2 = set(elements[size - overlap_size:])

    return set1, set2


def calculate_core(dataframe):
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


def encode_categorical(df, columns):
    """Encode categorical columns to numeric representation."""
    label_encoder = LabelEncoder()
    for column in columns:
        df[column] = label_encoder.fit_transform(df[column])
    return df