"""
This file contains utility functions to be used in the adult project
"""

import os
import re
import requests
import json
from io import StringIO

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def banner(ban_char, nban_char, title=None):
    if not isinstance(ban_char, str) or len(ban_char) != 1:
        raise ValueError("ban_char must be a single character string.")

    if not isinstance(nban_char, int) or nban_char <= 0:
        raise ValueError("nban_char must be a positive integer.")

    if title is not None:
        print(ban_char * nban_char)
        print(title.center(nban_char))
        print(ban_char * nban_char)

    if title is None:
        print(ban_char * nban_char)


def extract_feature_names(data_path, start_line=0, end_line=None, pattern=r"([a-zA-Z0-9\-]+):"):
    """
    Fetches the content of a file from a URL or a local path, skips to a specified line,
    and extracts data based on a regex pattern.

    Parameters:
    - data_path (str): The URL or local path of the file.
    - start_line (int): The line number to start reading from (default is 0).
    - end_line (int or None): The line number to stop reading (default is None, which reads till the end).
    - pattern (str): The regex pattern to extract data from the file (default is r"([a-zA-Z0-9\\-]+):").

    Returns:
    - extracted_data (list): List of data extracted based on the provided pattern.
    """
    content = ""

    pattern = re.compile(pattern)

    try:
        if data_path.startswith('http://') or data_path.startswith('https://'):
            response = requests.get(data_path, stream=True)
            if response.status_code == 200:
                lines = response.iter_lines(decode_unicode=True)
                content = "\n".join(
                    [line for i, line in enumerate(lines) if i >= start_line and (end_line is None or i <= end_line)]
                )
            else:
                print(f"Error: Unable to fetch the file from the URL. Status code: {response.status_code}")
                return []

        elif os.path.exists(data_path):
            with open(data_path, 'r') as file:
                lines = file.readlines()
                if end_line is None:
                    content = "".join(lines[start_line:])
                else:
                    content = "".join(lines[start_line:end_line+1])
        else:
            print(f"Error: The file path '{data_path}' does not exist.")
            return []

        extracted_data = pattern.findall(content)

        return extracted_data

    except requests.exceptions.RequestException as e:
        print(f"Error: An error occurred while trying to fetch the URL. {e}")
        return []

    except Exception as e:
        print(f"Error: An unexpected error occurred. {e}")
        return []


def preview_raw_data_from_url(url, n_lines=5):
    """
    Preview the first `n_lines` of raw data from a URL.

    Parameters:
    url (str): The URL of the dataset.
    n_lines (int): The number of lines to preview. Default is 5.

    Returns:
    None: Prints the preview of raw data.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()

        data_lines = response.text.splitlines()
        for line in data_lines[:n_lines]:
            print(line)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")


def preview_data_from_url(url, n_lines=5):
    """
    Preview the first `n_lines` of data from a URL without loading the entire dataset.

    Parameters:
    url (str): The URL of the dataset.
    n_lines (int): The number of lines to preview. Default is 5.

    Returns:
    pd.DataFrame: A DataFrame containing the first `n_lines` of the dataset.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()

        data = StringIO(response.text)
        df_preview = pd.read_csv(data, nrows=n_lines)

        return df_preview

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except pd.errors.EmptyDataError:
        print("The data could not be loaded. It may be empty or improperly formatted.")
        return None


def get_feature_value_counts(df, column_name):
    """
    Print the value counts of a specified column from a given DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to get the value counts from.

    Returns:
    None: Prints the value counts of the column.
    """
    if column_name in df.columns:
        value_counts = df[column_name].value_counts(dropna=False)
        print(f"\nValue counts for '{column_name}':\n")
        print(value_counts.to_frame().reset_index().rename(columns={'index': column_name, column_name: 'Count'}))
    else:
        print(f"Column '{column_name}' does not exist in the DataFrame.")


def remove_whitespace(df, columns):
    """
    Removes leading and trailing whitespace from specified columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to remove whitespace from.

    Returns:
    pd.DataFrame: The DataFrame with whitespace removed from the specified columns.
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].str.strip()  # Remove leading/trailing spaces
        else:
            print(f"Column '{col}' not found in DataFrame")
    return df



def explore_numerical_features(data, numerical_columns, bins=20, color_palette='Set1'):
    """
    Function to explore and visualize the distribution of numerical columns.

    Parameters:
    - data: Pandas DataFrame containing the dataset.
    - numerical_columns: List of numerical column names to explore.
    - bins: Number of bins for the histograms (default: 20).
    - color_palette: Seaborn color palette for the plots (default: 'Set1').
    """
    for colname in numerical_columns:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.histplot(data[colname], bins=bins, kde=True, color=sns.color_palette(color_palette)[0])
        plt.title(f'Distribution of {colname}', fontsize=14)
        plt.xlabel(colname, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True)

        plt.subplot(1, 2, 2)
        sns.boxplot(x=data[colname], color=sns.color_palette(color_palette)[1])
        plt.title(f'Boxplot of {colname}', fontsize=14)
        plt.xlabel(colname, fontsize=12)
        plt.grid(True)

        plt.tight_layout()
        plt.show()

def plot_numeric_distribution(data, target_column, colname):
    df = data.copy()
    if colname in ('capital-gain', 'capital-loss'):
        df = df[(df[colname] != 0) & ~df[colname].isnull()]
    else:
        df = df.dropna(subset=[colname])
    low_income = df.loc[df[target_column] == '<=50K', colname]
    high_income = df.loc[df[target_column] == '>50K', colname]

    plt.figure(figsize=(10, 6))
    plt.title(f'Distribution of {colname} by {target_column}')
    sns.histplot(low_income, label='<=50K', bins=20, color='blue', kde=True, stat="density", alpha=0.5)
    sns.histplot(high_income, label='>50K', bins=20, color='green', kde=True, stat="density", alpha=0.5)

    plt.xlim(0, df[colname].max())
    plt.xlabel(colname)
    plt.ylabel('Density')

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def compare_categorical_columns(data, target_column, categorical_columns):
    """
    Compare categorical columns by visualizing the distribution of categories
    for two classes of a target variable (e.g., income '<=50K' and '>50K').

    Parameters:
    - data: pandas DataFrame
    - target_column: the name of the target variable (binary classification).
    - categorical_columns: list of categorical columns to compare.
    """
    for colname in categorical_columns:
        low_income = data.loc[data[target_column] == '<=50K', colname]
        high_income = data.loc[data[target_column] == '>50K', colname]

        low_income_stats = low_income.value_counts(normalize=True)  # normalized to get proportions
        high_income_stats = high_income.value_counts(normalize=True)

        fig, ax = plt.subplots(figsize=(8, 6))

        low_bar = ax.barh(
            low_income_stats.index,
            low_income_stats.values,
            alpha=0.5,
            label='<=50K',
            color='blue'
        )

        high_bar = ax.barh(
            high_income_stats.index,
            high_income_stats.values,
            alpha=0.5,
            label='>50K',
            color='orange'
        )

        ax.set_title(f'Distribution in Column: {colname}', fontsize=14)
        ax.set_xlabel('Proportion', fontsize=12)
        ax.set_ylabel('Categories', fontsize=12)

        ax.legend(loc='best')
        plt.tight_layout()
        plt.show()


def explore_categorical_features(data, categorical_columns, top_n=20, color_palette='Set2'):
    """
    Function to explore and visualize the distribution of categorical columns.

    Parameters:
    - data: Pandas DataFrame containing the dataset.
    - categorical_columns: List of categorical column names to explore.
    - top_n: Number of top categories to display (default: 20).
    - color_palette: Seaborn color palette for the plots (default: 'Set2').
    """
    for colname in categorical_columns:
        value_counts = data[colname].value_counts().head(top_n)

        unique_count = len(value_counts)
        colors = sns.color_palette(color_palette, unique_count)

        plt.figure(figsize=(10, 6))
        plt.title(f'Distribution of Top {top_n} Categories in: {colname}', fontsize=14)

        sns.barplot(x=value_counts.values,
                    y=value_counts.index,
                    hue=value_counts.index,
                    palette=colors,
                    dodge=False,
                    legend=False)

        plt.xlabel('Count', fontsize=12)
        plt.ylabel(f'{colname} Categories', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()


def get_possible_categorical_features(data, max_categories):
    """
    Returns a dictionary containing the number of unique categories for each column in the dataset,
    where the number of categories is less than or equal to the user-defined maximum.
    This can be useful for identifying categorical features that can be encoded, such as using one-hot encoding.

    Parameters:
    - data (pd.DataFrame): The dataset as a pandas DataFrame.
    - max_categories (int): The maximum number of categories a column can have to be considered for encoding.

    Returns:
    - dict: A dictionary where keys are column names and values are the number of unique categories.
            Only columns with categories <= max_categories are included.
    """
    categorical_info = {}

    for column in data.columns:
        num_classes = data[column].nunique()

        if num_classes <= max_categories:
            categorical_info[column] = num_classes

    return categorical_info


def print_value_counts_for_categorical_features(data, max_categories):
    """
    Identifies categorical features with categories <= max_categories and prints the value counts for each.

    Parameters:
    - data (pd.DataFrame): The dataset as a pandas DataFrame.
    - max_categories (int): The maximum number of categories for features to be considered for encoding.
    """
    categorical_features = get_possible_categorical_features(data, max_categories)

    for feature in categorical_features.keys():
        print(f"\nValue counts for feature: '{feature}'")
        print(data[feature].value_counts(dropna=False).to_frame())
