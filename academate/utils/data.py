"""
Data utilities for Academate.
"""

import pandas as pd
import numpy as np
import re
import hashlib


def preprocess_data(df, content_column="Record"):
    """
    Preprocess a DataFrame for use in Academate.

    Args:
        df (pd.DataFrame): Input DataFrame
        content_column (str, optional): Column with content. Defaults to "Record".

    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    # Make a copy of the DataFrame
    df = df.copy()

    # Reset index
    df.reset_index(drop=True, inplace=True)

    # Ensure content column exists
    if content_column not in df.columns:
        raise ValueError(f"Content column '{content_column}' not found in DataFrame")

    # Fill missing values
    df[content_column] = df[content_column].fillna('')

    # Generate uniqueid if not present
    if 'uniqueid' not in df.columns:
        df['uniqueid'] = df.apply(lambda row: generate_uniqueid(row, content_column), axis=1)

    # Remove duplicates
    df_len = len(df)
    df.drop_duplicates(subset='uniqueid', keep='first', inplace=True)
    if len(df) < df_len:
        print(f"Removed {df_len - len(df)} duplicates")

    return df


def generate_uniqueid(row, content_column="Record"):
    """
    Generate a unique ID for a record based on content.

    Args:
        row (pd.Series): DataFrame row
        content_column (str, optional): Column with content. Defaults to "Record".

    Returns:
        str: Unique ID
    """
    # Normalize content
    content = row[content_column]
    if not isinstance(content, str):
        content = str(content)

    # Remove special characters except basic punctuation
    normalized = re.sub(r'[^a-zA-Z0-9 \-():]', '', content)

    # Normalize whitespace
    normalized = ' '.join(normalized.split())

    # Generate hash
    key_string = normalized
    id_record = hashlib.sha256(key_string.encode()).hexdigest()[:20]

    return id_record


def merge_dataframes(df1, df2, on="uniqueid", how="left"):
    """
    Merge two DataFrames with proper handling of overlapping columns.

    Args:
        df1 (pd.DataFrame): First DataFrame
        df2 (pd.DataFrame): Second DataFrame
        on (str, optional): Column to merge on. Defaults to "uniqueid".
        how (str, optional): Merge type. Defaults to "left".

    Returns:
        pd.DataFrame: Merged DataFrame
    """
    # Find overlapping columns
    overlapping_cols = set(df1.columns).intersection(set(df2.columns))
    overlapping_cols.remove(on)  # Remove the merge column

    # Rename overlapping columns in df2
    if overlapping_cols:
        df2 = df2.copy()
        rename_dict = {col: f"{col}_y" for col in overlapping_cols}
        df2.rename(columns=rename_dict, inplace=True)

    # Merge DataFrames
    merged_df = df1.merge(df2, on=on, how=how)

    return merged_df