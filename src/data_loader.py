import pandas as pd

def load_data(filepath):
    """
    Load the raw credit card transactions dataset from CSV.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    df = pd.read_csv(filepath)
    return df
