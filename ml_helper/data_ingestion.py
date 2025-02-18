import pandas as pd


def convert_non_strings_to_strings(df):
    """
    Converts non-string objects in the DataFrame to strings.
    
    Parameters:
    - df (pd.DataFrame): The dataset to process.
    
    Returns:
    - pd.DataFrame: The dataset with all non-string objects converted to strings.
    """
    return df.applymap(lambda x: str(x) if not isinstance(x, str) else x)



def validate_dataset(df, text_column='statement', label_column='status'):
    """Quickly checks if the dataset is correctly formatted."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Dataset is not a pandas DataFrame.")
    
    if not {text_column, label_column}.issubset(df.columns):
        raise ValueError(f"Missing required columns: {set([text_column, label_column]) - set(df.columns)}")
    
    if df.empty:
        raise ValueError("Dataset is empty.")
    
    if df.isnull().values.any():
        print(f"Warning: {df.isnull().sum().sum()} missing values found.")

    if df.duplicated().sum() > 0:
        print(f"Warning: {df.duplicated().sum()} duplicate rows found.")

    print("Dataset validation successful! The dataset is correctly formatted.")
    return True


