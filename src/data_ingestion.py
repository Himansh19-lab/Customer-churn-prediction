
from logger import Logger
import pandas as pd
import os

logger =  Logger(name="DataIngestion", log_file="data_ingestion.log", level="debug")


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.
    Args:
        file_path (str): The path to the CSV file.
    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        logger.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully with shape {data.shape}")
        return data
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}. Error: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"No data: {file_path} is empty. Error: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading data from {file_path}. Error: {e}")
        raise

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names in a pandas DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame whose columns need to be standardized.
    Returns:
        pd.DataFrame: The DataFrame with standardized columns.
    """
    try:
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        logger.info(f"Columns standardized successfully. New columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        logger.error(f"An error occurred while standardizing columns. Error: {e}")
        raise

def standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    This Method is to clean the data 
    Example - 
        - Converting column to desire data types
        - mapping target churn column to 0 or 1 
        - Removing customer ID column
    """

    try:
        # Converting totalcharges column to numeric & fillna to 0.
        df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
        df['totalcharges'] = df['totalcharges'].fillna(0)

        # Mapping churn column to 0 or 1
        df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})

        # Droping ID columns
        df.drop(columns=['customerid'], inplace=True)
        logger.info(f"Data preprocess successfully")
        return df

    except Exception as e:
        logger.error(f"An error occured while processing the data. Error: {e}")

def save_data(df: pd.DataFrame, data_path: str):
    """
    Save a pandas DataFrame to a CSV file.
    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        data_path (str): The path where the CSV file will be saved.
    """
    try:
        raw_data_path = os.path.join(data_path, "raw")
        logger.info(f"Saving data into {raw_data_path}")
        os.makedirs(raw_data_path, exist_ok=True)
        file_path = os.path.join(raw_data_path, "data.csv")
        df.to_csv(file_path, index=False)
        logger.info(f"Data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"An error occurred while saving data to {file_path}. Error: {e}")
        raise



if __name__ == "__main__":

    logger.info("Starting data ingestion process....")
    file_path = "./data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_data(file_path)
    df_standardize = standardize_columns(df)
    df_preprocessed = standardize_data(df_standardize)
    save_data(df_preprocessed, "./data")