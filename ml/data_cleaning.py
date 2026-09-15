
import pandas as pd
from logger import Logger
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

logger = Logger(name="DataCleaning", log_file="data_cleaning.log", level="debug")

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.
    Args:
        file_path (str): The path to the CSV file.
    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """ 

    try:
        logger.info(f"Start loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path} with shape {df.shape}")
        return df
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}. Error: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"No data: {file_path} is empty. Error: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading data from {file_path}. Error: {e}")
        raise

def removing_outliers(df: pd.DataFrame):
    """
    This method is for detecting and removing outliers
    """

    try:
        numerical_col = ['tenure', 'monthlycharges', 'totalcharges']
        outliers = []
        for col in numerical_col:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier = ((df[col] > upper_bound ) | (df[col] < lower_bound)).sum()
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            outliers.append(outlier)
            logger.info(f"{col} :  {outlier} outliers")

        if any(outliers):
            logger.info(f"Outlier successfully removed ")
        else:
            logger.info(f"No outliers found in data")
        return df


    except Exception as e:
        logger.error(f"Unexpected error occured while removing outliers")
        logger.error(f"Erorr : {e}")

def remove_duplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    This method is check and remove duplicates in dataFrame
    Args: 
        pd.Dataframe : Input dataFrame to check and remove duplicate
    Return:
        pd.DataFrame
    """

    try:
        duplicate_values = df.duplicated().sum()
        logger.info(f"Total duplicated values found : {duplicate_values}")
        if duplicate_values:
            df = df.drop_duplicates()
            logger.info(f"Duplicated value removed succesfully")
        return df

    except Exception as e:
        logger.error(f"Unexpected error occur while checking and removing duplicates")
        logger.error(f"Error : {e}")


def check_impossible_data(df: pd.DataFrame):
    """
    This method is to check impossible value of dataFrame

    """

    try:
        logger.info(f"Negative tenure : {(df['tenure'] < 0 ).sum()}")
        logger.info(f"tenure > 100  : {(df['tenure'] > 100 ).sum()}")
        logger.info(f"Negative monthlycharges : {(df['monthlycharges'] < 0 ).sum()}")
        logger.info(f"Negative totalcharges : {(df['totalcharges'] < 0 ).sum()}")

    except Exception as e:
        logger.error("Unexpected error occured while checking impossible values")
        logger.error(e)


def get_data_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    This method is checking missing/duplicate/outlier data 
    """
    try:
        missing_data = df.isnull().sum().sum()
        logger.info(f"Total Missing data {missing_data}")
        df = remove_duplicate(df)
        check_impossible_data(df)
        return df

    except Exception as e:
        logger.error(f"Unexpected error occured while data preprocess")
        logger.error(f"Erorr : {e}")


def save_data(df: pd.DataFrame, data_path: str):
    """
    Save a pandas DataFrame to a CSV file.
    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        data_path (str): The path where the CSV file will be saved.
    """
    try:
        data_path = os.path.join(data_path, "cleaned")
        logger.info(f"Saving data into {data_path}")
        os.makedirs(data_path, exist_ok=True)
        file_path = os.path.join(data_path, "data.csv")
        df.to_csv(file_path, index=False)
        logger.info(f"Data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"An error occurred while saving data to {file_path}. Error: {e}")
        raise
if __name__ == "__main__":

    logger.info("Data Cleaning started...")
    file_path = "./data/raw/data.csv"
    df = load_data(file_path)
    df_clean = get_data_clean(df)
    save_data(df = df_clean, data_path='./data')


