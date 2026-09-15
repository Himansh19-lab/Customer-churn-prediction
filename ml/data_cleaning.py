
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


def removing_irrelevant_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    This method is used to droping the irrelevant feature

    Args :
        pd.DataFrame 
    
    Return: 
        pd.DataFrame
    """
    try:
        column_to_drop = ['gender','phoneservice']
        df = df.drop(columns=column_to_drop)
        logger.info(f'{column_to_drop} droped sucessfully')
        return df
    except Exception as e:
        logger.error("Unexpected error occured during feature droping.. ")
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
        df = removing_irrelevant_feature(df)
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


