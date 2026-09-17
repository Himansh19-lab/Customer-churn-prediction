
from logger import Logger
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import os


logger = Logger(name='FeatureEngineering',log_file='feature_engineering.log',level='debug')



def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.
    Args:
        file_path (str): The path to the CSV file.
    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """ 

    try:
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

def creating_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    This method adds relevvant fetaure to the dataFrame 

    Args:
        pd.DataFrame : df 
    
    Return:
        pd.DataFrame : Updated dataFrame with new column feature

    """
    try:
        df['internet_services_count'] = (
        (df['onlinesecurity'] == 'Yes').astype(int) +
        (df['onlinebackup'] == 'Yes').astype(int) +
        (df['deviceprotection'] == 'Yes').astype(int) +
        (df['techsupport'] == 'Yes').astype(int) +
        (df['streamingtv'] == 'Yes').astype(int)
        )
        return df 
    except Exception as e:
        logger.error("Unexpected error occured while create a new feature")
        logger.error(f"Error : {e}")

def data_split(df: pd.DataFrame):
    """
    Separates the input features and target variable, 
    then splits the data into training and testing datasets. 
    Parameters 
    ---------- 
    df : pd.DataFrame Input dataset containing the target column 'churn'.
    
    Returns
    ------- 
    X_train : pd.DataFrame Training features. 
    X_test : pd.DataFrame Testing features. 
    y_train : pd.Series Training target. 
    y_test : pd.Series Testing target.

    """
    try:

        # validate target column
        if 'churn'  not in df.columns:
            raise ValueError(f"Target column churn is not found in DataFrame")

        # seperate input and output dataset
        X = df.drop(columns=['churn'])
        y = df['churn']

        logger.info( f"Input and target data separated successfully. " f"X shape: {X.shape}, y shape: {y.shape}" )


        ## Dataset split
        X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
        logger.info( f"Train/test split completed successfully. " f"X_train: {X_train.shape}, " f"X_test: {X_test.shape}, " f"y_train: {y_train.shape}, " f"y_test: {y_test.shape}" )
        return X_train,X_test, y_train,y_test

    
    except Exception as e:
        logger.error("Unexpected error occured while data splitting")
        logger.error(f"Error : {e}")


def save_data(df: pd.DataFrame, data_path: str):
    """
    Save a pandas DataFrame to a CSV file.
    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        data_path (str): The path where the CSV file will be saved.
    """
    try:
        data_path = os.path.join(data_path, "featured")
        logger.info(f"Saving data into {data_path}")
        os.makedirs(data_path, exist_ok=True)
        file_path = os.path.join(data_path, "data.csv")
        df.to_csv(file_path, index=False)
        logger.info(f"Data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"An error occurred while saving data to {file_path}. Error: {e}")
        raise

if __name__ == "__main__":

    logger.info("Feature Engineering started...")
    file_path = './data/cleaned/data.csv'
    df = load_data(file_path)
    df = creating_feature(df)
    save_data(df, './data')
    