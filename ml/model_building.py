import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import os
import joblib

from logger import Logger
from data_preprocessing import create_preprocessor

logger = Logger(name="ModelBuilding", log_file='model_building.log', level = 'debug')


def load_train_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.
    Args:
        file_path (str): The path to the CSV file.
    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """ 

    try:
        logger.info(f"Start loading data from {file_path}")
        X_train = pd.read_csv(f"{file_path}/X_train.csv")
        y_train = pd.read_csv(f"{file_path}/y_train.csv")
        logger.info(f"Data loaded successfully from {file_path}")
        return X_train, y_train
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}. Error: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"No data: {file_path} is empty. Error: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading data from {file_path}. Error: {e}")
        raise

def create_model(params: dict):
     model = RandomForestClassifier(**params, random_state=42)
     return model

def build_pipeline(params: dict):
     model = create_model(params)
     pipeline = Pipeline(
          [
               ('preprocessor', create_preprocessor()),
               ('model', model)
          ]
     )
     return pipeline

def train_model(X_train, y_train, params):


    """
    This train_model function is responsible for training the model with training data 
    from featured output using Logistic Regression Algorithm.

    Parameters
    ----------
    X_train : pd.DataFrame
        X_train data with selected feature 
        featured X train.
    
    y_train : pd.DataFrame
        y_train data ( Labeled data )
    
    params : dict

    Returns
    -------
    X_train_selected : ndarray
        Training dataset containing only the selected features.

    X_test_selected : ndarray
        Test dataset transformed using the selected features.

    classifier : sklearn.linear_model.LogisticRegression
        Trained Logistic Regression model. 

    Raises
    ------
    ValueError
        If the input data or parameters are invalid.

    Exception
        For any unexpected errors during feature selection.
    """

    try:
        logger.info("Model Training started")
        if X_train.shape[0] != y_train.shape[0]:
                       raise ValueError("The number of samples in X_train and y_train must be the same.")

        pipeline = build_pipeline(params=params)
        pipeline.fit(X_train,y_train)

        logger.info("Model Training completed successfully")
        return pipeline

    except ValueError as e:
         logger.error(f"Invalid input data : {e}")
         raise
    except Exception as e:
        logger.error("Unexpected error occured while model training")
        logger.error(f"{e}")

def save_model(pipeline, file_path):
    """
    This save_model function is responsible for saving the trained model 

    Parameters
    ----------
    model : GradientBoostClassifier
        trained model
    
    file_path : str
        Destination path where to save model.

    Raises
    ------
    FileNotFound
        If the destination path is not exist

    Exception
        For any unexpected errors during feature selection.
    """
    try:
        logger.info("Start model saving ")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        joblib.dump(pipeline, file_path)

        logger.info(f"pipeline  saved at {file_path}")

    except FileNotFoundError as e:
        logger.error(f"File path not found {e}")
        raise

    except Exception as e:  
        logger.error(f"Unexpected error during pipeline saving : {e}")
        raise
        


def main():
    logger.info("Starting Model building ..")
    file_path = "./data/train_test"


    model_params = {'n_estimators': 500, 
              'min_samples_split': 2,
              'min_samples_leaf': 10, 
              'max_features': 'sqrt', 
              'max_depth': 20, 
              'class_weight': 'balanced_subsample'}

    X_train, y_train = load_train_data(file_path)
    pipeline = train_model(X_train=X_train, y_train=y_train, params=model_params)

    # 3. Saving model
    file_path = './models/churn_pipeline.joblib'

    save_model(pipeline=pipeline, file_path=file_path)

if __name__ == "__main__":
     main()


