
import pandas as pd
import os
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, average_precision_score
import json
import joblib

from logger import Logger

logger = Logger(name='model_evaluation',log_file='model_evaluation.log',level="debug")


def load_model(file_path):
    """
    This will load the saved trained model from a file 

    Parameter
    ---------
    file_path : str
        Location where trained joblib  file 
    
    Return
    ------
    
    model : GradientBossting classifier
        Trained model
    
    Raise
    -----

    ValueError:
        Given joblib file not found.
    
    Exception:
        Unexpected error during model loading

    """

    try:
        logger.info(f"Start loading pipeline from {file_path} ")

        pipeline = joblib.load(file_path)

        logger.info("pipeline Loaded successfully")
        return pipeline

    except FileNotFoundError as e:
        logger.error(f"File not found : {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error during model loading {e}")
        raise


def load_test_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.
    Args:
        file_path (str): The path to the CSV file.
    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """ 

    try:
        logger.info(f"Start loading data from {file_path}")
        X_test = pd.read_csv(f"{file_path}/X_test.csv")
        y_test = pd.read_csv(f"{file_path}/y_test.csv")
        logger.info(f"Data loaded successfully from {file_path}")
        return X_test, y_test
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}. Error: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"No data: {file_path} is empty. Error: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading data from {file_path}. Error: {e}")
        raise


def evaluate_model(pipeline, X_test, y_test, threshold):

    """
    This evaluate_model function is responsible for evaluating the trained model using test data.
    with different metrics 

    Parameters
    ----------

    model   : GradientBoosting Classifier
        A trained model 
    
    X_test   : pd.dataFrame
        test data X
    
    y_test   : pd.DataFrame
        test data y

    Returns
    -------
    metric_dict : Dictionary
        Contains all the metrics result with key-value pair

    Raises
    ------

    Exception
        Error during model evaluation.
    """
        

    try:
        y_pred_proba = pipeline.predict_proba(X_test)[:,1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test,y_pred)

        # Probability based metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)


        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1-score' : f1,
            'roc-auc': auc,
            'pr-auc' : pr_auc
        }
        logger.info('Model evaluation metrics calculated')
        logger.info(f"Evaluated metric : {metrics_dict}")
        return metrics_dict

    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics, file_path):

    """
    This save_metric function is responsible for saving the evaluation metrics dictionary into json file.

    Parameters
    ----------

    metric   : dictionary
        A evaluation metrics 
    
    file_path   : str
        file_path to json
    
    Raises
    ------
    
    FileNotFoundError:
        File not found 
    
    Exception:
        Error during model evaluation.
    """

        

    try:
        logger.info("Start svaing evaluation metrics to JSON file")
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info(f"Metrics saved a {file_path}")

    except FileNotFoundError as e:
        logger.error(f"File not found : {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error during saving metrics to json : {e}")
        raise


def main():

    try:
        # 1. Model loading 
        model_artifact = load_model('./models/churn_pipeline.joblib')

        model = model_artifact['pipeline']
        threshold = model_artifact['threshold']

        file_path = "./data/train_test"

        # 2. Test data loading
        X_test,y_test = load_test_data(file_path)
        logger.info("Test data loaded sucessfully")

        # 3. Model evaluation
        metrics = evaluate_model(model, X_test, y_test, threshold)    

        # 4. Model saved
        save_metrics(metrics, 'reports/metrics.json')

    except Exception as e:
        logger.error(f"Failed to complete the model evaluation process: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()