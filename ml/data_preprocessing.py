# data_preprocessing.py

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from logger import Logger

logger = Logger(name='DataPreprocessing', log_file="data_preprocessing.py", level='debug')


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TARGET_COLUMN = "churn"

NUMERICAL_COL = [
    "tenure",
    "monthlycharges",
    "totalcharges",
    "internet_services_count"
]

CATEGORICAL_COL = [
    "seniorcitizen",
    "partner",
    "dependents",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paperlessbilling",
    "paymentmethod"
]


# --------------------------------------------------
# Load Data
# --------------------------------------------------

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

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------

def split_data(df, test_size=0.2, random_state=42):
    """
    Separate features and target and perform train-test split.
    """

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"{TARGET_COLUMN} column is not found in DataFrame")
    
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        logger.info("Data splitted successfully ..")

        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.erorr("Unexpected error occured while data splitting")
        logger.error(e)



# --------------------------------------------------
# IQR Outlier Capper
# --------------------------------------------------

class IQRCapper(BaseEstimator, TransformerMixin):
    """
    Caps numerical values using IQR-based lower and upper bounds.

    Important:
    The bounds are learned only from training data during pipeline.fit().
    """

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):

        X = pd.DataFrame(X).copy()

        self.feature_names_in_ = X.columns.to_numpy()

        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)

        iqr = q3 - q1

        self.lower_bounds_ = q1 - self.factor * iqr
        self.upper_bounds_ = q3 + self.factor * iqr

        return self

    def transform(self, X):

        X = pd.DataFrame(X).copy()

        X.columns = self.feature_names_in_

        X = X.clip(
            lower=self.lower_bounds_,
            upper=self.upper_bounds_,
            axis=1
        )

        return X


# --------------------------------------------------
# Create Preprocessor
# --------------------------------------------------

def create_preprocessor():

    numerical_pipeline = Pipeline(
        steps=[
            ("iqr_capper", IQRCapper()),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_COL
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COL
            )
        ]
    )

    return preprocessor


# --------------------------------------------------
# Save Train-Test Data
# --------------------------------------------------

def save_data(
    X_train,
    X_test,
    y_train,
    y_test,
    data_path
):
    """
    Save raw train/test datasets.

    We intentionally do NOT save transformed datasets here.
    """
    try:
        output_dir = os.path.join(data_path, "train_test")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving data into {output_dir}")

        X_train.to_csv(
            os.path.join(output_dir, "X_train.csv"),
            index=False
        )

        X_test.to_csv(
            os.path.join(output_dir, "X_test.csv"),
            index=False
        )

        y_train.to_csv(
            os.path.join(output_dir, "y_train.csv"),
            index=False
        )

        y_test.to_csv(
            os.path.join(output_dir, "y_test.csv"),
            index=False
        )
        logger.info(f"Data saved successfully to {output_dir}")

    except Exception as e:
        logger.error("Unexpected error occured while data saving .. ")
        logger.error(e)

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    input_path = "data/featured/data.csv"
    df = load_data(input_path)

    X_train, X_test, y_train, y_test = split_data(df)

    save_data(
        X_train,
        X_test,
        y_train,
        y_test,
        './data'
    )

    logger.info("Data preprocessing completed.")
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape : {X_test.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"y_test shape : {y_test.shape}")


if __name__ == "__main__":
    main()