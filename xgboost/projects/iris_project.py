import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn import datasets
from sklearn.pipeline import Pipeline

def setup_logger(log_time: bool = True):
    """Sets up the logger with an optional timestamp.

    Parameters
    ----------
    log_time : bool, optional
        Whether to include the timestamp in the log messages (default is True).
    """
    logger = logging.getLogger()

    if logger.hasHandlers():
        logger.handlers.clear()

    log_format = "%(message)s"
    if log_time:
        log_format = "%(asctime)s - %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# Setup logger
setup_logger(log_time=False)
def load_data():
    """Load and return the Iris dataset as a pandas DataFrame."""
    iris = datasets.load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns=iris['feature_names'] + ['target'])
    return df

def preprocess_data(df):
    """Split the data into training and testing sets."""
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def build_model():
    """Build and return an XGBoost classifier."""
    xgb_clf = XGBClassifier(booster="gbtree",
                            objective="multi:softprob",
                            random_state=2,
                            n_jobs=-1,
                            eval_metric='mlogloss')
    return xgb_clf

def train_model(model, X_train, y_train):
    """Train the model and return the trained model."""
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return the accuracy score."""
    y_pred = model.predict(X_test)
    score = accuracy_score(y_pred, y_test)
    logging.info(f"Model Accuracy: {score:.4f}")
    return score

def hyperparameter_tuning(model, X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV."""
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    logging.info(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def main():
    """Main function to run the entire pipeline."""
    logging.info("Starting the pipeline...")

    start_time = time.time()

    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Build and train model
    model = build_model()
    model = train_model(model, X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Hyperparameter tuning
    model = hyperparameter_tuning(model, X_train, y_train)

    # Final evaluation after tuning
    evaluate_model(model, X_test, y_test)

    end_time = time.time()
    logging.info(f"Pipeline completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()