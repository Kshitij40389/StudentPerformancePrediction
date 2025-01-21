import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data...")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Define models and hyperparameters
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {"max_depth": [3, 5, 10]},
                "Random Forest": {"n_estimators": [10, 50, 100]},
                "Gradient Boosting": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
                "XGBoost": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
                "CatBoost": {"depth": [6, 8], "iterations": [100, 200]},
                "AdaBoost": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
                # Linear Regression does not have hyperparameters
                "Linear Regression": {},
            }

            # Evaluate models
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            logging.info(f"Model evaluation report: {model_report}")

            # Select the best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No model met the performance threshold.")

            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")

            # Save the best model
            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            # Test the best model
            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
