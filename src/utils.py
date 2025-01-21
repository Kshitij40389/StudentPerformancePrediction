import os
import sys
import pickle
from itertools import product
from sklearn.metrics import r2_score
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for model_name, model in models.items():
            best_score = -float("inf")
            best_params = None

            # If parameters exist for the model, perform a manual search
            if model_name in params and params[model_name]:
                param_grid = params[model_name]
                keys, values = zip(*param_grid.items())
                for combination in product(*values):
                    param_dict = dict(zip(keys, combination))
                    try:
                        model.set_params(**param_dict)
                    except Exception as e:
                        # Skip models that do not support parameter updates
                        continue

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score = r2_score(y_test, y_pred)

                    if score > best_score:
                        best_score = score
                        best_params = param_dict

                # Set best parameters if found
                if best_params:
                    try:
                        model.set_params(**best_params)
                    except Exception:
                        pass  # Ignore models that can't update parameters

            else:  # Train without hyperparameter tuning
                model.fit(X_train, y_train)
                best_score = r2_score(y_test, model.predict(X_test))

            report[model_name] = best_score
        return report
    except Exception as e:
        raise CustomException(e, sys)
