import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Define paths to model and preprocessor
            model_path = 'artifacts\\model.pkl'
            preprocessor_path = 'artifacts\\proprocessor.pkl'
            
            # Load model and preprocessor from disk
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Fill missing values (None or NaN) with a default value for categorical features
            features = features.fillna({'gender': 'unknown', 'Area': 'unknown', 'type': 'unknown'})

            # Transform features using preprocessor
            data_scaled = preprocessor.transform(features)
            
            # Predict using the loaded model
            preds = model.predict(data_scaled)
            
            return preds

        except Exception as e:
            # Raise custom exception in case of errors
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, gender:str, Area:str, type:str, verbal:int, logical:int):
        # Handle missing categorical values by assigning a default value
        self.gender = gender if gender is not None else "unknown"
        self.Area = Area if Area is not None else "unknown"
        self.type = type if type is not None else "unknown"
        self.verbal = verbal
        self.logical = logical

    def get_data_as_data_frame(self):
        try:
            # Create DataFrame from input data
            custom_data_input_dict = {
                "gender": [self.gender],
                "Area": [self.Area],
                "type": [self.type],
                "verbal": [self.verbal],
                "logical": [self.logical],
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            # Raise custom exception in case of errors
            raise CustomException(e, sys)
