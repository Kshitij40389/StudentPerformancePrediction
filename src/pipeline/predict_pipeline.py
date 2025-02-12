import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path ='artifacts\model.pkl'
            preprocessor_path = 'artifacts\proprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,gender:str, Area:str, type:str, verbal:int, logical:int):
        self.gender = gender
        self.Area = Area
        self.type = type
        self.verbal = verbal
        self.logical = logical

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "Area": [self.Area],
                "type": [self.type],
                "verbal": [self.verbal],
                "logical": [self.logical],
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)