import numpy as np
import pandas as pd
import os
import sys
from sklearn.linear_model  import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from src.logger import logging
from src.exception import CustomException
from pathlib import Path 
sys.path.append(str(Path(__file__).parent.parent))
from dataclasses import dataclass
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerconfig:
    trained_model_file_path = os.path.join('artifacts' , 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerconfig()
        
    def initiate_model_training(self, train_array , test_array):
        try:
            logging.info('splitting independent and dependent variables from train array and test array')
            X_train , y_train , X_test , y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'Elasticnet':ElasticNet(),
                'DecisionTree': DecisionTreeRegressor()
            }
            
            model_report:dict = evaluate_model(X_train , y_train , X_test , y_test , models)
            print(model_report)
            print("===============================================================================================")
            logging.info(f'Model Report {model_report}')
            
            # To get the best score from the dictionary
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
       
            
        except Exception as e:
            logging.info('Error in running model Trainer')
            raise CustomException(e,sys)
        

