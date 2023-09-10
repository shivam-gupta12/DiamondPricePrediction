import os
import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).parent.parent))
from logger import logging
from exception import CustomException
from src.components.data_transformation import DataTransformation
import pandas as pd

from components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path , test_data_path = obj.initiate_data_ingestion()
    print(train_data_path , test_data_path)
    
    data_transformation = DataTransformation()
    
    train_arr , test_arr , _ = data_transformation.initiate_data_transformation(train_data_path=train_data_path , test_data_path=test_data_path)
    
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_array=train_arr , test_array=test_arr)
    
    