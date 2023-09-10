from sklearn.impute import SimpleImputer #Handle missing values
from sklearn.preprocessing import StandardScaler #handling feature scaling
from sklearn.preprocessing import OrdinalEncoder # ordinal encoding
# pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer # to combine two different pipelines
import os
import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).parent.parent))
from logger import logging
from exception import CustomException
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.utils import save_object




# Data transformation config
@dataclass  # no need for initialization constructor or self since we are using @dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts' , 'preprocessor.pkl')


# data transformation config class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()
        
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            
            # segregating numerical and categorical columns
            categorical_cols = ['cut' , 'color' , 'clarity']
            numerical_cols = ['carat' , 'depth' , 'table' , 'x' , 'y' , 'z']

            cut_categories = ['Fair' , 'Good' , 'Very Good' , 'Premium' , 'Ideal']
            color_categories = ['D' , 'E' , 'F' , 'G' , 'H' , 'I' , 'J']
            clarity_categories  = ['I1' , 'SI2' , 'SI1' , 'VS2' , 'VS1' , 'VVS2' , 'VVS1' , 'IF']
            
            # Numerical Pipeline
            num_pipeline = Pipeline(
                
                steps=[
                    ('imputer' , SimpleImputer(strategy='median')),
                    ('scalar' , StandardScaler())
                ]
            )


            # categorical pipeline
            cat_pipeline = Pipeline(
                steps = [
                    ('imputer' , SimpleImputer(strategy='most_frequent')),
                    ('encoder' , OrdinalEncoder(categories=[cut_categories , color_categories , clarity_categories])),
                    ('scalar' , StandardScaler())
                ]
            )


            preprocessor = ColumnTransformer([
                ('num_pipeline' , num_pipeline , numerical_cols),
                ('cat_pipeline' , cat_pipeline , categorical_cols)  
            ])
                
            logging.info('pipeline completed')
            return preprocessor
                
        except Exception as e:
            logging.info('Error in data transformation')
            raise CustomException(e,sys)
            
            
            
    def initiate_data_transformation(self, train_data_path , test_data_path):
        try:
            #Reading train and test data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            
            logging.info('Read Train and Test data')
            logging.info(f'Train DataFrame head \n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame head \n{test_df.head().to_string()}')
            
            logging.info('obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()
            
            target_column_name = 'price'
            drop_columns = [target_column_name , 'id']
            
            input_feature_train_df = train_df.drop(columns=drop_columns , axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=drop_columns , axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            #apply the transformation
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)            
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info('Applying transformation on training and testing datasets')
            
            train_arr = np.c_[input_feature_train_arr , np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr , np.array(target_feature_test_df)]
            
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path , obj = preprocessing_obj)

            logging.info('preprocessor pickle is created and saved')
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
            
        except Exception as e:
            logging.info('Exception occured in the initiate_data_transformation')
            raise CustomException(e,sys)
            

        
        
        
    
    