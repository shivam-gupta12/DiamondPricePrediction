import os
import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).parent.parent))
from logger import logging
from exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass



# initialize the data ingestion configuration
@dataclass  # no need for initialization constructor or self since we are using @dataclass
class DataIngestionconfig:
    train_data_path = os.path.join('artifacts' , 'train.csv')
    test_data_path = os.path.join('artifacts' , 'test.csv')
    raw_data_path = os.path.join('artifacts' , 'raw.csv')


# create a data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()
        
    def initiate_data_ingestion(self):
        logging.info('Data ingestion method starts')
        
        try:
            df = pd.read_csv(os.path.join('notebooks/data' , 'gemstone.csv'))
            logging.info('Dataset read as pandas dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path , index=False)
            
            logging.info('Train Test split')
            train_set , test_set = train_test_split(df , test_size=0.33 , random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path , index = False , header = True)            
            test_set.to_csv(self.ingestion_config.test_data_path , index = False , header = True)
            
            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logging.info('Error ocurred in data ingestion config')