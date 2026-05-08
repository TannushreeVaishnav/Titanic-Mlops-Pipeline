from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing
from src.model_training import ModelTraining
from src.feature_store import RedisFeatureStore
from config.database_config import DB_CONFIG
from config.path_config import *



if __name__=="__main__":
    data_ingestion=DataIngestion(db_params=DB_CONFIG,output_dir=RAW_DIR)
    data_ingestion.run()



    feature_store=RedisFeatureStore(host='localhost', port=6379, db=0)
    data_preprocessing=DataPreprocessing(train_data_path=TRAIN_PATH,test_data_path=TEST_PATH,feature_store=feature_store)
    data_preprocessing.run()


    feature_store=RedisFeatureStore()
    model_trainer=ModelTraining(feature_store)
    model_trainer.run() 