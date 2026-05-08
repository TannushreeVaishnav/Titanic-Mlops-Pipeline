import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.feature_store import RedisFeatureStore
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
import sys

logger=get_logger(__name__)

class DataPreprocessing:
    def __init__(self,train_data_path,test_data_path,feature_store:RedisFeatureStore):
        self.train_data_path=train_data_path
        self.test_data_path=test_data_path
        self.data=None
        self.test_data=None
        self.X_train=None
        self.X_test=None
        self.y_train=None
        self.y_test=None
        self.X_resampled=None
        self.y_resampled=None
        self.feature_store=feature_store
        logger.info("DataPreprocessing initialized")

    def load_data(self):
        try:
            self.data=pd.read_csv(self.train_data_path)
            self.test_data=pd.read_csv(self.test_data_path)
            logger.info("Data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException(str(e),sys)

    def preprocess_data(self):
        try:
            self.data['Age']=self.data['Age'].fillna(self.data['Age'].median())
            self.data['Embarked']=self.data['Embarked'].fillna(self.data['Embarked'].mode()[0])
            self.data['Fare']=self.data['Fare'].fillna(self.data['Fare'].median())
            self.data['Sex']=self.data['Sex'].map({'male':0,'female':1})
            self.data['Embarked']=self.data['Embarked'].astype('category').cat.codes

            self.data['FamilySize']=self.data['SibSp']+self.data['Parch']+1
            self.data['Isalone']=(self.data['FamilySize']==1).astype(int)
            self.data['HasCabin']=(self.data['Cabin'].notnull()).astype(int)
            self.data['Title']=self.data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False).map(
                {
                    'Mr': 0,
                    'Miss': 1,
                    'Mrs': 2,
                    'Master': 3,
                    'Rare': 4
                }
            ).fillna(4)
            self.data['Pclass_Fare']=self.data['Pclass']*self.data['Fare']
            self.data['Age_Fare']=self.data['Age']*self.data['Fare']
            logger.info("Data preprocessing completed")
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise CustomException(str(e),sys)

    def handle_imbalance(self):
        try:
            X=self.data[['Pclass','Sex','Age','Fare','Embarked','FamilySize','Isalone','HasCabin','Title','Pclass_Fare','Age_Fare']]
            y=self.data['Survived']
            smote=SMOTE(random_state=42)
            self.X_resampled,self.y_resampled=smote.fit_resample(X,y)
            logger.info("Imbalance handling completed")
        except Exception as e:
            logger.error(f"Error handling imbalance: {e}")
            raise CustomException(str(e),sys)

    def store_features_in_redis(self):
        try:
            batch_data={}
            for idx, row in self.data.iterrows():
                entity_id=row['PassengerId']
                features={
                    'Pclass': row['Pclass'],
                    'Sex': row['Sex'],
                    'Age': row['Age'],
                    'Fare': row['Fare'],
                    'Embarked': row['Embarked'],
                    'FamilySize': row['FamilySize'],
                    'Isalone': row['Isalone'],
                    'HasCabin': row['HasCabin'],
                    'Title': row['Title'],
                    'Pclass_Fare': row['Pclass_Fare'],
                    'Age_Fare': row['Age_Fare'],
                    'Survived': row['Survived']
                }
                batch_data[entity_id] = features

            self.feature_store.store_batch_features(batch_data)
            logger.info("Features stored in Redis successfully")
        except Exception as e:
            logger.error(f"Error storing features in Redis: {e}")
            raise CustomException(str(e),sys)

    def retrieve_features_from_redis(self,entity_id):
        try:
            features=self.feature_store.get_features(entity_id)
            if features:
                logger.info(f"Features retrieved for entity_id {entity_id}")
                return features
            else:
                logger.warning(f"No features found for entity_id {entity_id}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving features from Redis: {e}")
            raise CustomException(str(e),sys)

    def run(self):
        try:
            logger.info("Starting data preprocessing")
            self.load_data()
            self.preprocess_data()
            self.handle_imbalance()
            self.store_features_in_redis()
            logger.info("Data preprocessing completed successfully")
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            raise CustomException(str(e),sys)

if __name__=="__main__":
    feature_store=RedisFeatureStore(host='localhost', port=6379, db=0)
    data_preprocessing=DataPreprocessing(train_data_path=TRAIN_PATH,test_data_path=TEST_PATH,feature_store=feature_store)
    data_preprocessing.run()
    print(data_preprocessing.retrieve_features_from_redis(entity_id=332))
