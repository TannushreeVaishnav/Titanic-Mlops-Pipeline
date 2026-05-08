from src.logger import get_logger
from src.custom_exception import CustomException
import pandas as pd
from src.feature_store import RedisFeatureStore
from sklearn.model_selection import train_test_split
import os
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import pickle

logger=get_logger(__name__)

class ModelTraining:
    def __init__(self,feature_store:RedisFeatureStore,model_save_path="artifacts/model/model.pkl"):
        self.feature_store=feature_store
        self.model_save_path=model_save_path
        self.model=None

        os.makedirs(self.model_save_path, exist_ok=True)
        logger.info("ModelTraining initialized")
    
    def load_data_from_redis(self,entity_ids):
        try:
            logger.info("Loading data from Redis")
            data=[]
            for entity_id in entity_ids:
                features=self.feature_store.get_features(entity_id)
                if features:
                    data.append(features)
                else:
                    logger.warning(f"No features found for entity_id: {entity_id}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from Redis: {e}")
            raise CustomException(str(e),sys)
    
    def prepare_data(self):
        try:
            entity_ids=self.feature_store.get_all_entities()
            train_entity_ids, test_entity_ids=train_test_split(entity_ids,test_size=0.2,random_state=42)
            train_data=self.load_data_from_redis(train_entity_ids)
            test_data=self.load_data_from_redis(test_entity_ids)
            train_df=pd.DataFrame(train_data)
            test_df=pd.DataFrame(test_data)
            X_train=train_df.drop('Survived',axis=1)
            y_train=train_df['Survived']
            X_test=test_df.drop('Survived',axis=1)
            y_test=test_df['Survived']
            logger.info("Data preparation completed")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise CustomException(str(e),sys)
        
    def hyperparameter_tuning(self,X_train,y_train):
        try:
            param_distribution={
                'n_estimators':[100,200,300],
                'max_depth':[10,20,30],
                'min_samples_split':[2,5],
                'min_samples_leaf':[1,2],
            }
            rf=RandomForestClassifier(random_state=42)
            random_search=RandomizedSearchCV(rf,param_distribution,n_iter=10,cv=3,random_state=42,scoring='accuracy')
            random_search.fit(X_train,y_train)
            logger.info(f"Best hyperparameters found: {random_search.best_params_}")
            return random_search.best_estimator_
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {e}")
            raise CustomException(str(e),sys)
    
    def train_and_evaluate(self,X_train,y_train,X_test,y_test):
        try:
            best_rf=self.hyperparameter_tuning(X_train,y_train)
            y_pred=best_rf.predict(X_test)
            accu=accuracy_score(y_test,y_pred)
            logger.info(f"Model accuracy: {accu:.4f}")
            self.save_model(best_rf)
            return accu

        except Exception as e:
            logger.error(f"Error during model training and evaluation: {e}")
            raise CustomException(str(e),sys)
        
    def save_model(self,model):
        try:
            model_filename=f"{self.model_save_path}/model.pkl"
            with open(model_filename, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"Model saved at {model_filename}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise CustomException(str(e),sys)

    def run(self):
        try:
            logger.info("Starting model training process")
            X_train, y_train, X_test, y_test=self.prepare_data()
            acc=self.train_and_evaluate(X_train,y_train,X_test,y_test)

            logger.info(f"Model training process completed")
        except Exception as e:
            logger.error(f"Model training process failed: {e}")
            raise CustomException(str(e),sys)
        
if __name__=="__main__":
    

    '''Before running this script, ensure that the data preprocessing step has been completed and features are stored in Redis.
    from src.feature_store import RedisFeatureStore
    store = RedisFeatureStore(host='localhost', port=6379, db=0)
    store.client.flushdb()
    '''

    feature_store=RedisFeatureStore()
    model_trainer=ModelTraining(feature_store)
    model_trainer.run()


        