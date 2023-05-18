import pandas as pd
import yaml
import os
import argparse
import logging
from src.utils.common import read_csv,read_yaml,creat_dir
from sklearn.preprocessing import StandardScaler
from src.utils.model_utils import save_binary
from sklearn.model_selection import train_test_split


creat_dir(["logs"])
STAGE="PREPROCESSING"

logging.basicConfig(
    filename=os.path.join("logs","running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)





def main(config_path):

    try:
        configs = read_yaml(config_path)
        path = os.path.join(configs['SOURCE_DATA_DIRS']["DIR"],configs['SOURCE_DATA_DIRS']["FILENAME"])  
        logging.info(f"Read input data path from configs file :{path}")
        #print("path:",path)
        data = read_csv(path)
        logging.info(f"Data from {path} read successfully.")
        #print(data.head())

        y = data.diagnosis
        y = y.map({"B":0,"M":1})
        print("*******",y)
        X = data.drop(["id","diagnosis","Unnamed: 32"],axis=1)


        to_drop = configs["PARAMS"]["COLS_TO_DROP"]
        logging.info(f"Cols drop from input data {to_drop}")
        filtered_X = X.drop(to_drop,axis=1)

        


        X_train,X_test,y_train,y_test = train_test_split(filtered_X,y,train_size=configs["PARAMS"]["TRAIN_SPLIT_SIZE"])
        print(X_train.isna().sum())
        print(y_train.isna().sum())


        ARTIFACTS = configs["ARTIFACTS"]

        sc1 = StandardScaler()

        sc1.fit(X_train)
        artifacts_path = ARTIFACTS["ARTIFACTS_PATH"]
        creat_dir([artifacts_path])
        scaler_path = os.path.join(artifacts_path,ARTIFACTS["SCALER_BIN"])
        save_binary(sc1, scaler_path)
        
        train_path = os.path.join(artifacts_path,ARTIFACTS["TRAIN_LOADER_BIN"])
        test_path = os.path.join(artifacts_path,ARTIFACTS["TEST_LOADER_BIN"])
        
        

        X_train_f = sc1.transform(X_train)
        X_train_f1 = pd.DataFrame(X_train_f,columns=filtered_X.columns).reset_index(drop=True)


        train = pd.concat([y_train.reset_index(drop=True),X_train_f1],axis=1)
        X_test_f = sc1.transform(X_test)
        X_test_f1 = pd.DataFrame(X_test_f,columns=filtered_X.columns).reset_index(drop=True)
        test = pd.concat([y_test.reset_index(drop=True),X_test_f1],axis=1)

        save_binary(train, train_path)
        save_binary(test,test_path)
        return 

    except Exception as e:
        logging.info(e)
        raise e




if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="configs/configs.yaml")
    parsed_args = args.parse_args()
    try:
        logging.info("\n*********************************")
        logging.info(f">>>>>>>>>> stage {STAGE} started <<<<<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>>>>>>>>>>>> stage {STAGE} completed.<<<<< \n")
    except Exception as e:
        logging.info(e)
        raise e        

















