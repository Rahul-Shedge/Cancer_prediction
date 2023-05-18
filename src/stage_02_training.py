import os
import argparse
import mlflow
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.utils.common import read_yaml,creat_dir
from src.utils.model_utils import load_binary,save_binary
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")
# exp_id = mlflow.set_experiment('case-study-one')

STAGE = "TRAINING"

logging.basicConfig(
    filename=os.path.join("logs","running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s:%(levelname)s:%(module)s]:%(message)s",
    filemode="a"

)


def training(X,y,configs):
    try:
        logging.info("Training Started........")
        clf = RandomForestClassifier(random_state=configs["PARAMS"]["RANDOM_SEED"])
        clf.fit(X,y)
        ypred = clf.predict(X)
        mlflow.log_param("Seed",configs["PARAMS"]["RANDOM_SEED"])

        acc = accuracy_score(y,ypred)
        roc = roc_auc_score(y,ypred)
        f1 = f1_score(y, ypred)



        
        logging.info("Training completed.")
        return acc,roc,f1,clf
    except Exception as e:
        logging.info(e)
        raise e






def main(config_path):
    configs = read_yaml(config_path)

    ARTIFACTS = configs["ARTIFACTS"]
    ARTIFACTS_PATH = ARTIFACTS["ARTIFACTS_PATH"]
    TRAIN_LOADER_BIN = ARTIFACTS["TRAIN_LOADER_BIN"]
    SCALER_BIN =  ARTIFACTS["SCALER_BIN"]
    MODEL = ARTIFACTS["MODEL"]


    train_data_path = os.path.join(ARTIFACTS_PATH,TRAIN_LOADER_BIN)
    train = load_binary(train_data_path)

    X_train = train.drop("diagnosis",axis=1)
    y_train = train.diagnosis
    #with mlflow.start_run(run_name="Training",experiment_id=exp_id.experiment_id):

    training_acc , training_roc ,training_f1 , clf = training(X_train,y_train,configs)

    logging.info(f'''
    Training accuracy_score : {training_acc} \n
    Training ROC score : {training_roc} \n 
    Training f1 score : {training_f1}

    ''')

    mlflow.log_metric("Training_accuracy", training_acc)
    mlflow.log_metric("Training_roc_auc_score", training_roc)
    mlflow.log_metric("Training_f1_score", training_f1)

    model_path = os.path.join(ARTIFACTS_PATH,MODEL)

    save_binary(clf, model_path)
    mlflow.sklearn.log_model(clf, "model")



    # model = load_binary(model_path)
    # y_score = model.predict(X_train)
    # print(accuracy_score(y_train, y_score))

    return 


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="configs/configs.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n*****************************************************")
        logging.info(f">>>>>>>>>>>>>> stage {STAGE} started >>>>>>>>>>>>>>>>>>>>>>")
        main(parsed_args.config)
        logging.info(f">>>>>>>>>>>>>>> stage {STAGE} completed.<<<<< \n")
    except Exception as e:
        logging.info(e)
        raise e                


