import pandas
import argparse
import os
import logging
from src.utils.common import read_yaml
from src.utils.model_utils import load_binary
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score


STAGE = "TESTING"

logging.basicConfig(
    filename=os.path.join("logs","running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s : %(levelname)s : %(module)s]: %(message)s",
    filemode="a"
)


def main(config_path):
    try:
        configs = read_yaml(config_path)
        ARTIFACTS = configs["ARTIFACTS"]
        ARTIFACTS_PATH = ARTIFACTS["ARTIFACTS_PATH"]
        TEST_DATA = ARTIFACTS["TEST_LOADER_BIN"]
        MODEL = ARTIFACTS["MODEL"]
        model_path = os.path.join(ARTIFACTS_PATH,MODEL)
        test_data_path = os.path.join(ARTIFACTS_PATH,TEST_DATA)

        model = load_binary(model_path)
        test_data = load_binary(test_data_path)

        test = test_data.drop(["diagnosis"],axis=1)
        y_test = test_data.diagnosis

        y_score = model.predict(test)

        acc = accuracy_score(y_test ,y_score)
        f1 = f1_score(y_test, y_score)
        roc = roc_auc_score(y_test, y_score)


        logging.info(f'''
        Testing accuracy_score : {acc} \n
        Testing ROC score : {roc} \n 
        Testing f1 score : {f1}

        ''')
        return 

    except Exception as e:
        logging.info(e)
        raise e


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="configs/configs.yaml")
    parsed_arg = args.parse_args()
    try:
        logging.info(f"\n*****************************************")
        logging.info(f">>>>>>>>>>>>>>>>>> Stage {STAGE} has started  >>>>>>>>>>>>>")
        main(parsed_arg.config)
        logging.info(f">>>>>>>>>>>>>>>>>> Stage {STAGE} has completed  >>>>>>>>>>>>>")
    except Exception as e:
        logging.info(e)
        raise e
