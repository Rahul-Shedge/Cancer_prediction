import joblib
import logging




def save_binary(model_config,filepath):
    joblib.dump(model_config,filepath)
    logging.info(f"model config is saved at {filepath}")

def load_binary(filepath):
    bin_file = joblib.load(filepath)
    logging.info(f"model config is loaded from {filepath}")
    return bin_file

