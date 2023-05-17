
import pandas as pd
import yaml
import os
import logging



def read_csv(path):
    data = pd.read_csv(path)
    return data

def read_yaml(path):
    with open(path,"r") as yaml_file:
        content = yaml.safe_load(yaml_file)
        print(content)
    return content

def creat_dir(path_to_dir:list)->None:
    for path in path_to_dir:
        os.makedirs(path,exist_ok=True)

