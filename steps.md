
  ### Mlflow MLOps Steps :-


```
conda create --prefix ./env python=3.8.0 -y && conda activate ./env
```
then run

```
pip install -r requirements.txt
```
Create entire code structure.

```
- configs
  - configs.yaml

- research
  - research.ipynb

- src
  - utils
    - __init__.py
    - common.py
    - model_utils.py
  - main.py
  - stage_01_preprocesing.py
  - stage_02_training.py
  - stage_03_testing.py

- requirements.txt
- .gitignore
- README.md
- MLprojects
- setup.py
```

![screenshot](/screenshot/Screenshot_2023-05-19_214028.png)














```
pip install -e .
```

Now add mlflow code to source code 


------------------------------------------------
                                  MLFLOW  METHOD 1

------------------------------------------------

add below line in the code. 
```
mlflow.set_tracking_uri("http://127.0.0.1:5000")
exp_id = mlflow.set_experiment('case-study-one')
```

Need to add main code with mlflow.start_run

```
with mlflow.start_run(run_name="Training",experiment_id=exp_id.experiment_id):

    training_acc , training_roc ,training_f1 , clf = training(X_train,y_train,configs)

    logging.info
    (
    f'''

    Training accuracy_score : {training_acc} \n
    Training ROC score : {training_roc} \n 
    Training f1 score : {training_f1}
    
    '''
    )

    mlflow.log_metric("Training_accuracy", training_acc)
    mlflow.log_metric("Training_roc_auc_score", training_roc)
    mlflow.log_metric("Training_f1_score", training_f1)


    scaler_path = os.path.join(ARTIFACTS_PATH,ARTIFACTS["SCALER_BIN"])
    # load_binary(scaler_path)
    mlflow.log_artifact(scaler_path)

    model_path = os.path.join(ARTIFACTS_PATH,MODEL)


    save_binary(clf, model_path)
    mlflow.sklearn.log_model(clf, "model",registered_model_name="RFmodel")
```

& then run the respective file:

```
python stage_02_training.py
```

then we can visit http://127.0.0.1:5000  
to check the artifacts and registry

------------------------------------------------->file by file method end.<---------------------------------------



------------------------------------------------
                                  MLFLOW  METHOD 2

------------------------------------------------

Simply add mlflow code to source code (log_model,artifacts and params )


```
conda env export > conda.yaml
```

***Edits conda.yaml***

Create **MLProject**

add all the 

![screenshot](/screenshot/Screenshot_2023-05-20_002416.png)


from CLI:

**Keep terminal active**

```
mlflow ui
```
From another terminal run:

```
mlflow run . -P experiment_name=<name> --experiment-name <name>
```
OR to with default experiment :
```
mlflow run .
```
```
./mlruns 
```
&&
```
./mlartifacts
```

Folders will gets created.


------------------------------------------------
                          MLFLOW  METHOD 3 Storing Artifacts to database

------------------------------------------------

Make sure you set Tracking uri in **System variables**:

![screenshot](/screenshot/Screenshot_2023-05-20_002743.png)

With sqlite db:

**Keep Terminal Active**

```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 --port 5000
```

    Folder
    ./mlflow.db 
    will get created

From another terminal

```
mlflow models serve --model-uri models:/<model-name>/<level(Staging/Production etc)> -p 1234 --no-conda
```

Now we can do prediction :

    WITH 

    "http://localhost:1234/invocations"

    As endpoint.




  **SEE BELOW :**

![screenshot](/screenshot/Screenshot_2023-05-20003745.png)


**DATA INPUT FORMAT BELOW :** JSON

```

{
  "dataframe_split":{
    "columns": [
      "texture_mean","smoothness_mean","compactness_mean","symmetry_mean","fractal_dimension_mean",
      "texture_se",
      "area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se",
      "area_worst","smoothness_worst","compactness_worst",
      "concavity_worst","concave points_worst","symmetry_worst",
      "fractal_dimension_worst"
      ]
      ,

  "data":
    [
      [
      1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,
      0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,
      0.7119,0.2654,0.4601,0.1189
      ]
    ]
  }
}

```