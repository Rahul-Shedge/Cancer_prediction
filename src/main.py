import mlflow


def main():
    exp_id = mlflow.set_experiment("Trail_run_1") # provide same experiment name in CLI with "--experiment-name Trail_run_1" do not use quotes
    with mlflow.start_run(run_name="main__",experiment_id=exp_id.experiment_id) as run:
        mlflow.run(".","prep",env_manager="local")
        mlflow.run(".","train",env_manager="local")
        mlflow.run(".","test",env_manager="local")


if __name__ == "__main__":
    main()


