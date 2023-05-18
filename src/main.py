import mlflow


def main():
    with mlflow.start_run(run_name="main") as run:
        mlflow.run(".","prep",env_manager="local")
        mlflow.run(".","train",env_manager="local")
        mlflow.run(".","test",env_manager="local")


if __name__ == "__main__":
    main()


