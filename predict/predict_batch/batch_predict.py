import os
import pickle

import click
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer


def load_object(path: str) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


def batch_predict(
    path_to_data: str,
    path_to_serialization: str,
    output: str,
    model_name: str = "model.pkl",
    transformer_name: str = "transformer.pkl",
):
    model_path = os.path.join(path_to_serialization, model_name)
    transformer_path = os.path.join(path_to_serialization, transformer_name)
    validate(model_path, path_to_data, transformer_path)

    model = load_object(model_path)
    transformer: ColumnTransformer = load_object(transformer_path)

    data = pd.read_csv(path_to_data)
    transformed_data = transformer.transform(data)
    predicts = np.exp(model.predict(transformed_data))
    ids = data["Id"]
    predict_df = pd.DataFrame(list(zip(ids, predicts)), columns=["Id", "Predict"])
    predict_df.to_csv(output, index=False)


def validate(model_path: str, path_to_data: str, transformer_path: str):
    if not os.path.exists(path_to_data):
        raise FileExistsError(f"{path_to_data} doesn't exists")
    if not os.path.exists(model_path):
        raise FileExistsError(f"{model_path} doesn't exists")
    if not os.path.exists(transformer_path):
        raise FileExistsError(f"{transformer_path} doesn't exists")


@click.command(name="batch_predict")
@click.argument("PATH_TO_DATA", default=os.getenv("PATH_TO_DATA"))
@click.argument("PATH_TO_SERIALIZATION", default=os.getenv("PATH_TO_SERIALIZATION"))
@click.argument("OUTPUT", default=os.getenv("OUTPUT"))
def batch_predict_command(path_to_data: str, path_to_serialization: str, output: str):
    batch_predict(path_to_data, path_to_serialization, output)


if __name__ == "__main__":
    batch_predict_command()
