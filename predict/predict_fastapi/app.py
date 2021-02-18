import logging
import os
import pickle
from typing import List, Union, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, conlist
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


def load_object(path: str) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


class HousePricesModel(BaseModel):
    data: List[conlist(Union[float, str, None], min_items=80, max_items=180)]
    features: List[str]


class PriceResponse(BaseModel):
    id: str
    price: float


model: Optional[RandomForestRegressor] = None
feature_transformer: Optional[ColumnTransformer] = None


def make_predict(
    data: List,
    features: List[str],
    model: RandomForestRegressor,
    feature_transformer: ColumnTransformer,
) -> List[PriceResponse]:
    data = pd.DataFrame(data, columns=features)
    ids = [int(x) for x in data["Id"]]
    features = feature_transformer.transform(data)
    predicts = np.exp(model.predict(features))

    return [
        PriceResponse(id=id_, price=float(price)) for id_, price in zip(ids, predicts)
    ]


app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model():
    global model
    global feature_transformer
    serialization_path = os.getenv("PATH_TO_SERIALIZATION")
    if serialization_path is None:
        err = f"serialization_path {serialization_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_object(os.path.join(serialization_path, "model.pkl"))
    feature_transformer = load_object(
        os.path.join(serialization_path, "transformer.pkl")
    )


@app.get("/predict/", response_model=List[PriceResponse])
def predict(request: HousePricesModel):
    return make_predict(request.data, request.features, model, feature_transformer)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
