import os
from typing import List

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from ml_example.data import read_data
from ml_example.enities import FeatureParams
from ml_example.features.build_features import column_transformer


@pytest.fixture()
def dataset_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "train_data_sample.csv")


@pytest.fixture()
def target_col():
    return "SalePrice"


@pytest.fixture()
def categorical_features() -> List[str]:
    return [
        "MSZoning",
        "Neighborhood",
        "RoofStyle",
        "MasVnrType",
        "BsmtQual",
        "BsmtExposure",
        "HeatingQC",
        "CentralAir",
        "KitchenQual",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "PavedDrive",
    ]


@pytest.fixture
def numerical_features() -> List[str]:
    return [
        "OverallQual",
        "MSSubClass",
        "OverallCond",
        "GrLivArea",
        "GarageCars",
        "1stFlrSF",
        "Fireplaces",
        "BsmtFullBath",
        "YrSold",
        "YearRemodAdd",
        "LotFrontage",
    ]


@pytest.fixture()
def features_to_drop() -> List[str]:
    return ["YrSold"]


@pytest.fixture()
def fitted_transformer(
    dataset: pd.DataFrame, feature_params: FeatureParams
) -> ColumnTransformer:
    fitted_transformer = column_transformer(feature_params)
    fitted_transformer.fit(dataset)
    return fitted_transformer


@pytest.fixture
def feature_params(
    categorical_features: List[str],
    features_to_drop: List[str],
    numerical_features: List[str],
    target_col: str,
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=features_to_drop,
        target_col=target_col,
        use_log_trick=True,
    )
    return params


@pytest.fixture()
def dataset(dataset_path: str) -> pd.DataFrame:
    data = read_data(dataset_path)
    return data
