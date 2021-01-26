from typing import List

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.compose import ColumnTransformer

from ml_example.data import split_train_val_data
from ml_example.data.make_dataset import read_data
from ml_example.enities import SplittingParams
from ml_example.enities.feature_params import FeatureParams
from ml_example.features.build_features import make_features, column_transformer


@pytest.fixture()
def fitted_transformer(
    dataset: pd.DataFrame, feature_params: FeatureParams
) -> ColumnTransformer:
    fitted_transformer = column_transformer(feature_params)
    fitted_transformer.fit(dataset)
    return fitted_transformer


@pytest.fixture()
def dataset(dataset_path: str) -> pd.DataFrame:
    data = read_data(dataset_path)
    return data


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


def test_make_features(
    dataset: pd.DataFrame,
    fitted_transformer,
    feature_params: FeatureParams,
    dataset_path: str,
):
    features, target = make_features(fitted_transformer, dataset, feature_params)
    assert not pd.isnull(features).any().any()
    assert_allclose(
        np.log(dataset[feature_params.target_col].to_numpy()), target.to_numpy()
    )


def test_split_train_features(
    dataset: pd.DataFrame,
    fitted_transformer: ColumnTransformer,
    feature_params: FeatureParams,
    dataset_path: str,
):

    train_df, val_df = split_train_val_data(dataset, SplittingParams(val_size=0.2))

    train_features, _ = make_features(fitted_transformer, train_df, feature_params)
    val_features, _ = make_features(fitted_transformer, val_df, feature_params)

    assert train_features.columns.equals(val_features.columns)
