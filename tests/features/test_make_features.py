from typing import List

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from ml_example.data import split_train_val_data
from ml_example.data.make_dataset import read_data
from ml_example.enities import SplittingParams
from ml_example.enities.feature_params import FeatureParams
from ml_example.features.build_features import make_features


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
    feature_params: FeatureParams, dataset_path: str,
):
    data = read_data(dataset_path)
    features, target = make_features(data, feature_params)
    assert not pd.isnull(features).any().any()
    assert_allclose(
        np.log(data[feature_params.target_col].to_numpy()), target.to_numpy()
    )


def test_split_train_features(feature_params: FeatureParams, dataset_path: str):
    data = read_data(dataset_path)

    train_df, val_df = split_train_val_data(data, SplittingParams(val_size=0.2))

    train_features, _ = make_features(train_df, feature_params)
    val_features, _ = make_features(val_df, feature_params)
    train_features, val_features = train_features.align(
        val_features, join="left", axis=1
    )

    assert train_features.columns.equals(val_features.columns)
