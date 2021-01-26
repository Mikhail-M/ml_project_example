import os
import pickle
from typing import Tuple

import pandas as pd
import pytest
from py._path.local import LocalPath
from sklearn.ensemble import RandomForestRegressor

from ml_example.enities import TrainingParams
from ml_example.enities.feature_params import FeatureParams
from ml_example.features.build_features import make_features
from ml_example.models.model_fit_predict import train_model, serialize_model


@pytest.fixture
def features_and_target(
    dataset: pd.DataFrame, fitted_transformer, feature_params: FeatureParams,
) -> Tuple[pd.DataFrame, pd.Series]:
    features, target = make_features(fitted_transformer, dataset, feature_params)
    return features, target


def test_train_model(features_and_target: Tuple[pd.DataFrame, pd.Series]):
    features, target = features_and_target
    model = train_model(features, target, train_params=TrainingParams())
    assert isinstance(model, RandomForestRegressor)
    assert model.predict(features).shape[0] == target.shape[0]


def test_serialize_model(tmpdir: LocalPath):
    expected_output = tmpdir.join("model.pkl")
    n_estimators = 10
    model = RandomForestRegressor(n_estimators=n_estimators)
    real_output = serialize_model(model, expected_output)
    assert real_output == expected_output
    assert os.path.exists
    with open(real_output, "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, RandomForestRegressor)
