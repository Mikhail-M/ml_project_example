from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.impute._base import _BaseImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_example.enities.feature_params import FeatureParams


def get_imputer(strategy: str) -> _BaseImputer:
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    return imputer


def get_categorical_imputer() -> _BaseImputer:
    return get_imputer(strategy="most_frequent")


def get_numerical_imputer() -> _BaseImputer:
    return get_imputer(strategy="mean")


def process_categorical_features(
    pipeline: Pipeline, categorical_df: pd.DataFrame
) -> pd.DataFrame:
    one_df = pd.DataFrame(
        pipeline.transform(categorical_df).toarray(),
        columns=pipeline["encoder"].get_feature_names(),
    )
    return one_df


def categorical_pipeline() -> Pipeline:
    imputer = get_categorical_imputer()
    encoder = OneHotEncoder()
    pipeline = Pipeline([("imputer", imputer), ("encoder", encoder)])
    return pipeline


def numerical_pipeline() -> Pipeline:
    imputer = get_numerical_imputer()
    return Pipeline([("imputer", imputer)])


def make_features(
    transformer: ColumnTransformer,
    df: pd.DataFrame,
    params: FeatureParams,
    test_mode: bool = False,
) -> pd.DataFrame:
    ready_features_df = pd.DataFrame(transformer.transform(df).toarray())
    return ready_features_df


def column_transformer(params) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            ("categorical", categorical_pipeline(), params.categorical_features),
            ("numerical", numerical_pipeline(), params.numerical_features),
        ]
    )
    return transformer


def extract_target(
    df: pd.DataFrame, params: FeatureParams
) -> Tuple[pd.DataFrame, pd.Series]:
    features = df.drop(params.target_col, 1)
    target = df[params.target_col]
    if params.use_log_trick:
        target = pd.Series(np.log(target.to_numpy()))
    return features, target
