from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
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


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:
    pipeline = categorical_pipeline(categorical_df)
    one_df = pd.DataFrame(
        pipeline.transform(categorical_df).toarray(),
        columns=pipeline["encoder"].get_feature_names(),
    )
    return one_df


def categorical_pipeline(categorical_df) -> Pipeline:
    imputer = get_categorical_imputer()
    encoder = OneHotEncoder()
    pipeline = Pipeline([("imputer", imputer), ("encoder", encoder)])
    pipeline.fit(categorical_df)
    return pipeline


def numerical_pipeline() -> Pipeline:
    imputer = get_numerical_imputer()
    return Pipeline([("imputer", imputer)])


def drop_features(
    df: pd.DataFrame, params: FeatureParams
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    num_features = params.numerical_features.copy()
    cat_features = params.categorical_features.copy()

    df = df.drop(params.features_to_drop, axis=1)
    for x in params.features_to_drop:
        if x in num_features:
            num_features.remove(x)
        if x in cat_features:
            cat_features.remove(x)
    return df, cat_features, num_features


def make_features(
    df: pd.DataFrame, params: FeatureParams, test_mode: bool = False
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    features = df[params.numerical_features + params.categorical_features]
    features, categorical_features, numerical_features = drop_features(features, params)

    categorical_features_df = features[categorical_features]
    categorical_features_transformed = process_categorical_features(
        categorical_features_df
    )

    numerical_features_df = process_numerical_features(features[numerical_features])

    ready_features_df = pd.concat(
        [categorical_features_transformed, numerical_features_df], axis=1
    )
    if test_mode:
        return ready_features_df, None
    else:
        target = df[params.target_col]
        if params.use_log_trick:
            target = pd.Series(np.log(target.to_numpy()))
        return ready_features_df, target
