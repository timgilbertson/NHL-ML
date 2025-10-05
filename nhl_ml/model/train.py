from typing import Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
    


def calc_new_season(
    train_targets: pd.DataFrame, train_features: pd.DataFrame, full_features: pd.DataFrame, output: pd.DataFrame, nn: bool = True
) -> Tuple[pd.DataFrame, Sequential]:
    if nn:
        return _predict_nn(train_targets, train_features, full_features)
    else:
        return _predict_xgb(train_targets, train_features, full_features)


def _predict_xgb(
    train_target: pd.DataFrame, train_features: pd.DataFrame, full_features: pd.DataFrame
) -> Tuple[pd.DataFrame, xgb.XGBRegressor]:
    train_features_numeric = train_features.apply(pd.to_numeric, errors='coerce').fillna(0)
    train_target_numeric = train_target.apply(pd.to_numeric, errors='coerce').fillna(0)
    full_features_numeric = full_features.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    model = xgb.XGBRegressor(
        tree_method="hist",
        grow_policy="lossguide",
        subsample=0.9,
        multi_strategy="multi_output_tree",
    )
    model.fit(train_features_numeric, train_target_numeric)
    final_pred = model.predict(full_features_numeric)

    return final_pred, model


def _predict_nn(
    train_target: pd.DataFrame, train_features: pd.DataFrame, full_features: pd.DataFrame
) -> Tuple[pd.DataFrame, Sequential]:
    player_position = full_features[["Player", "Position"]].copy()

    train_features = train_features[full_features.columns]

    train_features_numeric = train_features.drop(columns=["Player", "Position"]).apply(pd.to_numeric, errors='coerce').fillna(0)
    train_target_numeric = train_target.drop(columns=["Player", "Position"]).apply(pd.to_numeric, errors='coerce').fillna(0)
    full_features_numeric = full_features.drop(columns=["Player", "Position"]).apply(pd.to_numeric, errors='coerce').fillna(0)
    
    model = _build_neural_net(in_shape=train_features_numeric.shape[1], out_shape=train_target_numeric.shape[1])
    model.fit(train_features_numeric.values, train_target_numeric.values, epochs=100, verbose=1, validation_split=0.15)

    final_pred = model.predict(full_features_numeric.values)
    pred_df = pd.DataFrame(final_pred)

    pred_df.columns = [col for col in train_target.columns if col not in ["Player", "Position"]]

    return pred_df.assign(Player=player_position["Player"], Position=player_position["Position"]), model


def _build_neural_net(in_shape: int, out_shape: int) -> Sequential:
    """Creates the neural net model"""
    model = Sequential()
    model.add(Dense(units=64, activation="relu", input_dim=in_shape))
    for _ in range(5):
        model.add(Dropout(0.05))
        model.add(Dense(units=256, activation="relu"))
    model.add(Dense(units=out_shape, activation="linear"))

    model.compile(loss="mse", optimizer=Adam(learning_rate=0.000001))
    return model
