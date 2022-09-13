from typing import Tuple

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


def calc_new_season(
    train_targets: pd.DataFrame, train_features: pd.DataFrame, full_features: pd.DataFrame, output: pd.DataFrame
) -> Tuple[pd.DataFrame, Sequential]:
    target_pred, trained_model = _predict(train_targets, train_features, full_features.drop(columns=["Player", "Position"]))
    
    return output.join(pd.DataFrame(target_pred, columns=train_targets.columns.str[:-2])), trained_model


def _predict(train_target: pd.DataFrame, train_features: pd.DataFrame, full_features: pd.DataFrame) -> Tuple[pd.DataFrame, Sequential]:
    model = _build_neural_net(in_shape=train_features.shape[1], out_shape=train_target.shape[1])
    model.fit(train_features.values, train_target.values, epochs=100, verbose=1, validation_split=0.15)
    final_pred = model.predict(full_features.astype(float).values)

    return final_pred, model


def _build_neural_net(in_shape: int, out_shape: int) -> Sequential:
    """Creates the neural net model"""
    model = Sequential()
    model.add(Dense(units=64, activation="relu", input_dim=in_shape))
    for _ in range(5):
        model.add(Dropout(0.05))
        model.add(Dense(units=512, activation="relu"))
    model.add(Dense(units=out_shape, activation="linear"))

    model.compile(loss="mse", optimizer=Adam(lr=0.000001))
    return model
