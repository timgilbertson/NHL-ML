from typing import List
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import xgboost as xgb

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def main():
    player_data = _load_csvs()
    targets, features, full_features = _transform_player_data(player_data)
    
    prediction_df = _calc_new_season(targets, features, full_features, full_features.copy(deep=True))

    _rank_players(prediction_df)

    import pdb; pdb.set_trace()


def _load_csvs(player_data=pd.DataFrame()) -> pd.DataFrame:
    """Loads in CSV stats downloaded from https://www.naturalstattrick.com/playerteams.php

    Args:
        player_data ([type], optional): stupid way to initialize a frame. Defaults to pd.DataFrame().

    Returns:
        pd.DataFrame: DataFrame of all the data
    """
    path = r"csvs"
    all_files = glob.glob(path + "/*.csv")
    for file_name in all_files:
        if "age" in file_name:
            age_data = (
                pd.read_csv(file_name)
                .assign(Player=lambda x: x["Player"].str.split("\\").str[0])
                .drop_duplicates("Player")
            )
        else:
            if player_data.empty:
                player_data = pd.read_csv(file_name)[
                    [
                        "Player",
                        "Goals",
                        "Total Assists",
                        "PIM",
                        "Total Points",
                        "Shots",
                        "GP",
                        "Position",
                    ]
                ].rename(
                    columns={
                        "Goals": f"goals_{int(file_name[5:-4])}",
                        "Total Assists": f"assists_{int(file_name[5:-4])}",
                        "PIM": f"PIM_{int(file_name[5:-4])}",
                        "Total Points": f"points_{int(file_name[5:-4])}",
                        "Shots": f"shots_{int(file_name[5:-4])}",
                        "GP": f"gp_{int(file_name[5:-4])}",
                    }
                )
            else:
                _player_data = pd.read_csv(file_name)[
                    [
                        "Player",
                        "Goals",
                        "Total Assists",
                        "PIM",
                        "Total Points",
                        "Shots",
                        "GP",
                        "Position",
                    ]
                ].rename(
                    columns={
                        "Goals": f"goals_{int(file_name[5:-4])}",
                        "Total Assists": f"assists_{int(file_name[5:-4])}",
                        "PIM": f"PIM_{int(file_name[5:-4])}",
                        "Total Points": f"points_{int(file_name[5:-4])}",
                        "Shots": f"shots_{int(file_name[5:-4])}",
                        "GP": f"gp_{int(file_name[5:-4])}",
                    }
                )
                player_data = player_data.merge(_player_data, on=["Player", "Position"], how="outer")
    player_data = player_data.groupby("Player", group_keys=False).apply(_fix_multiple_positions)
    return _scale_player_data(player_data)


def _fix_multiple_positions(player_data: pd.DataFrame) -> pd.DataFrame:
    if len(player_data) == 1:
        return player_data
    player = player_data["Player"].iloc[0]
    return player_data.agg(sum).to_frame().T.assign(Player=player)


def _scale_player_data(player_data: pd.DataFrame) -> pd.DataFrame:
    float_columns = player_data.drop(columns=["Player", "Position"])
    string_columns = player_data[["Player", "Position"]]
    scaler = MinMaxScaler()
    scaled_player_data = scaler.fit_transform(float_columns)
    return pd.DataFrame(np.hstack([string_columns, scaled_player_data]), columns=list(string_columns.columns) + list(float_columns.columns))


def _transform_player_data(player_data: pd.DataFrame) -> pd.DataFrame:
    features = []
    targets = []
    feature_columns = ["Player", "Position", "goals_3", "assists_3", "PIM_3", "shots_3", "goals_2", "assists_2", "PIM_2", "shots_2", "goals_1", "assists_1", "PIM_1", "shots_1"]
    target_columns = ["goals_0", "assists_0", "PIM_0", "shots_0"]
    for year in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
        features.append(pd.DataFrame(player_data[_create_feature_list(year)].values, columns=feature_columns))
        targets.append(pd.DataFrame(player_data[_create_target_list(year)].values, columns=target_columns))
    full_features = pd.DataFrame(player_data[_create_feature_list(22)].fillna(0).values, columns=feature_columns)
    features = pd.concat(features, ignore_index=True)
    targets = pd.concat(targets, ignore_index=True)
    both = features.join(targets).fillna(0)
    features = both[feature_columns]
    targets = both[target_columns]
    return targets, features, full_features


def _create_target_list(year: int) -> List[str]:
    return [f"goals_{year}", f"assists_{year}", f"PIM_{year}", f"shots_{year}"]


def _create_feature_list(target_year: int) -> List[str]:
    feature_list = ["Player", "Position"]
    for year in range(target_year - 3, target_year):
        feature_list.extend([f"goals_{year}", f"assists_{year}", f"PIM_{year}", f"shots_{year}"])
    return feature_list


def _predict(target: pd.DataFrame, features: pd.DataFrame, full_features: pd.DataFrame):
    train_features, test_features, train_target, test_target = train_test_split(
        features.drop(columns=["Player", "Position"]), target, test_size=0.25, random_state=42
    )

    model = _build_neural_net(in_shape=train_features.shape[1], out_shape=1)
    model.fit(train_features.values, train_target.values, epochs=15, verbose=0)
    target_pred = model.predict(test_features.values)
    final_pred = model.predict(full_features.values)

    return r2_score(test_target, target_pred), final_pred


def _build_neural_net(in_shape: int, out_shape: int):
    """
    Creates the neural net model
    """
    model = Sequential()
    model.add(Dense(units=150, activation="relu", input_dim=in_shape))
    for _ in range(15):
        model.add(Dense(units=150, activation="relu"))
    model.add(Dense(units=out_shape, activation="linear"))

    model.compile(loss="mse", optimizer=Adam(lr=0.00001))
    return model


def _calc_new_season(targets: pd.DataFrame, features: pd.DataFrame, full_features: pd.DataFrame, output: pd.DataFrame) -> pd.DataFrame:
    for target in targets:
        target_frame = targets[target]
        r2, target_pred = _predict(target_frame, features, full_features.drop(columns=["Player", "Position"]))
        print(f"{target[:-2]} R2: {r2:.2f}")
        output[f"{target[:-2]}"] = target_pred
    return output
    # r2, target_pred = _predict(targets, features, full_features.drop(columns=["Player", "Position"]))
    # print(r2)
    # return output.join(pd.DataFrame(target_pred, columns=targets.columns.str[:-3]))


def _rank_players(prediction_df: pd.DataFrame) -> pd.DataFrame:
    centres = prediction_df[prediction_df["Position"].str.contains("C")].reset_index()
    left = prediction_df[prediction_df["Position"].str.contains("L")].reset_index()
    right = prediction_df[prediction_df["Position"].str.contains("R")].reset_index()
    defence = prediction_df[prediction_df["Position"].str.contains("D")].reset_index()

    centres = centres.assign(rank=lambda x: x["goals"] + x["assists"] + x["PIM"] + x["shots"]).sort_values("rank", ascending=False)
    left = left.assign(rank=lambda x: x["goals"] + x["assists"] + x["PIM"] + x["shots"]).sort_values("rank", ascending=False)
    right = right.assign(rank=lambda x: x["goals"] + x["assists"] + x["PIM"] + x["shots"]).sort_values("rank", ascending=False)
    defence = defence.assign(rank=lambda x: x["goals"] + x["assists"] + x["PIM"] + x["shots"]).sort_values("rank", ascending=False)
    prediction_df = prediction_df.assign(rank=lambda x: x["goals"] + x["assists"] + x["PIM"] + x["shots"]).sort_values("rank", ascending=False)

    print(centres.head(20))
    print(left.head(20))
    print(right.head(20))
    print(defence.head(20))
    print(prediction_df.head(20))
    import pdb; pdb.set_trace()


def _gridsearch_xgb(features: pd.DataFrame, targets: pd.DataFrame):
    model = xgb.XGBRegressor()
    parameters = {
        "objective": ["reg:squarederror"],
        "max_depth": [None],
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [1000],
        "booster": ["gbtree", "dart"],
        "tree_method": ["exact"],
        "gamma": [0.8, 1.1],
        "min_child_weight": [0],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.8, 0.9],
        "colsample_bylevel": [0.7, 0.8],
        "colsample_bynode": [0.4, 0.5],
        "reg_alpha": [1.2, 1.5],
        "reg_lambda": [1.5, 1.8],
        "base_score": [None],
        "num_parallel_tree": [5],
    }
    xgb_grid = GridSearchCV(model, parameters, cv=None, n_jobs=5, verbose=True)

    xgb_grid.fit(features, targets)
    importances = xgb_grid.best_estimator_.feature_importances_
    feature_importances = pd.DataFrame(data={'feature': features.columns, 'importance': importances})

    print(xgb_grid.best_params_)
    print(feature_importances.sort_values("importance", ascending=False))
    return xgb_grid.best_params_, feature_importances[7:]

if __name__ == "__main__":
    main()
