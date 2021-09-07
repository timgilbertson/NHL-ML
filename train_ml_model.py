import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import xgboost as xgb


def main():
    player_data = _load_csvs()
    targets, features, full_features = _transform_player_data(player_data)
    
    prediction_df = _calc_new_season(targets, features, full_features, full_features.copy(deep=True))

    _rank_players(prediction_df)

    import pdb; pdb.set_trace()


def _load_csvs(player_data=pd.DataFrame()) -> pd.DataFrame:
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
                player_data = player_data.merge(_player_data, on="Player", how="outer")
    player_data = player_data
    return player_data.merge(age_data, how="left", on="Player")


def _transform_player_data(player_data: pd.DataFrame) -> pd.DataFrame:
    target_list = ["goals_20", "assists_20", "PIM_20", "shots_20"]
    features_list = [
        "Player",
        "Position",
        "Age",
        "goals_17",
        "goals_18",
        "goals_19",
        "assists_17",
        "assists_18",
        "assists_19",
        "points_17",
        "points_18",
        "points_19",
        "shots_17",
        "shots_18",
        "shots_19",
        "gp_17",
        "gp_18",
        "gp_19",
        "PIM_17",
        "PIM_18",
        "PIM_19",
    ]

    full_features_list = [
        "Player",
        "Position",
        "Age",
        "goals_18",
        "goals_19",
        "goals_20",
        "assists_18",
        "assists_19",
        "assists_20",
        "points_18",
        "points_19",
        "points_20",
        "shots_18",
        "shots_19",
        "shots_20",
        "gp_18",
        "gp_19",
        "gp_20",
        "PIM_18",
        "PIM_19",
        "PIM_20",
    ]

    cleaned_df = player_data[features_list + target_list].dropna()
    target = cleaned_df[target_list]
    features = cleaned_df[features_list]
    full_features = player_data[full_features_list].dropna()
    return target, features, full_features


def _predict(target: pd.DataFrame, features: pd.DataFrame, full_features: pd.DataFrame):
    train_features, test_features, train_target, test_target = train_test_split(
        features.drop(columns=["Player", "Position"]), target, test_size=0.25, random_state=42
    )

    parameters, best_features = _gridsearch_xgb(train_features, train_target)
    model = xgb.XGBRegressor()
    # parameters = {'base_score': None, 'booster': 'dart', 'colsample_bylevel': 0.8, 'colsample_bynode': 0.5, 'colsample_bytree': 0.8, 'gamma': 1, 'learning_rate': 0.1, 'max_depth': None, 'min_child_weight': 0, 'n_estimators': 1000, 'num_parallel_tree': 5, 'objective': 'reg:squarederror', 'reg_alpha': 1.5, 'reg_lambda': 1.5, 'subsample': 0.8, 'tree_method': 'exact'}
    model.set_params(**parameters)
    model.fit(np.array(train_features), np.array(train_target).ravel())
    target_pred = model.predict(np.array(test_features))
    final_pred = model.predict(np.array(full_features))

    return r2_score(test_target, target_pred), final_pred



def _calc_new_season(targets: pd.DataFrame, features: pd.DataFrame, full_features: pd.DataFrame, output: pd.DataFrame) -> pd.DataFrame:

    for target in targets:
        target_frame = targets[target]
        r2, target_pred = _predict(target_frame, features, full_features.drop(columns=["Player", "Position"]))
        print(f"{target[:-3]} R2: {r2:.2f}")
        output[f"{target[:-3]}"] = target_pred

    return output


def _rank_players(prediction_df: pd.DataFrame) -> pd.DataFrame:
    centres = prediction_df[prediction_df["Position"].str.contains("C")].reset_index()
    left = prediction_df[prediction_df["Position"].str.contains("L")].reset_index()
    right = prediction_df[prediction_df["Position"].str.contains("R")].reset_index()
    defence = prediction_df[prediction_df["Position"].str.contains("D")].reset_index()

    for target in ["goals", "assists", "PIM", "shots"]:
        centres[f"{target}"] = centres[f"{target}"].rank(ascending=False)
        left[f"{target}"] = left[f"{target}"].rank(ascending=False)
        right[f"{target}"] = right[f"{target}"].rank(ascending=False)
        defence[f"{target}"] = defence[f"{target}"].rank(ascending=False)
        prediction_df[f"{target}"] = prediction_df[f"{target}"].rank(ascending=False)

    centres = centres.assign(rank=lambda x: x["goals"] + x["assists"] + x["PIM"] + x["shots"]).sort_values("rank", ascending=True)
    left = left.assign(rank=lambda x: x["goals"] + x["assists"] + x["PIM"] + x["shots"]).sort_values("rank", ascending=True)
    right = right.assign(rank=lambda x: x["goals"] + x["assists"] + x["PIM"] + x["shots"]).sort_values("rank", ascending=True)
    defence = defence.assign(rank=lambda x: x["goals"] + x["assists"] + x["PIM"] + x["shots"]).sort_values("rank", ascending=True)
    prediction_df = prediction_df.assign(rank=lambda x: x["goals"] + x["assists"] + x["PIM"] + x["shots"]).sort_values("rank", ascending=True)

    print(centres.head(20))
    print(left.head(20))
    print(right.head(20))
    print(defence.head(20))
    print(prediction_df.head(80))
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
