import pandas as pd
from sklearn.model_selection import train_test_split

from ..io.inbound import THIS_YEAR, FIRST_YEAR


def transform_player_data(
    player_data: pd.DataFrame, 
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features, targets, full_features = _generate_features_and_targets(player_data)

    train_features, test_features, train_target, test_target = train_test_split(features, targets, test_size=0.15, random_state=97)

    return train_features, test_features, train_target, test_target, full_features


def _generate_features_and_targets(player_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    player_position = player_data[["Player", "Position"]].copy()

    full_features, _ = _generate_window(player_data, THIS_YEAR - 1)

    features = []
    targets = []
    for year_3 in range(FIRST_YEAR + 3, THIS_YEAR - 4):
        feature_data, target_data = _generate_window(player_data, year_3)
        features.append(feature_data.assign(Player=player_position["Player"], Position=player_position["Position"]))
        targets.append(target_data.assign(Player=player_position["Player"], Position=player_position["Position"]))
    
    return pd.concat(features), pd.concat(targets), full_features.assign(Player=player_position["Player"], Position=player_position["Position"])


def _generate_window(player_data: pd.DataFrame, year_3: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    year_1 = year_3 - 2
    year_2 = year_3 - 1
    target_year = year_3 + 1

    target_columns = _find_year_columns(player_data, target_year)
    out_target_columns = [col.replace(f"_{target_year}", "") for col in target_columns]

    year_1_columns = _find_year_columns(player_data, year_1)
    year_2_columns = _find_year_columns(player_data, year_2)
    year_3_columns = _find_year_columns(player_data, year_3)

    feature_columns = year_1_columns + year_2_columns + year_3_columns
    out_feature_columns = [col.replace(f"_{year_1}", "_1") for col in year_1_columns] + [col.replace(f"_{year_2}", "_2") for col in year_2_columns] + [col.replace(f"_{year_3}", "_3") for col in year_3_columns]

    feature_data = player_data[feature_columns]
    feature_data.columns = out_feature_columns

    target_data = player_data[target_columns]
    target_data.columns = out_target_columns

    return feature_data, target_data


def _find_year_columns(player_data: pd.DataFrame, year: int) -> list[str]:
    year_columns = []
    for col in player_data.columns:
        try:
            if col.endswith(f"_{year}"):
                year_columns.append(col)
        except:
            continue
    return year_columns