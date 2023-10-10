from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split


def transform_player_data(player_data: pd.DataFrame) -> pd.DataFrame:
    features = []
    targets = []
    int_columns = [
        "goals_3",
        "assists_3",
        "PIM_3",
        "shots_3",
        "first_assists_3",
        "second_assists_3",
        "toi_3",
        "blocks_3",
        "goals_2",
        "assists_2",
        "PIM_2",
        "shots_2",
        "first_assists_2",
        "second_assists_2",
        "toi_2",
        "blocks_2",
        "goals_1",
        "assists_1",
        "PIM_1",
        "shots_1",
        "first_assists_1",
        "second_assists_1",
        "toi_1",
        "blocks_1",
    ]
    str_columns = [
        "Player",
        "Position",
    ]

    feature_columns = str_columns + int_columns

    target_columns = ["goals_0", "assists_0", "PIM_0", "shots_0", "toi_0", "blocks_0"]
    for year in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
        features.append(pd.DataFrame(player_data[_create_feature_list(year)].values, columns=feature_columns))
        targets.append(pd.DataFrame(player_data[_create_target_list(year)].values, columns=target_columns))

    full_features = pd.DataFrame(player_data[_create_feature_list(24)].fillna(0).values, columns=feature_columns)

    features = pd.concat(features, ignore_index=True)
    targets = pd.concat(targets, ignore_index=True)

    joined = features.join(targets).fillna(0)

    both = joined[joined[int_columns].sum(axis=1) > 0]

    features = both[feature_columns]
    targets = both[target_columns]

    train_features, test_features, train_target, test_target = train_test_split(
        features.drop(columns=["Player", "Position"]), targets, test_size=0.15, random_state=97
    )

    return train_features, test_features, train_target, test_target, full_features


def _create_target_list(year: int) -> List[str]:
    return [f"goals_{year}", f"assists_{year}", f"PIM_{year}", f"shots_{year}", f"toi_{year}", f"blocks_{year}"]


def _create_feature_list(target_year: int) -> List[str]:
    feature_list = ["Player", "Position"]
    for year in range(target_year - 3, target_year):
        feature_list.extend([
            f"goals_{year}",
            f"assists_{year}",
            f"PIM_{year}",
            f"shots_{year}",
            f"first_assists_{year}",
            f"second_assists_{year}",
            f"toi_{year}",
            f"blocks_{year}",
        ])

    return feature_list
