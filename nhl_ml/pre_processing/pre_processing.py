import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def run_preprocessing(player_data: pd.DataFrame) -> pd.DataFrame:
    de_duplicated_positions = player_data.groupby("Player", group_keys=False).apply(_fix_multiple_positions)
    return _scale_player_data(de_duplicated_positions).pipe(_encode_position)


def _fix_multiple_positions(player_data: pd.DataFrame) -> pd.DataFrame:
    if len(player_data) == 1:
        return player_data

    if all(player_data["Player"] == "Sebastian Aho"):
        player_data = player_data[player_data["Position"] != "D"]

    player = player_data["Player"].iloc[0]
    return player_data.agg("sum").to_frame().T.assign(Player=player)


def _scale_player_data(player_data: pd.DataFrame) -> pd.DataFrame:
    float_columns = player_data.drop(columns=["Player", "Position"])
    string_columns = player_data[["Player", "Position"]]
    scaler = MinMaxScaler()
    scaled_player_data = scaler.fit_transform(float_columns)
    return pd.DataFrame(
        np.hstack([string_columns, scaled_player_data]),
        columns=list(string_columns.columns) + list(float_columns.columns),
    )


def _encode_position(player_data: pd.DataFrame) -> pd.DataFrame:
    replacement_dict = {
        'CC, L': "CL",
        'RC, RC': "CR",
        'LC': "CL",
        'C, LC': "CL",
        'RC, R': "CR",
        'C, LCL': "CL",
        'CC, R': "CR",
        'LL, R': "RL",
        'C, LL': "CL",
        'RL, R': "RL",
        'RC': "CR",
        'C, RR': "CR",
        'C, RRC': "CR",
        'CLC, L': "CL",
        'LCC, L': "CL",
        'C, RC': "CR",
        'L, RRL': "RL",
        'LR': "RL",
        'C, LLC': "CL",
        'LCR': "CLR",
        'CRC, R': "CR",
        'DRD, R': "DR",
        'L, RLR': "RL",
        'RD': "DR",
        'L, RR': "RL",
        'CC, RL, RC, LL': "CLR",
        'C, RCRL': "CLR",
        'LC, LC, RC': "CLR",
        'L, RL': "RL",
        'LL, RR': "RL",
        'C, LC, RCL': "CLR",
        "LC, LC": "CL",
        "LC, L": "CL",
        "CC, LL": "CL",
        "CC, RR": "CR",
    }
    player_data_cleaned = player_data["Position"].replace(replacement_dict)
    
    encoder = OneHotEncoder()

    encoded_positions = encoder.fit_transform(player_data_cleaned.values.reshape(-1, 1)).todense()

    return player_data.join(pd.DataFrame(encoded_positions, columns=["position"] * encoded_positions.shape[1]))
