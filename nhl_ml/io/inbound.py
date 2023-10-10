import glob
from typing import Dict

import pandas as pd

IN_COLUMNS = [
    "Player",
    "Goals",
    "Total Assists",
    "PIM",
    "Total Points",
    "Shots",
    "GP",
    "Position",
    "First Assists",
    "Second Assists",
    "TOI",
    "Shots Blocked"
]


def load_csvs(params: Dict[str, str], player_data=pd.DataFrame()) -> pd.DataFrame:
    """Loads in CSV stats downloaded from https://www.naturalstattrick.com/playerteams.php

    Args:
        player_data ([type], optional): a stupid way to initialize an empty frame. Defaults to pd.DataFrame().

    Returns:
        pd.DataFrame: DataFrame of all the player data
    """
    path = params["input_data"]
    all_files = glob.glob(path + "/*.csv")

    for file_name in all_files:
        if player_data.empty:
            player_data = pd.read_csv(file_name)[IN_COLUMNS].rename(columns=_column_map(file_name))
        else:
            _player_data = pd.read_csv(file_name)[IN_COLUMNS].rename(columns=_column_map(file_name))
            player_data = player_data.merge(_player_data, on=["Player", "Position"], how="outer")
    return player_data


def _column_map(file_name):
    return {
        "Goals": f"goals_{int(file_name[5:-4])}",
        "Total Assists": f"assists_{int(file_name[5:-4])}",
        "PIM": f"PIM_{int(file_name[5:-4])}",
        "Total Points": f"points_{int(file_name[5:-4])}",
        "Shots": f"shots_{int(file_name[5:-4])}",
        "GP": f"gp_{int(file_name[5:-4])}",
        "First Assists": f"first_assists_{int(file_name[5:-4])}",
        "Second Assists": f"second_assists_{int(file_name[5:-4])}",
        "TOI": f"toi_{int(file_name[5:-4])}",
        "Shots Blocked": f"blocks_{int(file_name[5:-4])}",
    }
