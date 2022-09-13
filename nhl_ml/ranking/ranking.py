import pandas as pd


def rank_players(prediction_df: pd.DataFrame) -> pd.DataFrame:
    centres = prediction_df[prediction_df["Position"].str.contains("C")].reset_index()
    left = prediction_df[prediction_df["Position"].str.contains("L")].reset_index()
    right = prediction_df[prediction_df["Position"].str.contains("R")].reset_index()
    defence = prediction_df[prediction_df["Position"].str.contains("D")].reset_index()

    centres = centres.assign(rank=lambda x: x["goals"] + x["assists"] + x["PIM"] + x["shots"]).sort_values(
        "rank", ascending=False
    )
    left = left.assign(rank=lambda x: x["goals"] + x["assists"] + x["PIM"] + x["shots"]).sort_values(
        "rank", ascending=False
    )
    right = right.assign(rank=lambda x: x["goals"] + x["assists"] + x["PIM"] + x["shots"]).sort_values(
        "rank", ascending=False
    )
    defence = defence.assign(rank=lambda x: x["goals"] + x["assists"] + x["PIM"] + x["shots"]).sort_values(
        "rank", ascending=False
    )
    prediction_df = prediction_df.assign(rank=lambda x: x["goals"] + x["assists"] + x["PIM"] + x["shots"]).sort_values(
        "rank", ascending=False
    )

    import pdb; pdb.set_trace()