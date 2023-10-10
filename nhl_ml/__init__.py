import argparse

from .nhl_predict import rank_next_season


def main():
    parser = argparse.ArgumentParser(description="NHL Fantasy ML Predictor")
    parser.add_argument("--input-data", help="Raw player data", required=True)

    args = parser.parse_args()
    params = {arg: getattr(args, arg) for arg in vars(args)}
    rank_next_season(params)