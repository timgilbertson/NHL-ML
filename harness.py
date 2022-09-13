from nhl_ml.nhl_predict import rank_next_season


def main():
    params = {
        "input_data": "csvs",
    }

    rank_next_season(params)


if __name__ == "__main__":
    main()
