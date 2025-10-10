from nhl_ml.nhl_predict import rank_with_custom_weights


def main():
    # Bunch of jerks
    # rank_with_custom_weights(
    #     skater_columns=["goals", "assists", "penaltyminutes", "ppgoals", "pppoints", "shgoals", "gamewinninggoals", "shots"],
    #     skater_weights=[6, 4, 1, 2, 1, 3, 2, 0.5],
    #     goalie_columns=["wins", "goalsagainst", "saves", "shutouts"],
    #     goalie_weights=[5, -3, 0.5, 5],
    #     use_neural_net=True
    # )
    # Michael grier cup
    rank_with_custom_weights(
        skater_columns=["goals", "assists", "points", "plusminus", "penaltyminutes", "pppoints", "shpoints", "shots"],
        skater_weights=[1, 1, 1, 1, 1, 1, 1, 1],
        goalie_columns=["wins", "goalsagainstaverage", "savepct"],
        goalie_weights=[1, 1, 1],
        use_neural_net=True
    )


if __name__ == "__main__":
    main()
