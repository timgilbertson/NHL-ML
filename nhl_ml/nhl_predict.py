import coloredlogs
import logging
from typing import Dict

from .io.inbound import load_csvs
from .model.train import calc_new_season
from .pre_processing.pre_processing import run_preprocessing
from .ranking.ranking import rank_players
from .validation.create_validation import transform_player_data
from .validation.run_validation import validate_model

logger = logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def rank_next_season(params: Dict[str, str]):
    logger.info("Loading Player Data")
    player_data = load_csvs(params)

    logger.info("Pre-processing Player Data")
    pre_processed_player_data = run_preprocessing(player_data)

    logger.info("Splitting Validation Data")
    train_features, test_features, train_target, test_target, full_features = transform_player_data(pre_processed_player_data)

    logger.info("Predicting Next Season")
    prediction_df, trained_model = calc_new_season(train_target, train_features, full_features, full_features.copy(deep=True))

    logger.info("Validating Model")
    validate_model(trained_model, test_target, test_features)

    logger.info("Ranking Players")
    rank_players(prediction_df)
