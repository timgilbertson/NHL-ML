import coloredlogs
import logging
from typing import Dict, Optional

from .io.inbound import load_all_data
from .model.train import calc_new_season
from .pre_processing.pre_processing import run_preprocessing
from .ranking.ranking import rank_players, print_rankings
from .validation.create_validation import transform_player_data
from .validation.run_validation import validate_model
from .config import PredictionConfig, create_custom_config

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def rank_next_season(
    skater_config: Optional[PredictionConfig] = None,
    goalie_config: Optional[PredictionConfig] = None,
    use_neural_net: bool = True
):
    """
    Rank players for next season using configurable prediction columns and weights.
    
    Args:
        skater_config: Configuration for skater predictions
        goalie_config: Configuration for goalie predictions
    """
    logger.info("Loading Player Data")
    player_data = load_all_data()

    logger.info("Pre-processing Player Data")
    pre_processed_player_data = run_preprocessing(player_data)

    logger.info("Splitting Validation Data")
    train_features, test_features, train_target, test_target, full_features = transform_player_data(pre_processed_player_data)

    logger.info("Predicting Next Season")
    prediction_df, trained_model = calc_new_season(train_target, train_features, full_features, full_features.copy(deep=True), nn=use_neural_net)

    logger.info("Validating Model")
    validate_model(trained_model, test_target, test_features[full_features.columns])

    logger.info("Ranking Players")
    ranked_df = rank_players(prediction_df, skater_config, goalie_config)
    
    logger.info("Printing Rankings")
    print_rankings(ranked_df, 100)
    
    return ranked_df


def rank_with_custom_weights(
    skater_columns: list[str],
    skater_weights: list[float],
    goalie_columns: list[str],
    goalie_weights: list[float],
    use_neural_net: bool = True
):
    """
    Convenience function to rank with custom column weights.
    
    Args:
        params: Dictionary containing data paths
        skater_columns: List of skater prediction columns
        skater_weights: List of weights for skater columns
        goalie_columns: List of goalie prediction columns
        goalie_weights: List of weights for goalie columns
        
    Returns:
        DataFrame with rankings
    """
    skater_config, goalie_config, _ = create_custom_config(
        skater_columns=skater_columns,
        skater_weights=skater_weights,
        goalie_columns=goalie_columns,
        goalie_weights=goalie_weights
    )
    
    return rank_next_season(skater_config, goalie_config, use_neural_net)
