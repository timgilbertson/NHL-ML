import pandas as pd
from typing import Dict, List, Optional
from ..config import PredictionConfig


def rank_players(
    prediction_df: pd.DataFrame, 
    skater_config: Optional[PredictionConfig] = None,
    goalie_config: Optional[PredictionConfig] = None
) -> pd.DataFrame:
    """
    Rank players using configurable columns and weights.
    
    Args:
        prediction_df: DataFrame with predictions
        skater_config: Configuration for skater ranking
        goalie_config: Configuration for goalie ranking
        
    Returns:
        DataFrame with rankings
    """
    from ..config import DEFAULT_SKATER_PREDICTION, DEFAULT_GOALIE_PREDICTION
    
    skater_config = skater_config or DEFAULT_SKATER_PREDICTION
    goalie_config = goalie_config or DEFAULT_GOALIE_PREDICTION
    
    skaters = prediction_df[~prediction_df["Position"].str.contains("G", na=False)].copy()
    goalies = prediction_df[prediction_df["Position"].str.contains("G", na=False)].copy()
    
    if not skaters.empty:
        skaters = _rank_skaters_by_position(skaters, skater_config)
    
    if not goalies.empty:
        goalies = _rank_goalies(goalies, goalie_config)
    
    if not skaters.empty and not goalies.empty:
        combined_df = pd.concat([skaters, goalies], ignore_index=True)
    elif not skaters.empty:
        combined_df = skaters
    elif not goalies.empty:
        combined_df = goalies
    else:
        return prediction_df
    
    combined_df = _add_overall_ranking(combined_df, skater_config, goalie_config)
    
    return combined_df


def _rank_skaters_by_position(skaters_df: pd.DataFrame, config: PredictionConfig) -> pd.DataFrame:
    """Rank skaters by position using the given configuration."""
    positions = {
        "Centers": skaters_df[skaters_df["Position"].str.contains("C", na=False)],
        "Left Wing": skaters_df[skaters_df["Position"].str.contains("L", na=False)],
        "Right Wing": skaters_df[skaters_df["Position"].str.contains("R", na=False)],
        "Defense": skaters_df[skaters_df["Position"].str.contains("D", na=False)]
    }
    
    ranked_positions = []
    for pos_name, pos_df in positions.items():
        if not pos_df.empty:
            pos_df = pos_df.copy()
            pos_df = _calculate_weighted_score(pos_df, config)
            pos_df = pos_df.sort_values("weighted_score", ascending=False).reset_index(drop=True)
            pos_df["position_rank"] = range(1, len(pos_df) + 1)
            pos_df["position_name"] = pos_name
            ranked_positions.append(pos_df)
    
    return pd.concat(ranked_positions, ignore_index=True) if ranked_positions else skaters_df


def _rank_goalies(goalies_df: pd.DataFrame, config: PredictionConfig) -> pd.DataFrame:
    """Rank goalies using the given configuration."""
    goalies_df = goalies_df.copy()
    goalies_df = _calculate_weighted_score(goalies_df, config)
    goalies_df = goalies_df.sort_values("weighted_score", ascending=False).reset_index(drop=True)
    goalies_df["position_rank"] = range(1, len(goalies_df) + 1)
    goalies_df["position_name"] = "Goalie"
    return goalies_df


def _calculate_weighted_score(df: pd.DataFrame, config: PredictionConfig) -> pd.DataFrame:
    """Calculate weighted score based on configuration."""
    df = df.copy()
    
    weighted_score = 0
    for column, weight in zip(config.columns, config.weights):
        if column in df.columns:
            weighted_score += df[column] * weight
        else:
            print(f"Warning: Column '{column}' not found in data")
    
    df["weighted_score"] = weighted_score
    return df


def _add_overall_ranking(df: pd.DataFrame, skater_config: PredictionConfig, goalie_config: PredictionConfig) -> pd.DataFrame:
    """Add overall ranking across all positions."""
    df = df.copy()
    
    skaters = df[~df["Position"].str.contains("G", na=False)]
    goalies = df[df["Position"].str.contains("G", na=False)]
    
    if not skaters.empty and not goalies.empty:

        skaters["normalized_score"] = (skaters["weighted_score"] - skaters["weighted_score"].min()) / (skaters["weighted_score"].max() - skaters["weighted_score"].min())
        goalies["normalized_score"] = (goalies["weighted_score"] - goalies["weighted_score"].min()) / (goalies["weighted_score"].max() - goalies["weighted_score"].min())
        

        combined = pd.concat([skaters, goalies], ignore_index=True)
        combined = combined.sort_values("normalized_score", ascending=False).reset_index(drop=True)
        combined["overall_rank"] = range(1, len(combined) + 1)
        
        return combined
    else:

        df = df.sort_values("weighted_score", ascending=False).reset_index(drop=True)
        df["overall_rank"] = range(1, len(df) + 1)
        return df


def print_rankings(df: pd.DataFrame, top_n: int = 100) -> None:
    """Print top N players from each position and overall."""
    print("=" * 80)
    print("NHL PLAYER RANKINGS")
    print("=" * 80)
    
    print(f"\nTOP {top_n} OVERALL:")
    print("-" * 40)
    overall_top = df.nsmallest(top_n, "overall_rank")[["Player", "Position", "weighted_score", "overall_rank"]]
    for _, row in overall_top.iterrows():
        print(f"{row['overall_rank']:2d}. {row['Player']:<25} {row['Position']:<3} Score: {row['weighted_score']:.2f}")
    
    for position in ["Centers", "Left Wing", "Right Wing", "Defense", "Goalie"]:
        pos_df = df[df["position_name"] == position]
        if not pos_df.empty:
            print(f"\nTOP {min(top_n, len(pos_df))} {position.upper()}:")
            print("-" * 40)
            pos_top = pos_df.nsmallest(top_n, "position_rank")[["Player", "weighted_score", "position_rank"]]
            for _, row in pos_top.iterrows():
                print(f"{row['position_rank']:2d}. {row['Player']:<25} Score: {row['weighted_score']:.2f}")
