"""
Configuration system for NHL ML predictions.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PredictionConfig:
    """Configuration for prediction columns and their weights."""
    columns: List[str]
    weights: List[float]
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if len(self.columns) != len(self.weights):
            raise ValueError("Number of columns must match number of weights")
        if all(w == 0 for w in self.weights):
            raise ValueError("At least one weight must be non-zero")
        # Allow negative weights (useful for metrics where lower is better, like GAA)


@dataclass
class DataConfig:
    """Configuration for data sources and processing."""
    season: str = "20242025"  # Default to current season
    game_type: str = "2"  # Regular season
    skater_columns: List[str] = None
    goalie_columns: List[str] = None
    position_column: str = "Position"
    player_column: str = "Player"
    
    def __post_init__(self):
        """Set default columns if not provided."""
        if self.skater_columns is None:
            self.skater_columns = DEFAULT_SKATER_COLUMNS
        if self.goalie_columns is None:
            self.goalie_columns = DEFAULT_GOALIE_COLUMNS


# Default configurations
DEFAULT_SKATER_COLUMNS = [
    "Goals", "Total Assists", "PIM", "Total Points", 
    "Shots", "GP", "First Assists", "Second Assists", 
    "TOI", "Shots Blocked"
]

DEFAULT_GOALIE_COLUMNS = [
    "GP", "TOI", "Shots Against", "Saves", "Goals Against", 
    "SV%", "GAA", "GSAA", "xG Against", "HD Shots Against", 
    "HD Saves", "HD Goals Against", "HDSV%", "HDGAA", "HDGSAA"
]

# Default prediction configurations
DEFAULT_SKATER_PREDICTION = PredictionConfig(
    columns=["goals", "assists", "PIM", "shots", "toi", "blocks"],
    weights=[1.0, 1.0, 0.1, 0.5, 0.3, 0.2]
)

DEFAULT_GOALIE_PREDICTION = PredictionConfig(
    columns=["sv_percent", "gaa", "gsaa", "hdsv_percent"],
    weights=[1.0, -0.5, 1.0, 1.0]  # Negative weight for GAA (lower is better)
)

DEFAULT_DATA_CONFIG = DataConfig(
    season="20242025",
    game_type="2",
    skater_columns=DEFAULT_SKATER_COLUMNS,
    goalie_columns=DEFAULT_GOALIE_COLUMNS
)


def create_custom_config(
    skater_columns: Optional[List[str]] = None,
    skater_weights: Optional[List[float]] = None,
    goalie_columns: Optional[List[str]] = None,
    goalie_weights: Optional[List[float]] = None,
    season: str = "20242025",
    game_type: str = "2"
) -> tuple[PredictionConfig, PredictionConfig, DataConfig]:
    """
    Create custom prediction and data configurations.
    
    Args:
        skater_columns: List of skater prediction columns
        skater_weights: List of weights for skater columns
        goalie_columns: List of goalie prediction columns  
        goalie_weights: List of weights for goalie columns
        season: NHL season in format like "20242025"
        game_type: Game type ("2" for regular season)
        
    Returns:
        Tuple of (skater_config, goalie_config, data_config)
    """
    # Use defaults if not provided
    skater_cols = skater_columns or DEFAULT_SKATER_PREDICTION.columns
    skater_wts = skater_weights or DEFAULT_SKATER_PREDICTION.weights
    goalie_cols = goalie_columns or DEFAULT_GOALIE_PREDICTION.columns
    goalie_wts = goalie_weights or DEFAULT_GOALIE_PREDICTION.weights
    
    skater_config = PredictionConfig(columns=skater_cols, weights=skater_wts)
    goalie_config = PredictionConfig(columns=goalie_cols, weights=goalie_wts)
    data_config = DataConfig(
        season=season,
        game_type=game_type,
        skater_columns=DEFAULT_SKATER_COLUMNS,
        goalie_columns=DEFAULT_GOALIE_COLUMNS
    )
    
    return skater_config, goalie_config, data_config
