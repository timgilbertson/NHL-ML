# NHL ML

This NHL ML system now supports configurable prediction columns with weights and includes goalie predictions alongside skater predictions.

## Key Features

- **Configurable Prediction Columns**: Specify which columns to use for predictions and their weights
- **Goalie Support**: Full goalie data loading and prediction alongside skaters
- **Flexible Ranking**: Rank players by position and overall using weighted scores
- **Backward Compatibility**: Existing code continues to work

## Quick Start

### Basic Usage

```python
from nhl_ml.nhl_predict import rank_next_season

params = {
    "skater_data_path": "csvs/skaters",
    "goalie_data_path": "csvs/goalies",
}

# Use default configurations
rankings = rank_next_season(params)
```

### Custom Weights

```python
from nhl_ml.nhl_predict import rank_with_custom_weights

# Emphasize goals and assists for skaters
rankings = rank_with_custom_weights(
    params=params,
    skater_columns=["goals", "assists", "PIM", "shots", "toi", "blocks"],
    skater_weights=[2.0, 1.5, 0.1, 0.3, 0.2, 0.1],  # Higher weight for goals
    goalie_columns=["sv_percent", "gaa", "gsaa", "hdsv_percent"],
    goalie_weights=[1.0, -0.5, 1.0, 1.0]  # Negative weight for GAA
)
```

### Advanced Configuration

```python
from nhl_ml.config import create_custom_config, PredictionConfig
from nhl_ml.nhl_predict import rank_next_season

# Create custom configurations
skater_config, goalie_config, _ = create_custom_config(
    skater_columns=["goals", "assists", "shots"],
    skater_weights=[2.0, 1.5, 0.5],
    goalie_columns=["sv_percent", "gaa"],
    goalie_weights=[1.0, -0.5]
)

# Use custom configurations
rankings = rank_next_season(params, skater_config, goalie_config)
```

## Data Structure

### Skater Data
Expected columns in `csvs/skaters/*.csv`:
- Player, Position, Goals, Total Assists, PIM, Total Points, Shots, GP, First Assists, Second Assists, TOI, Shots Blocked

### Goalie Data
Expected columns in `csvs/goalies/*.csv`:
- Player, GP, TOI, Shots Against, Saves, Goals Against, SV%, GAA, GSAA, xG Against, HD Shots Against, HD Saves, HD Goals Against, HDSV%, HDGAA, HDGSAA

## Configuration Options

### Default Skater Prediction Columns
- `goals`, `assists`, `PIM`, `shots`, `toi`, `blocks`
- Default weights: `[1.0, 1.0, 0.1, 0.5, 0.3, 0.2]`

### Default Goalie Prediction Columns
- `sv_percent`, `gaa`, `gsaa`, `hdsv_percent`
- Default weights: `[1.0, -0.5, 1.0, 1.0]` (negative weight for GAA since lower is better)

### Customizing Weights

- **Positive weights**: Higher values are better (goals, assists, save percentage)
- **Negative weights**: Lower values are better (GAA, goals against)
- **Zero weights**: Column is ignored
- **Relative weights**: The ratio between weights determines their relative importance

## Ranking System

The system ranks players in multiple ways:

1. **By Position**: Centers, Left Wing, Right Wing, Defense, Goalies
2. **Overall**: Combined ranking across all positions
3. **Weighted Score**: Calculated using your specified columns and weights

## Testing

Run the test script to verify everything works:

```bash
python test_system.py
```

## Examples

### Example 1: Emphasize Offensive Stats
```python
rank_with_custom_weights(
    params=params,
    skater_columns=["goals", "assists", "shots", "toi"],
    skater_weights=[3.0, 2.0, 1.0, 0.5],  # Heavy emphasis on goals
    goalie_columns=["sv_percent", "gsaa"],
    goalie_weights=[1.0, 1.0]
)
```

### Example 2: Emphasize Defensive Stats
```python
rank_with_custom_weights(
    params=params,
    skater_columns=["goals", "assists", "blocks", "toi"],
    skater_weights=[1.0, 1.0, 2.0, 1.5],  # Heavy emphasis on blocks
    goalie_columns=["sv_percent", "gaa", "hdsv_percent"],
    goalie_weights=[1.0, -1.0, 1.5]  # Heavy emphasis on preventing goals
)
```

### Example 3: Balanced Approach
```python
rank_with_custom_weights(
    params=params,
    skater_columns=["goals", "assists", "PIM", "shots", "toi", "blocks"],
    skater_weights=[1.0, 1.0, 0.1, 0.5, 0.3, 0.2],  # Balanced weights
    goalie_columns=["sv_percent", "gaa", "gsaa", "hdsv_percent"],
    goalie_weights=[1.0, -0.5, 1.0, 1.0]  # Balanced goalie weights
)
```

## File Structure

```
nhl_ml/
├── config.py              # Configuration system
├── io/
│   └── inbound.py         # Data loading (skaters + goalies)
├── ranking/
│   └── ranking.py         # Configurable ranking system
├── nhl_predict.py         # Main prediction pipeline
└── ...

csvs/
├── skaters/               # Skater data files
│   ├── 08.csv
│   ├── 09.csv
│   └── ...
└── goalies/               # Goalie data files
    ├── 18.csv
    ├── 19.csv
    └── ...
```

## Migration from Old System

The old system continues to work without changes:

```python
# Old way (still works)
from nhl_ml.io.inbound import load_csvs
from nhl_ml.nhl_predict import rank_next_season

params = {"input_data": "csvs/skaters"}
rank_next_season(params)
```

The new system adds flexibility while maintaining backward compatibility.
