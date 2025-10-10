from datetime import datetime

import pandas as pd
from nhlpy import NHLClient
from nhlpy.api.query.builder import QueryBuilder, QueryContext
from nhlpy.api.query.filters.season import SeasonQuery
from nhlpy.api.query.filters.game_type import GameTypeQuery
from nhlpy.api.query.filters.position import PositionQuery, PositionTypes
from tqdm import tqdm

# Initialize NHL client
nhl_client = NHLClient()
THIS_YEAR = datetime.now().year
FIRST_YEAR = 1960

# Column mappings from NHL API to our expected format
SKATER_COLUMN_MAPPING = {
    "skaterFullName": "Player",
    "positionCode": "Position", 
    "goals": "goals",
    "assists": "assists",
    "penaltyMinutes": "penaltyMinutes",
    "points": "points",
    "shots": "shots",
    "gamesPlayed": "gamesPlayed",
    "evGoals": "evGoals",
    "evPoints": "evPoints",
    "faceoffWinPct": "faceoffWinPct",
    "gameWinningGoals": "gameWinningGoals",
    "otGoals": "otGoals",
    "plusMinus": "plusMinus",
    "pointsPerGame": "pointsPerGame",
    "ppGoals": "ppGoals",
    "ppPoints": "ppPoints",
    "shGoals": "shGoals",
    "shPoints": "shPoints",
    "shootingPct": "shootingPct",
    "timeOnIcePerGame": "timeOnIcePerGame"
}

GOALIE_COLUMN_MAPPING = {
    "goalieFullName": "Player",
    "gamesPlayed": "gamesPlayed",
    "timeOnIce": "timeOnIce",
    "shotsAgainst": "shotsAgainst",
    "saves": "saves",
    "goalsAgainst": "goalsAgainst",
    "savePct": "savePct",
    "goalsAgainstAverage": "goalsAgainstAverage",
    "assists": "goalieAssists",  # Prefix to avoid conflict with skater assists
    "gamesStarted": "gamesStarted",
    "goals": "goalieGoals",      # Prefix to avoid conflict with skater goals
    "losses": "losses",
    "otLosses": "otLosses",
    "penaltyMinutes": "goaliePenaltyMinutes",  # Prefix to avoid conflict
    "points": "goaliePoints",    # Prefix to avoid conflict with skater points
    "shutouts": "shutouts",
    "ties": "ties",
    "wins": "wins"
}


def load_skater_data() -> pd.DataFrame:
    """Load skater data from the NHL Edge API.

    Returns:
        pd.DataFrame: DataFrame of all the skater data
    """
    combined_data = []
    for start_season in tqdm(range(FIRST_YEAR, THIS_YEAR)):
        season = f"{start_season}{start_season + 1}"

        if season == "20042005":
            continue
        
        all_skater_data = []
        
        forwards_filters = [
            GameTypeQuery(game_type="2"),
            SeasonQuery(season_start=season, season_end=season),
            PositionQuery(position=PositionTypes.ALL_FORWARDS)
        ]
        
        forwards_query_context = QueryBuilder().build(filters=forwards_filters)
        forwards_data = nhl_client.stats.skater_stats_with_query_context(
            report_type='summary',
            query_context=forwards_query_context,
            aggregate=True,
            limit=1000,
        )
        
        if forwards_data and 'data' in forwards_data:
            forwards_df = pd.DataFrame(forwards_data['data'])
            all_skater_data.append(forwards_df)
        
        defense_filters = [
            GameTypeQuery(game_type="2"),
            SeasonQuery(season_start=season, season_end=season),
            PositionQuery(position=PositionTypes.DEFENSE)
        ]
        
        defense_query_context = QueryBuilder().build(filters=defense_filters)
        defense_data = nhl_client.stats.skater_stats_with_query_context(
            report_type='summary',
            query_context=defense_query_context,
            aggregate=True,
            limit=1000,
        )
        
        if defense_data and 'data' in defense_data:
            defense_df = pd.DataFrame(defense_data['data'])
            all_skater_data.append(defense_df.assign(Position="D"))
        
        if not all_skater_data:
            print(f"No skater data returned from NHL API for season {season}")
            continue
            
        skater_df = pd.concat(all_skater_data, ignore_index=True)
        
        available_columns = [col for col in SKATER_COLUMN_MAPPING.keys() if col in skater_df.columns]
        mapped_df = skater_df[available_columns].rename(columns=SKATER_COLUMN_MAPPING)
        
        year_suffix = start_season
        season_column_mapping = {}
        for col in mapped_df.columns:
            if col not in ['Player', 'Position']:
                season_column_mapping[col] = f"{col.lower()}_{year_suffix}"

        mapped_df = mapped_df.rename(columns=season_column_mapping).drop_duplicates("Player", keep="first").set_index(["Player", "Position"])

        combined_data.append(mapped_df)
    
    combined_data = pd.concat(combined_data, axis=1)

    return combined_data.replace(" ", "_")


def load_goalie_data() -> pd.DataFrame:
    """Load goalie data from the NHL Edge API.

    Returns:
        pd.DataFrame: DataFrame of all the goalie data
    """
    goalie_data = []
    for season in tqdm(range(FIRST_YEAR, THIS_YEAR)):
        start_season = f"{season}{season + 1}"
        end_season = f"{season + 1}{season + 2}"
        
        data = nhl_client.stats.goalie_stats_summary(
            start_season=start_season,
            end_season=end_season,
            limit=1000,
        )
            
        goalie_df = pd.DataFrame(data)
        
        available_columns = [col for col in GOALIE_COLUMN_MAPPING.keys() if col in goalie_df.columns]
        mapped_df = goalie_df[available_columns].rename(columns=GOALIE_COLUMN_MAPPING)
        
        mapped_df["Position"] = "G"
        
        season_column_mapping = {}
        for col in mapped_df.columns:
            if col not in ['Player', 'Position']:
                season_column_mapping[col] = f"{col.lower()}_{season}"
        
        mapped_df = mapped_df.rename(columns=season_column_mapping).drop_duplicates("Player", keep="first").set_index(["Player", "Position"])

        goalie_data.append(mapped_df)

    combined_data = pd.concat(goalie_data, axis=1)

    return combined_data.replace(" ", "_")


def load_all_data() -> pd.DataFrame:
    """Load both skater and goalie data and combine them.

    Args:
        params: dictionary containing season and other parameters

    Returns:
        pd.DataFrame: Combined DataFrame of skater and goalie data
    """
    skater_data = load_skater_data().reset_index()
    goalie_data = load_goalie_data().reset_index()

    return pd.merge(skater_data, goalie_data, on=["Player", "Position"], how="outer", suffixes=("_s", "_g"))
