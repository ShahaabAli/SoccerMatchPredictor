import sqlite3
import pandas as pd
import kagglehub

def load_data():
    # Download dataset and connect to SQLite database
    path = kagglehub.dataset_download("hugomathien/soccer")
    conn = sqlite3.connect(f'{path}/database.sqlite')

    # Load tables into DataFrames
    team_attributes = pd.read_sql("SELECT * FROM Team_Attributes;", conn)
    matches = pd.read_sql("SELECT * FROM Match;", conn)
    teams = pd.read_sql("SELECT * FROM Team;", conn)

    # Add year column to prepare for merging and analysis
    team_attributes['date'] = pd.to_datetime(team_attributes['date'], errors='coerce')
    matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
    team_attributes['year'] = team_attributes['date'].dt.year
    matches['year'] = matches['date'].dt.year

    # Select and merge relevant team attributes
    features = [
        'buildUpPlaySpeed', 'buildUpPlaySpeedClass', 'buildUpPlayDribbling',
        'buildUpPlayDribblingClass', 'buildUpPlayPassing', 'buildUpPlayPassingClass',
        'buildUpPlayPositioningClass', 'chanceCreationPassing', 'chanceCreationPassingClass',
        'chanceCreationCrossing', 'chanceCreationCrossingClass', 'chanceCreationShooting',
        'chanceCreationShootingClass', 'chanceCreationPositioningClass', 'defencePressure',
        'defencePressureClass', 'defenceAggression', 'defenceAggressionClass',
        'defenceTeamWidth', 'defenceTeamWidthClass', 'defenceDefenderLineClass'
    ]
    team_attributes = team_attributes[['team_api_id', 'year'] + features]
    team_attributes = team_attributes.merge(
        teams[['team_api_id', 'team_long_name']], on='team_api_id', how='left'
    )

    # Calculate historical stats (win percentage, average goal difference)
    team_stats = {}

    def update_stats(row):
        for team, goals, opponent_goals in [
            (row['home_team_api_id'], row['home_team_goal'], row['away_team_goal']),
            (row['away_team_api_id'], row['away_team_goal'], row['home_team_goal'])
        ]:
            if team not in team_stats:
                team_stats[team] = {'wins': 0, 'matches': 0, 'goal_diff': 0}
            team_stats[team]['matches'] += 1
            team_stats[team]['goal_diff'] += goals - opponent_goals
            if goals > opponent_goals:
                team_stats[team]['wins'] += 1

    matches.apply(update_stats, axis=1)

    def add_features(row):
        home = team_stats[row['home_team_api_id']]
        away = team_stats[row['away_team_api_id']]
        row['home_win_percentage'] = home['wins'] / home['matches'] if home['matches'] > 0 else 0
        row['away_win_percentage'] = away['wins'] / away['matches'] if away['matches'] > 0 else 0
        row['home_avg_goal_diff'] = home['goal_diff'] / home['matches'] if home['matches'] > 0 else 0
        row['away_avg_goal_diff'] = away['goal_diff'] / away['matches'] if away['matches'] > 0 else 0
        return row

    matches = matches.apply(add_features, axis=1)

    # Merge team attributes for home and away teams
    home_attrs = team_attributes.rename(
        columns=lambda x: f'home_{x}' if x not in ['team_api_id', 'year'] else x
    )
    matches = matches.merge(
        home_attrs, how='left', left_on=['home_team_api_id', 'year'], right_on=['team_api_id', 'year']
    ).drop(columns=['team_api_id'])

    away_attrs = team_attributes.rename(
        columns=lambda x: f'away_{x}' if x not in ['team_api_id', 'year'] else x
    )
    matches = matches.merge(
        away_attrs, how='left', left_on=['away_team_api_id', 'year'], right_on=['team_api_id', 'year']
    ).drop(columns=['team_api_id'])

    # Create target column for match outcomes
    matches['result'] = matches.apply(
        lambda row: 2 if row['home_team_goal'] > row['away_team_goal'] else 
                    (1 if row['home_team_goal'] < row['away_team_goal'] else 0),
        axis=1
    )

    # Reorganize columns and clean up the final dataset
    ordered_columns = (
        [f'home_{col}' for col in features] +
        [f'away_{col}' for col in features] +
        ['home_win_percentage', 'away_win_percentage', 'home_avg_goal_diff', 'away_avg_goal_diff']
    )
    matches = matches[ordered_columns + ['result']].dropna()

    # One-hot encode categorical features
    categorical_cols = [col for col in matches.columns if col.endswith('Class')]
    matches = pd.get_dummies(matches, columns=categorical_cols, drop_first=True)

    # Additional Feature Engineering
    matches['stat_diff'] = matches['home_buildUpPlayPassing'] - matches['away_buildUpPlayPassing']
    matches['chance_creation_diff'] = matches['home_chanceCreationPassing'] - matches['away_chanceCreationPassing']
    matches['goal_difference'] = matches['home_avg_goal_diff'] - matches['away_avg_goal_diff']
    matches['win_percentage_diff'] = matches['home_win_percentage'] - matches['away_win_percentage']

    # Move target column to the end
    result_col = matches.pop('result')
    matches['result'] = result_col

    conn.close()
    matches.to_csv('matches.csv', index=False)
    return matches