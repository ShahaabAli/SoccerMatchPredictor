import sqlite3
import pandas as pd
import kagglehub

def load_data():
    # Download dataset
    path = kagglehub.dataset_download("hugomathien/soccer")

    # Connect to the SQLite database
    db_path = f'{path}/database.sqlite'
    conn = sqlite3.connect(db_path)

    # Query data from the database
    team_attributes_df = pd.read_sql("SELECT * FROM Team_Attributes;", conn)
    matches_df = pd.read_sql("SELECT * FROM Match;", conn)
    team_df = pd.read_sql("SELECT * FROM Team;", conn)

    # Convert `date` column to datetime in both DataFrames
    team_attributes_df['date'] = pd.to_datetime(team_attributes_df['date'], errors='coerce')
    matches_df['date'] = pd.to_datetime(matches_df['date'], errors='coerce')

    # Add year column to `Team_Attributes` and `Match` DataFrames
    team_attributes_df['year'] = team_attributes_df['date'].dt.year
    matches_df['year'] = matches_df['date'].dt.year

    # Keep desired features and retain `team_api_id` and `year` for merging
    selected_features = [
        'buildUpPlaySpeed', 'buildUpPlaySpeedClass', 'buildUpPlayDribbling',
        'buildUpPlayDribblingClass', 'buildUpPlayPassing', 'buildUpPlayPassingClass',
        'buildUpPlayPositioningClass', 'chanceCreationPassing', 'chanceCreationPassingClass',
        'chanceCreationCrossing', 'chanceCreationCrossingClass', 'chanceCreationShooting',
        'chanceCreationShootingClass', 'chanceCreationPositioningClass', 'defencePressure',
        'defencePressureClass', 'defenceAggression', 'defenceAggressionClass',
        'defenceTeamWidth', 'defenceTeamWidthClass', 'defenceDefenderLineClass'
    ]
    team_attributes_df = team_attributes_df[['team_api_id', 'year'] + selected_features]

    # Add `team_long_name` from `team_df`
    team_attributes_df = team_attributes_df.merge(
        team_df[['team_api_id', 'team_long_name']],
        on='team_api_id',
        how='left'
    )

    # Merge team attributes for home team
    home_attributes = team_attributes_df.rename(
        columns=lambda x: f'home_{x}' if x not in ['team_api_id', 'year'] else x
    )
    matches_df = matches_df.merge(
        home_attributes,
        how='left',
        left_on=['home_team_api_id', 'year'],
        right_on=['team_api_id', 'year']
    ).drop(columns=['team_api_id'])

    # Merge team attributes for away team
    away_attributes = team_attributes_df.rename(
        columns=lambda x: f'away_{x}' if x not in ['team_api_id', 'year'] else x
    )
    matches_df = matches_df.merge(
        away_attributes,
        how='left',
        left_on=['away_team_api_id', 'year'],
        right_on=['team_api_id', 'year']
    ).drop(columns=['team_api_id'])

    # Add a 'result' column to `matches_df` based on the match outcome
    def calculate_result(row):
        if row['home_team_goal'] > row['away_team_goal']:
            return 2  # Home win
        elif row['home_team_goal'] < row['away_team_goal']:
            return 1  # Away win
        else:
            return 0  # Draw

    matches_df['result'] = matches_df.apply(calculate_result, axis=1)

    # Reorder columns: home first, then away
    ordered_columns = (
        [f'home_{col}' for col in selected_features] +
        [f'away_{col}' for col in selected_features]
    )
    matches_df = matches_df[ordered_columns + ['result']]

    # Drop rows with null values
    matches_df = matches_df.dropna()

    # One-hot encode categorical features (columns ending with 'Class')
    categorical_columns = [col for col in matches_df.columns if col.endswith('Class')]
    matches_df = pd.get_dummies(matches_df, columns=categorical_columns, drop_first=True)

    # Ensure 'result' is the last column
    result_column = matches_df.pop('result')
    matches_df['result'] = result_column

    # Close the database connection
    conn.close()

    # Save processed DataFrame to a CSV file for verification
    matches_df.to_csv('test.csv', index=False)

    # Print the first few rows of the processed DataFrame
    print("Processed Matches DataFrame:")
    print(matches_df.head())

    return matches_df