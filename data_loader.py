import zipfile
import sqlite3
import pandas as pd
import kagglehub

# Loads and processes data
# Returns two dataframes
# One for Match Data
# One for Team Data
def load_data():
    path = kagglehub.dataset_download("hugomathien/soccer")

    # Connect to the SQLite database
    db_path = f'{path}/database.sqlite'
    conn = sqlite3.connect(db_path)

    # Query data from the database into dataframe
    team_attributes_df = pd.read_sql("SELECT * FROM Team_Attributes;", conn)
    matches_df = pd.read_sql("SELECT * FROM Match;", conn)
    league_df = pd.read_sql("SELECT * FROM League;", conn)
    team_df = pd.read_sql("SELECT * FROM Team;", conn)

    # Filter for matches in the England Premier League
    england_league_id = league_df[league_df['name'] == 'England Premier League']['id'].values[0]
    england_matches_df = matches_df[matches_df['league_id'] == england_league_id].copy()

    # Add year column to the matches dataframe from date column
    england_matches_df['year'] = pd.to_datetime(england_matches_df['date']).dt.year

    # Add a 'result' column based on home and away team goals
    def calculate_result(row):
        if row['home_team_goal'] > row['away_team_goal']:
            return 1 # Home win
        elif row['home_team_goal'] < row['away_team_goal']:
            return -1 # Home loss
        else:
            return 0 # Draw

    england_matches_df['result'] = england_matches_df.apply(calculate_result, axis=1)  # Apply previous function to each row

    # Replace home team IDs with home team names
    england_matches_df = england_matches_df.merge(
        team_df[['team_api_id', 'team_long_name']], 
        how='left', 
        left_on='home_team_api_id', 
        right_on='team_api_id'
    ).rename(columns={'team_long_name': 'home_team_name'})

    # Replace away team IDs with away team names
    england_matches_df = england_matches_df.merge(
        team_df[['team_api_id', 'team_long_name']], 
        how='left', 
        left_on='away_team_api_id', 
        right_on='team_api_id'
    ).rename(columns={'team_long_name': 'away_team_name'})

    # Drop unnecessary columns
    england_matches_df = england_matches_df[['home_team_name', 'away_team_name', 'year', 'result']]

    # Filter team attributes to include only Premier League teams
    premier_league_team_names = pd.concat([
        england_matches_df['home_team_name'], 
        england_matches_df['away_team_name']
    ]).unique()

    team_attributes_df = team_attributes_df.merge(
        team_df[['team_api_id', 'team_long_name']],
        how='left',
        on='team_api_id'
    )

    # Filter by year and team names
    team_attributes_df['year'] = pd.to_datetime(team_attributes_df['date']).dt.year
    team_attributes_df = team_attributes_df[
        (team_attributes_df['year'].isin([2014, 2015, 2016, 2017])) & 
        (team_attributes_df['team_long_name'].isin(premier_league_team_names))
    ]

    # Drop unnecessary columns
    team_attributes_df = team_attributes_df.drop(columns=['id', 'team_fifa_api_id', 'team_api_id'])
    team_attributes_df = team_attributes_df[['team_long_name'] + team_attributes_df.select_dtypes(include='number').columns.tolist()]

    # Print the dataframes
    print("Team Attributes:")
    print(team_attributes_df.head())

    print()
    print("England Premier League Matches:")
    print(england_matches_df.head())

    conn.close()

    return team_attributes_df, england_matches_df