import pandas as pd
import requests
import io
from datetime import datetime

class SoccerDataFetcher:
    def __init__(self):
        self.base_url = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"
        
    def generate_seasons(self):
        seasons = []
        current_year = datetime.now().year
        for i in range(20):
            start_yr = (current_year - i - 1) % 100
            end_yr = (current_year - i) % 100
            season_str = f"{start_yr:02d}{end_yr:02d}"
            seasons.append(season_str)
        return seasons

    def fetch_historical_data(self):
        seasons = self.generate_seasons()
        combined_df = []
        
        print("Gathering 20 years of historical league data...")
        for season in seasons:
            url = self.base_url.format(season)
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    df = pd.read_csv(io.StringIO(response.text))
                    valid_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
                    if all(col in df.columns for col in valid_cols):
                        df = df[valid_cols].dropna()
                        combined_df.append(df)
            except Exception as e:
                continue
                
        if not combined_df:
            raise Exception("Failed to harvest soccer data from historical web endpoints.")
            
        final_df = pd.concat(combined_df, ignore_index=True)
        final_df['FTHG'] = final_df['FTHG'].astype(int)
        final_df['FTAG'] = final_df['FTAG'].astype(int)
        final_df['HomeTeam'] = final_df['HomeTeam'].str.strip()
        final_df['AwayTeam'] = final_df['AwayTeam'].str.strip()
        return final_df