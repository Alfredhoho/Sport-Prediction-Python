import numpy as np
import pandas as pd
from scipy.stats import poisson

class ScorePredictor:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.teams = sorted(data['HomeTeam'].unique())
        self.compute_team_strengths()
        
    def compute_team_strengths(self):
        self.avg_home_scored = self.data['FTHG'].mean()
        self.avg_away_scored = self.data['FTAG'].mean()
        
     
        home_attack = self.data.groupby('HomeTeam')['FTHG'].mean() / self.avg_home_scored
        home_defense = self.data.groupby('HomeTeam')['FTAG'].mean() / self.avg_away_scored
        

        away_attack = self.data.groupby('AwayTeam')['FTAG'].mean() / self.avg_away_scored
        away_defense = self.data.groupby('AwayTeam')['FTHG'].mean() / self.avg_home_scored
        

        self.team_stats = {}
        for team in self.teams:
            self.team_stats[team] = {
                'home_attack': home_attack.get(team, 1.0),
                'home_defense': home_defense.get(team, 1.0),
                'away_attack': away_attack.get(team, 1.0),
                'away_defense': away_defense.get(team, 1.0)
            }

    def predict_match(self, home_team: str, away_team: str):
        if home_team not in self.team_stats or away_team not in self.team_stats:
            return {"error": "One or both teams not found in historical data records."}
            
  
        lambda_home = self.team_stats[home_team]['home_attack'] * self.team_stats[away_team]['away_defense'] * self.avg_home_scored
        lambda_away = self.team_stats[away_team]['away_attack'] * self.team_stats[home_team]['home_defense'] * self.avg_away_scored
        
    
        max_goals = 6
        home_probs = [poisson.pmf(i, lambda_home) for i in range(max_goals)]
        away_probs = [poisson.pmf(i, lambda_away) for i in range(max_goals)]
        
        
        best_prob = 0
        pred_home_score, pred_away_score = 0, 0
        
        home_win_prob = 0
        away_win_prob = 0
        draw_prob = 0
        
        for h in range(max_goals):
            for a in range(max_goals):
                prob = home_probs[h] * away_probs[a]
                if prob > best_prob:
                    best_prob = prob
                    pred_home_score, pred_away_score = h, a
                
                if h > a:
                    home_win_prob += prob
                elif a > h:
                    away_win_prob += prob
                else:
                    draw_prob += prob

        return {
            "predicted_score": f"{pred_home_score} - {pred_away_score}",
            "probabilities": {
                "home_win": round(float(home_win_prob) * 100, 2),
                "draw": round(float(draw_prob) * 100, 2),
                "away_win": round(float(away_win_prob) * 100, 2)
            },
            "expected_goals": {
                "home": round(float(lambda_home), 2),
                "away": round(float(lambda_away), 2)
            }
        }