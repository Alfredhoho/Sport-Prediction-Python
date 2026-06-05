"""Soccer prediction model with exponential recency weighting."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}


@dataclass(frozen=True)
class TeamProfile:
    team: str
    matches: int
    weighted_matches: float
    attack_home: float
    defense_home: float
    attack_away: float
    defense_away: float


class SoccerPoissonModel:

    def __init__(self, data_path: str | Path, max_goals: int = 8, half_life_days: int = 365) -> None:
        self.data_path = Path(data_path)
        self.max_goals = max_goals
        self.half_life_days = half_life_days
        self.matches = self._load_matches(self.data_path)
        self.matches = self._add_recency_weights(self.matches)
        self.leagues = sorted(self.matches["League"].dropna().unique().tolist()) if "League" in self.matches else []
        self.avg_home_goals = self._weighted_mean(self.matches["FTHG"], self.matches["weight"])
        self.avg_away_goals = self._weighted_mean(self.matches["FTAG"], self.matches["weight"])
        self.profiles = self._build_profiles()

    @staticmethod
    def _load_matches(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found: {path}. Run scripts/download_football_data.py first."
            )
        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        df = df.copy()
        if "League" not in df.columns:
            df["League"] = df["Div"] if "Div" in df.columns else "Unknown"
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        else:
            df["Date"] = pd.NaT
        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        df = df[(df["FTHG"] >= 0) & (df["FTAG"] >= 0)]
        return df.reset_index(drop=True)

    def _add_recency_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        dated = df["Date"].notna()
        if not dated.any():
            df["weight"] = 1.0
            return df

        latest_date = df.loc[dated, "Date"].max()
        age_days = (latest_date - df["Date"]).dt.days
        age_days = age_days.fillna(self.half_life_days * 2).clip(lower=0)
        df["weight"] = 0.5 ** (age_days / self.half_life_days)
        df["weight"] = df["weight"].clip(lower=0.05, upper=1.0)
        return df

    @staticmethod
    def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
        total_weight = float(weights.sum())
        if total_weight <= 0:
            return float(values.mean())
        return float((values * weights).sum() / total_weight)

    def _weighted_rate(self, values: pd.Series, weights: pd.Series, smoothing_value: float) -> float:
        return float(((values * weights).sum() + smoothing_value) / (weights.sum() + 1.0))

    def _build_profiles(self) -> dict[str, TeamProfile]:
        profiles: dict[str, TeamProfile] = {}
        teams = sorted(set(self.matches["HomeTeam"]).union(set(self.matches["AwayTeam"])))

        for team in teams:
            home = self.matches[self.matches["HomeTeam"] == team]
            away = self.matches[self.matches["AwayTeam"] == team]
            matches = len(home) + len(away)
            weighted_matches = float(home["weight"].sum() + away["weight"].sum())

            home_for = self._weighted_rate(home["FTHG"], home["weight"], self.avg_home_goals)
            home_against = self._weighted_rate(home["FTAG"], home["weight"], self.avg_away_goals)
            away_for = self._weighted_rate(away["FTAG"], away["weight"], self.avg_away_goals)
            away_against = self._weighted_rate(away["FTHG"], away["weight"], self.avg_home_goals)

            profiles[team] = TeamProfile(
                team=team,
                matches=matches,
                weighted_matches=weighted_matches,
                attack_home=home_for / max(self.avg_home_goals, 0.01),
                defense_home=home_against / max(self.avg_away_goals, 0.01),
                attack_away=away_for / max(self.avg_away_goals, 0.01),
                defense_away=away_against / max(self.avg_home_goals, 0.01),
            )
        return profiles

    @property
    def teams(self) -> list[str]:
        return sorted(self.profiles.keys())

    def _expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        home = self.profiles[home_team]
        away = self.profiles[away_team]

        expected_home = self.avg_home_goals * home.attack_home * away.defense_away
        expected_away = self.avg_away_goals * away.attack_away * home.defense_home
        return min(max(expected_home, 0.15), 5.0), min(max(expected_away, 0.15), 5.0)

    @staticmethod
    def _poisson_probability(lam: float, goals: int) -> float:
        return (exp(-lam) * (lam ** goals)) / factorial(goals)

    def _score_matrix(self, expected_home: float, expected_away: float) -> list[list[float]]:
        return [
            [
                self._poisson_probability(expected_home, h) * self._poisson_probability(expected_away, a)
                for a in range(self.max_goals + 1)
            ]
            for h in range(self.max_goals + 1)
        ]

    def predict(self, home_team: str, away_team: str) -> dict[str, Any]:
        if home_team == away_team:
            raise ValueError("Choose two different teams.")
        if home_team not in self.profiles:
            raise ValueError(f"Unknown home team: {home_team}")
        if away_team not in self.profiles:
            raise ValueError(f"Unknown away team: {away_team}")

        expected_home, expected_away = self._expected_goals(home_team, away_team)
        matrix = self._score_matrix(expected_home, expected_away)

        home_win = draw = away_win = 0.0
        score_probs: list[dict[str, Any]] = []
        for home_goals, row in enumerate(matrix):
            for away_goals, prob in enumerate(row):
                if home_goals > away_goals:
                    home_win += prob
                elif home_goals == away_goals:
                    draw += prob
                else:
                    away_win += prob
                score_probs.append({
                    "score": f"{home_goals}-{away_goals}",
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "probability": prob,
                })

        total = home_win + draw + away_win
        home_win, draw, away_win = home_win / total, draw / total, away_win / total
        top_scores = sorted(score_probs, key=lambda item: item["probability"], reverse=True)[:5]

        home_profile = self.profiles[home_team]
        away_profile = self.profiles[away_team]
        confidence = min(0.95, 0.45 + min(home_profile.weighted_matches, away_profile.weighted_matches) / 80)

        latest_date = self.matches["Date"].max()
        latest_date_value = latest_date.date().isoformat() if pd.notna(latest_date) else "unknown"

        return {
            "home_team": home_team,
            "away_team": away_team,
            "expected_goals": {"home": round(expected_home, 3), "away": round(expected_away, 3)},
            "probabilities": {
                "home_win": round(home_win, 4),
                "draw": round(draw, 4),
                "away_win": round(away_win, 4),
            },
            "top_scores": [
                {"score": item["score"], "probability": round(item["probability"] / total, 4)}
                for item in top_scores
            ],
            "model_info": {
                "matches_used": int(len(self.matches)),
                "data_path": str(self.data_path),
                "latest_match_date": latest_date_value,
                "avg_home_goals": round(self.avg_home_goals, 3),
                "avg_away_goals": round(self.avg_away_goals, 3),
                "home_team_matches": home_profile.matches,
                "away_team_matches": away_profile.matches,
                "home_team_weighted_matches": round(home_profile.weighted_matches, 1),
                "away_team_weighted_matches": round(away_profile.weighted_matches, 1),
                "half_life_days": self.half_life_days,
                "confidence": round(confidence, 2),
                "method": "Recency-weighted Maher-style attack/defense Poisson model",
            },
        }
