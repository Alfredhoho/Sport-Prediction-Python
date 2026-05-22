#!/usr/bin/env python3
"""Predict European football scores using real match data and optional AI predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests  # type: ignore
except ImportError:
    requests = None


@dataclass
class Match:
    date: str
    competition: str
    home_team: str
    away_team: str
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None


@dataclass
class TeamStrength:
    attack: float
    defense: float
    home_advantage: float = 1.05


@dataclass
class Prediction:
    fixture: Match
    home_goals: int
    away_goals: int
    probability: Optional[float]
    source: str


def load_matches_from_csv(path: Path) -> List[Match]:
    matches: List[Match] = []
    with path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            home_goals = int(row["home_goals"]) if row.get("home_goals") else None
            away_goals = int(row["away_goals"]) if row.get("away_goals") else None
            matches.append(
                Match(
                    date=row.get("date", ""),
                    competition=row.get("competition", ""),
                    home_team=row["home_team"].strip(),
                    away_team=row["away_team"].strip(),
                    home_goals=home_goals,
                    away_goals=away_goals,
                )
            )
    return matches


def estimate_team_strengths(matches: List[Match]) -> Tuple[Dict[str, TeamStrength], float]:
    team_home_goals: Dict[str, List[int]] = defaultdict(list)
    team_away_goals: Dict[str, List[int]] = defaultdict(list)
    team_home_allowed: Dict[str, List[int]] = defaultdict(list)
    team_away_allowed: Dict[str, List[int]] = defaultdict(list)

    total_goals = 0
    total_matches = 0

    for match in matches:
        if match.home_goals is None or match.away_goals is None:
            continue

        h = match.home_team
        a = match.away_team
        hg = match.home_goals
        ag = match.away_goals

        team_home_goals[h].append(hg)
        team_away_goals[a].append(ag)
        team_home_allowed[h].append(ag)
        team_away_allowed[a].append(hg)

        total_goals += hg + ag
        total_matches += 1

    league_avg = (total_goals / total_matches) if total_matches else 1.5
    strengths: Dict[str, TeamStrength] = {}

    for team in set(team_home_goals) | set(team_away_goals) | set(team_home_allowed) | set(team_away_allowed):
        home_score_avg = sum(team_home_goals[team]) / len(team_home_goals[team]) if team_home_goals[team] else league_avg
        away_score_avg = sum(team_away_goals[team]) / len(team_away_goals[team]) if team_away_goals[team] else league_avg
        home_allowed_avg = sum(team_home_allowed[team]) / len(team_home_allowed[team]) if team_home_allowed[team] else league_avg
        away_allowed_avg = sum(team_away_allowed[team]) / len(team_away_allowed[team]) if team_away_allowed[team] else league_avg

        attack = (home_score_avg + away_score_avg) / league_avg / 2
        defense = ((home_allowed_avg + away_allowed_avg) / 2) / league_avg
        strengths[team] = TeamStrength(attack=max(0.5, attack), defense=max(0.5, defense))

    return strengths, league_avg


def fetch_json(url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, str]] = None) -> dict:
    headers = headers or {}
    if requests is not None:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} error fetching {url}: {exc.reason}") from exc


def fetch_fixtures_from_football_data_org(
    api_key: str,
    date_from: str,
    date_to: str,
    competitions: List[str],
) -> List[Match]:
    url = "https://api.football-data.org/v4/matches"
    headers = {
        "X-Auth-Token": api_key,
        "User-Agent": "SportPrediction/1.0",
    }
    params = {
        "dateFrom": date_from,
        "dateTo": date_to,
        "competitions": ",".join(competitions),
    }
    payload = fetch_json(url, headers=headers, params=params)
    matches: List[Match] = []
    for item in payload.get("matches", []):
        status = item.get("status", "")
        if status not in {"SCHEDULED", "TIMED", "IN_PLAY", "PAUSED"}:
            continue

        home = item.get("homeTeam", {}).get("name", "").strip()
        away = item.get("awayTeam", {}).get("name", "").strip()
        competition = item.get("competition", {}).get("name", "")
        date = item.get("utcDate", "")[:10]

        if home and away:
            matches.append(Match(date=date, competition=competition, home_team=home, away_team=away))

    return matches


def predict_expected_goals(
    home: str,
    away: str,
    strengths: Dict[str, TeamStrength],
    league_avg: float,
    default_strength: TeamStrength,
) -> Tuple[float, float]:
    home_strength = strengths.get(home, default_strength)
    away_strength = strengths.get(away, default_strength)

    home_exp = league_avg * home_strength.attack * away_strength.defense * home_strength.home_advantage
    away_exp = league_avg * away_strength.attack * home_strength.defense

    return max(0.05, home_exp), max(0.05, away_exp)


def poisson_probability(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def most_likely_score(home_exp: float, away_exp: float, max_goals: int = 6) -> Tuple[int, int, float]:
    best_score = (0, 0)
    best_prob = 0.0
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob = poisson_probability(home_goals, home_exp) * poisson_probability(away_goals, away_exp)
            if prob > best_prob:
                best_prob = prob
                best_score = (home_goals, away_goals)
    return best_score[0], best_score[1], best_prob


def predict_fixtures(
    fixtures: List[Match],
    strengths: Dict[str, TeamStrength],
    league_avg: float,
    max_goals: int = 6,
) -> List[Prediction]:
    default_strength = TeamStrength(attack=1.0, defense=1.0)
    predictions: List[Prediction] = []
    for fixture in fixtures:
        home_exp, away_exp = predict_expected_goals(
            fixture.home_team,
            fixture.away_team,
            strengths,
            league_avg,
            default_strength,
        )
        home_goals, away_goals, probability = most_likely_score(home_exp, away_exp, max_goals=max_goals)
        predictions.append(
            Prediction(
                fixture=fixture,
                home_goals=home_goals,
                away_goals=away_goals,
                probability=probability,
                source="poisson",
            )
        )
    return predictions


def extract_json_array(text: str) -> Optional[str]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def call_openai_chat_api(api_key: str, model: str, prompt: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a football analytics assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }
    if requests is not None:
        response = requests.post(url, headers=headers, json=body, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenAI request failed: {exc.code} {exc.reason}") from exc


def predict_scores_with_ai(
    fixtures: List[Match],
    strengths: Dict[str, TeamStrength],
    league_avg: float,
    openai_api_key: str,
    model: str = "gpt-3.5-turbo",
) -> List[Prediction]:
    default_strength = TeamStrength(attack=1.0, defense=1.0)
    lines: List[str] = []
    for fixture in fixtures:
        home_exp, away_exp = predict_expected_goals(
            fixture.home_team,
            fixture.away_team,
            strengths,
            league_avg,
            default_strength,
        )
        lines.append(
            f"{fixture.home_team} vs {fixture.away_team} ({fixture.competition}) on {fixture.date} - expected goals {home_exp:.2f}:{away_exp:.2f}"
        )

    prompt = (
        "Predict the most likely final score for each fixture listed below. "
        "Use the expected goal estimates to determine the likely score, and return only a JSON array with objects "
        "containing home_team, away_team, and predicted_score in X-Y format. Do not include any other explanation. "
        "If a team appears that the model does not recognize, still respond using the team names exactly as written.\n\n"
        "Fixtures:\n"
        + "\n".join(lines)
    )

    response_text = call_openai_chat_api(openai_api_key, model, prompt)
    json_text = extract_json_array(response_text)
    predictions: List[Prediction] = []
    if not json_text:
        raise ValueError("Could not parse JSON array from OpenAI response.")

    data = json.loads(json_text)
    if not isinstance(data, list):
        raise ValueError("OpenAI returned JSON that is not a list.")

    fixture_map = {(f.home_team, f.away_team, f.date): f for f in fixtures}
    for item in data:
        home_team = item.get("home_team") if isinstance(item, dict) else None
        away_team = item.get("away_team") if isinstance(item, dict) else None
        score_text = item.get("predicted_score") or item.get("score") if isinstance(item, dict) else None
        if not home_team or not away_team or not score_text or "-" not in score_text:
            continue

        home_goals_str, away_goals_str = score_text.strip().split("-")
        try:
            home_goals = int(home_goals_str)
            away_goals = int(away_goals_str)
        except ValueError:
            continue

        fixture = fixture_map.get((home_team, away_team, next((f.date for f in fixtures if f.home_team == home_team and f.away_team == away_team), "")))
        if fixture is None:
            fixture = Match(date="", competition="", home_team=home_team, away_team=away_team)

        predictions.append(
            Prediction(
                fixture=fixture,
                home_goals=home_goals,
                away_goals=away_goals,
                probability=None,
                source="ai",
            )
        )

    if not predictions:
        raise ValueError("AI returned predictions but none could be parsed into scores.")

    return predictions


def sample_european_fixtures() -> List[Match]:
    return [
        Match(date="2026-05-22", competition="UEFA Champions League", home_team="Manchester City", away_team="Real Madrid"),
        Match(date="2026-05-22", competition="UEFA Europa League", home_team="Sevilla", away_team="Arsenal"),
        Match(date="2026-05-23", competition="Premier League", home_team="Liverpool", away_team="Chelsea"),
        Match(date="2026-05-24", competition="La Liga", home_team="Barcelona", away_team="Atletico Madrid"),
        Match(date="2026-05-24", competition="Serie A", home_team="Juventus", away_team="Inter Milan"),
        Match(date="2026-05-25", competition="Bundesliga", home_team="Bayern Munich", away_team="Borussia Dortmund"),
    ]


def sample_historical_data() -> List[Match]:
    return [
        Match(date="2026-04-01", competition="Premier League", home_team="Manchester City", away_team="Liverpool", home_goals=3, away_goals=1),
        Match(date="2026-04-03", competition="La Liga", home_team="Barcelona", away_team="Real Madrid", home_goals=2, away_goals=2),
        Match(date="2026-04-05", competition="Bundesliga", home_team="Bayern Munich", away_team="Borussia Dortmund", home_goals=4, away_goals=2),
        Match(date="2026-04-06", competition="Serie A", home_team="Juventus", away_team="Inter Milan", home_goals=1, away_goals=1),
        Match(date="2026-04-07", competition="UEFA Champions League", home_team="Arsenal", away_team="Sevilla", home_goals=3, away_goals=0),
        Match(date="2026-04-08", competition="UEFA Europa League", home_team="Real Madrid", away_team="Manchester City", home_goals=2, away_goals=3),
        Match(date="2026-04-10", competition="Premier League", home_team="Chelsea", away_team="Manchester City", home_goals=1, away_goals=2),
        Match(date="2026-04-11", competition="La Liga", home_team="Atletico Madrid", away_team="Barcelona", home_goals=0, away_goals=2),
    ]


def print_predictions(predictions: List[Prediction]) -> None:
    print("Predicted European football scores:")
    print(
        "Date       | Competition              | Home Team          | Away Team          | Predicted | Prob   | Source"
    )
    print("-" * 108)
    for prediction in predictions:
        fixture = prediction.fixture
        prob_text = f"{prediction.probability:.3%}" if prediction.probability is not None else "N/A"
        print(
            f"{fixture.date:10} | {fixture.competition:24} | {fixture.home_team:18} | {fixture.away_team:18} | "
            f"{prediction.home_goals}-{prediction.away_goals:3} | {prob_text:6} | {prediction.source}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict European football match scores using sample data, real fixtures, and optional AI-based scoring."
    )
    parser.add_argument(
        "--historical",
        type=Path,
        help="CSV file of past match results with columns: date,competition,home_team,away_team,home_goals,away_goals",
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        help="CSV file of upcoming fixtures with columns: date,competition,home_team,away_team",
    )
    parser.add_argument(
        "--fetch-real-fixtures",
        action="store_true",
        help="Fetch upcoming European fixtures from football-data.org using an API key.",
    )
    parser.add_argument(
        "--football-data-api-key",
        help="Football-data.org API key. Can also be set with FOOTBALL_DATA_API_KEY.",
    )
    parser.add_argument(
        "--competitions",
        nargs="+",
        default=["CL", "PL", "BL1", "SA", "PD", "ELC"],
        help="Competition codes for football-data.org. Defaults to Champions League, Premier League, Bundesliga, Serie A, La Liga, Europa League.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days from today to fetch upcoming real fixtures.",
    )
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key for AI-based score predictions. Can also be set with OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--ai-model",
        default="gpt-3.5-turbo",
        help="OpenAI model name for AI score prediction.",
    )
    parser.add_argument(
        "--max-goals",
        type=int,
        default=6,
        help="Maximum number of goals to consider for Poisson-based predictions.",
    )
    return parser.parse_args()


def build_date_range(days: int) -> Tuple[str, str]:
    today = datetime.utcnow().date()
    return today.isoformat(), (today + timedelta(days=days)).isoformat()


def main() -> None:
    args = parse_args()

    if args.historical and args.historical.exists():
        historical_matches = load_matches_from_csv(args.historical)
    else:
        print("Using built-in sample historical match data because no historical CSV was provided.")
        historical_matches = sample_historical_data()

    strengths, league_avg = estimate_team_strengths(historical_matches)

    fixtures: List[Match]
    if args.fixtures and args.fixtures.exists():
        fixtures = load_matches_from_csv(args.fixtures)
    elif args.fetch_real_fixtures:
        api_key = args.football_data_api_key or os.getenv("FOOTBALL_DATA_API_KEY")
        if not api_key:
            print("Error: --fetch-real-fixtures requires --football-data-api-key or FOOTBALL_DATA_API_KEY.", file=sys.stderr)
            sys.exit(1)
        date_from, date_to = build_date_range(args.days)
        fixtures = fetch_fixtures_from_football_data_org(api_key, date_from, date_to, args.competitions)
        if not fixtures:
            print("No fixtures were returned from football-data.org; falling back to sample fixtures.")
            fixtures = sample_european_fixtures()
    else:
        print("Using built-in sample European fixtures because no fixture CSV or real fixture fetch was provided.")
        fixtures = sample_european_fixtures()

    openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    predictions: List[Prediction]
    if openai_api_key:
        try:
            predictions = predict_scores_with_ai(fixtures, strengths, league_avg, openai_api_key, model=args.ai_model)
        except Exception as exc:
            print(f"AI prediction failed ({exc}); falling back to Poisson predictions.")
            predictions = predict_fixtures(fixtures, strengths, league_avg, max_goals=args.max_goals)
    else:
        predictions = predict_fixtures(fixtures, strengths, league_avg, max_goals=args.max_goals)

    print_predictions(predictions)


if __name__ == "__main__":
    main()
