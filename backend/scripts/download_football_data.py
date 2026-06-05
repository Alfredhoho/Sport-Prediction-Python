"""Download and normalize Football-Data.co.uk match-result CSV files."""

from __future__ import annotations

import csv
import io
import urllib.request
from pathlib import Path

SEASONS = ["2122", "2223", "2324", "2425", "2526"]
LEAGUES = {
    "E0": "England - Premier League",
    "SP1": "Spain - La Liga",
    "D1": "Germany - Bundesliga",
    "I1": "Italy - Serie A",
    "F1": "France - Ligue 1",
}
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{division}.csv"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "downloaded_matches.csv"
FIELDNAMES = ["Div", "League", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def download_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "APCSA-classroom-project/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("latin-1")


def result_code(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def normalize_csv(season: str, division: str, text: str) -> list[dict[str, object]]:
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, object]] = []
    for row in reader:
        if not row.get("HomeTeam") or not row.get("AwayTeam"):
            continue
        try:
            home_goals = int(float(row.get("FTHG", "")))
            away_goals = int(float(row.get("FTAG", "")))
        except ValueError:
            # Future fixtures have blank final-score fields.
            continue

        rows.append(
            {
                "Div": division,
                "League": LEAGUES[division],
                "Date": row.get("Date", ""),
                "HomeTeam": row.get("HomeTeam", "").strip(),
                "AwayTeam": row.get("AwayTeam", "").strip(),
                "FTHG": home_goals,
                "FTAG": away_goals,
                "FTR": row.get("FTR") or result_code(home_goals, away_goals),
            }
        )
    return rows


def main() -> None:
    all_rows: list[dict[str, object]] = []

    for season in SEASONS:
        for division in LEAGUES:
            url = BASE_URL.format(season=season, division=division)
            print(f"Downloading {division} {season}: {url}")
            try:
                text = download_text(url)
                rows = normalize_csv(season, division, text)
                all_rows.extend(rows)
                print(f"  kept {len(rows)} completed matches")
            except Exception as exc:
                print(f"  skipped: {exc}")

    if not all_rows:
        raise RuntimeError("No matches downloaded. Check your internet connection or source URLs.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} matches to {OUTPUT}")


if __name__ == "__main__":
    main()
