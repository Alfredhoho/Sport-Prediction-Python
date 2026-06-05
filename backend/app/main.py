from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .ai_explainer import explain_prediction
from .model import SoccerPoissonModel

load_dotenv()

DATA_PATH = Path(os.getenv("DATA_PATH", "./data/downloaded_matches.csv"))
HALF_LIFE_DAYS = int(os.getenv("HALF_LIFE_DAYS", "365"))
MODEL = SoccerPoissonModel(DATA_PATH, half_life_days=HALF_LIFE_DAYS)

app = FastAPI(
    title="FormGuide Soccer Predictor API",
    description="Educational soccer predictor using Football-Data.co.uk results, recency weighting, and optional AI explanation.",
    version="1.1.0",
)

allowed_origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    home_team: str = Field(..., min_length=1)
    away_team: str = Field(..., min_length=1)
    include_ai_explanation: bool = True


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "matches_loaded": len(MODEL.matches),
        "teams_loaded": len(MODEL.teams),
        "data_path": str(DATA_PATH),
        "half_life_days": HALF_LIFE_DAYS,
    }


@app.get("/teams")
def teams() -> dict[str, object]:
    return {
        "teams": MODEL.teams,
        "leagues": MODEL.leagues,
        "matches_loaded": len(MODEL.matches),
        "half_life_days": HALF_LIFE_DAYS,
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, object]:
    try:
        prediction = MODEL.predict(request.home_team, request.away_team)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if request.include_ai_explanation:
        prediction["ai_explanation"] = explain_prediction(prediction)
    else:
        prediction["ai_explanation"] = None
    return prediction
