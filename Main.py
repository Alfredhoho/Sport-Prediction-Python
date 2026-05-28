from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from app.services.football_service import get_team_stats
from app.services.prediction_service import predict_winner
from app.services.ai_service import generate_analysis

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")


@app.get("/")
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )


@app.get("/predict")
async def predict(home_team: int, away_team: int):

    home_stats = await get_team_stats(home_team)

    away_stats = await get_team_stats(away_team)

    prediction = predict_winner(
        home_stats,
        away_stats
    )

    analysis = await generate_analysis(prediction)

    return {
        "prediction": prediction,
        "analysis": analysis
    }