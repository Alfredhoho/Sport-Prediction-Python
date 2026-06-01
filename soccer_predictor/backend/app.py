from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from data_fetcher import SoccerDataFetcher
from predictor import ScorePredictor

app = FastAPI(title="AI Soccer Predictor Engine")

# Fetch and train on init
fetcher = SoccerDataFetcher()
historical_data = fetcher.fetch_historical_data()
predictor = ScorePredictor(historical_data)

class PredictionRequest(BaseModel):
    home_team: str
    away_team: str

@app.get("/teams")
def get_teams():
    return {"teams": predictor.teams}

@app.post("/predict")
def predict(request: PredictionRequest):
    if request.home_team not in predictor.teams or request.away_team not in predictor.teams:
        raise HTTPException(status_code=400, detail="Requested team parameters are invalid.")
    
    result = predictor.predict_match(request.home_team, request.away_team)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)