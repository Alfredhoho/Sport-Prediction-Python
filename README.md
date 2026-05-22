# Sport-Prediction-Python
Pregame sports analysis tool using historical data and AI

## Usage

Run the prediction script with optional real fixture fetching and AI score prediction.

Example with sample data:

    python3 Main.py

Example using football-data.org fixtures:

    export FOOTBALL_DATA_API_KEY="your_football_data_api_key"
    python3 Main.py --fetch-real-fixtures --days 7

Example using OpenAI for score predictions:

    export OPENAI_API_KEY="your_openai_api_key"
    python3 Main.py --fetch-real-fixtures --days 7

If you prefer local CSV files, pass:

    python3 Main.py --historical historical.csv --fixtures fixtures.csv

