"""Optional OpenAI-powered explanation layer.

The AI does not make the prediction. It receives the numeric model output and
turns it into structured, user-friendly interpretation for the frontend.
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


FALLBACK_TEMPLATE = {
    "headline": "Prediction generated from historical scoring patterns.",
    "key_factors": [],
    "caution": "This is an educational forecast, not a guarantee. Soccer is noisy and upsets happen.",
    "plain_english": "The model estimates expected goals from team attack and defense strengths, then converts those goal rates into win/draw probabilities."
}


def explain_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_explanation(prediction)

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    system_prompt = (
        "You explain sports-model output for high-school computer science students. "
        "Do not give betting advice, odds shopping, or wagering suggestions. "
        "Return only valid JSON with keys: headline, key_factors, caution, plain_english. "
        "key_factors must be a list of 3 short strings."
    )
    user_prompt = (
        "Explain this soccer prediction. The statistical model, not you, produced the numbers. "
        f"Prediction JSON: {json.dumps(prediction, sort_keys=True)}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        return _validate_explanation(data, prediction)
    except Exception:
        return _fallback_explanation(prediction)


def _fallback_explanation(prediction: dict[str, Any]) -> dict[str, Any]:
    probs = prediction["probabilities"]
    home = prediction["home_team"]
    away = prediction["away_team"]
    winner = max(probs, key=probs.get)
    readable = {
        "home_win": f"{home} win",
        "draw": "draw",
        "away_win": f"{away} win",
    }[winner]
    return {
        "headline": f"The model's most likely outcome is: {readable}.",
        "key_factors": [
            f"Expected goals: {home} {prediction['expected_goals']['home']}, {away} {prediction['expected_goals']['away']}.",
            f"Top scoreline: {prediction['top_scores'][0]['score']}.",
            f"Model confidence: {prediction['model_info']['confidence']} based on available match counts.",
        ],
        "caution": "This is an educational forecast, not a guarantee. Do not use it for betting.",
        "plain_english": "The backend estimates each team's scoring rate from past matches and uses a Poisson distribution to calculate outcome probabilities.",
    }


def _validate_explanation(data: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    fallback = _fallback_explanation(prediction)
    headline = str(data.get("headline") or fallback["headline"])
    key_factors = data.get("key_factors")
    if not isinstance(key_factors, list) or len(key_factors) < 1:
        key_factors = fallback["key_factors"]
    key_factors = [str(item)[:180] for item in key_factors[:3]]
    caution = str(data.get("caution") or fallback["caution"])
    plain_english = str(data.get("plain_english") or fallback["plain_english"])
    return {
        "headline": headline[:220],
        "key_factors": key_factors,
        "caution": caution[:240],
        "plain_english": plain_english[:350],
    }
