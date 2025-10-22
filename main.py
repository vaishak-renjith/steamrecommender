import os
import re
import time
import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import google.generativeai as genai
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

analyzer = SentimentIntensityAnalyzer()
_gemini_model = None

@lru_cache(maxsize=1)
def get_latest_gemini_flash_model():
    """Pick the newest Gemini Flash model that supports generateContent."""
    fallback = "gemini-flash-latest"
    try:
        models = genai.list_models()
        candidates = [
            model.name
            for model in models
            if "flash" in model.name
            and "lite" not in model.name
            and "live" not in model.name
            and "image" not in model.name
            and "native" not in model.name
            and "tts" not in model.name
            and "generateContent" in getattr(model, "supported_generation_methods", [])
        ]
        if not candidates:
            return fallback

        def version_key(name):
            # Prefer higher semantic versions; fall back to lexical ordering.
            match = re.search(r"gemini-(\d+(?:\.\d+)*)-flash", name)
            if match:
                version_tuple = tuple(int(p) for p in match.group(1).split("."))
            else:
                version_tuple = (-1,)
            is_preview = "preview" in name
            return (version_tuple, not is_preview, name)

        latest = max(candidates, key=version_key)
        return latest.rsplit("/", 1)[-1]
    except Exception as e:
        print(f"Gemini model discovery error: {e}")
        return fallback


def preload_gemini_model():
    """Instantiate the Gemini model once and reuse it for subsequent requests."""
    global _gemini_model
    if _gemini_model is not None:
        return _gemini_model
    model_name = get_latest_gemini_flash_model()
    try:
        _gemini_model = genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"Gemini model preload error: {e}")
        _gemini_model = genai.GenerativeModel("gemini-flash-latest")
    return _gemini_model


# Preload the model during module import so the first request doesn't pay the setup cost.
preload_gemini_model()

def get_steam_app_id(game_name):
    search_url = f"https://store.steampowered.com/search/?term={game_name.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    match = re.search(r'data-ds-appid="(\d+)"', response.text)
    return match.group(1) if match else None

def scrape_steam_reviews(game_id, num_reviews=300, min_votes=0):
    url = f"https://store.steampowered.com/appreviews/{game_id}?json=1&filter=recent&num_per_page=100&language=english"
    all_reviews = []
    cursor = "*"
    while len(all_reviews) < num_reviews:
        try:
            response = requests.get(f"{url}&cursor={cursor}")
            data = response.json()
            reviews = data.get("reviews", [])
            if not reviews:
                break
            for review in reviews:
                if review["votes_up"] >= min_votes:
                    sentiment = analyzer.polarity_scores(review["review"])
                    all_reviews.append({
                        "review": review["review"][:300],
                        "sentiment": sentiment["compound"],
                        "votes_up": review["votes_up"],
                        "recommend": review["voted_up"],
                        "timestamp": review["timestamp_created"]
                    })
            cursor = data.get("cursor")
            if not cursor:
                break
        except:
            time.sleep(1)
            continue
    return pd.DataFrame(all_reviews[:num_reviews])

def summarize_with_gemini(reviews, review_type="positive"):
    if not reviews:
        return "No data to summarize."
    cleaned = [re.sub(r'\b(I|my|me|we|our)\b', '', r, flags=re.IGNORECASE) for r in reviews]
    cleaned = [r.strip()[:200] for r in cleaned if len(r.strip()) > 20]
    prompt = f"""Analyze the following game reviews and summarize consistent {review_type} points:
{cleaned}
    Rules:
    - List at least 2-3 consistent {review_type} points ONLY
    - No neutral/mixed feedback
    - No "this review" or "one review" qualifiers
    - No use of the phrase "subjectively"
    - Each point must apply to at least 2 reviews
    - Maximum 10 words per point
    - Format as bullet points
    - No introductory phrases
"""
    try:
        model = preload_gemini_model()
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return "Summary unavailable."

def summarize_top_reviews_gemini(df, top_n=3):
    df_sorted = df.sort_values(by="votes_up", ascending=False)
    positive = df_sorted[df_sorted["sentiment"] > 0.5]["review"].head(top_n).tolist()
    negative = df_sorted[df_sorted["sentiment"] < -0.5]["review"].head(top_n).tolist()
    return {
        "positive_summary": summarize_with_gemini(positive, "positive"),
        "negative_summary": summarize_with_gemini(negative, "negative")
    }

def format_bullet_points(text):
    text = ' '.join(text.split())
    text = text.replace('*', '•')
    text = re.sub(r'•\s*', '\n• ', text).strip()
    if text.startswith('•') and not text.startswith('\n•'):
        text = '\n' + text
    return text

