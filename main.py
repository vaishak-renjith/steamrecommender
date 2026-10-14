import os
import re
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env file relative to this script
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path)

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if api_key:
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)

analyzer = SentimentIntensityAnalyzer()

# Preferred fast, low-latency Flash models.
# Models like gemini-3.7/3.8 are thinking/reasoning models by default that take 20-40s or timeout (504).
# gemini-2.5-flash responds in ~1-3s, well within interactive web deadlines.
PREFERRED_MODELS = ["gemini-2.5-flash", "gemini-3.6-flash", "gemini-3.5-flash"]

# Bounded timeout (seconds) for each individual Gemini call.
GEMINI_TIMEOUT = 20

# Steam reviews routinely contain profanity and heated gamer criticism.
# Disable blocking so analysis of user reviews does not get silently blocked.
GEMINI_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

_gemini_models = {}
_analysis_cache = {}


@lru_cache(maxsize=1)
def get_fast_gemini_model():
    """Pick a fast, low-latency Gemini Flash model suited for real-time web responses."""
    override = os.getenv("GEMINI_MODEL")
    if override:
        return override
    try:
        models = genai.list_models()
        available_names = {
            m.name.rsplit("/", 1)[-1]
            for m in models
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        }
        for pref in PREFERRED_MODELS:
            if pref in available_names:
                return pref
    except Exception as e:
        print(f"Gemini model discovery error: {e}")
    return PREFERRED_MODELS[0]


def get_latest_gemini_flash_model():
    """Backwards-compatible alias for model discovery."""
    return get_fast_gemini_model()


def get_gemini_model(model_name=None):
    """Retrieve or instantiate a GenerativeModel for the given model name."""
    if model_name is None:
        model_name = get_fast_gemini_model()
    if model_name not in _gemini_models:
        _gemini_models[model_name] = genai.GenerativeModel(model_name)
    return _gemini_models[model_name]


def preload_gemini_model():
    """Instantiate the Gemini model once and reuse it for subsequent requests."""
    return get_gemini_model()


# Preload the model during module import so the first request doesn't pay setup cost
try:
    if api_key:
        preload_gemini_model()
except Exception as e:
    print(f"Gemini preload warning: {e}")


@lru_cache(maxsize=256)
def get_steam_app_id(game_name):
    """Finds the Steam App ID for a given game name using Steam search."""
    if not game_name or not game_name.strip():
        return None
    search_url = f"https://store.steampowered.com/search/?term={requests.utils.quote(game_name.strip())}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        match = re.search(r'data-ds-appid="(\d+)"', response.text)
        return match.group(1) if match else None
    except Exception as e:
        print(f"Steam app id search error for '{game_name}': {e}")
        return None


def _fetch_steam_review_batch(game_id, review_type="all", num_per_page=30):
    url = f"https://store.steampowered.com/appreviews/{game_id}?json=1&filter=all&language=english&review_type={review_type}&num_per_page={num_per_page}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("reviews", [])
    except Exception as e:
        print(f"Error fetching {review_type} reviews for app {game_id}: {e}")
    return []


def scrape_steam_reviews(game_id, num_reviews=60, min_votes=0):
    """
    Scrapes Steam reviews for the given game_id.
    Fetches helpful positive and negative reviews concurrently to ensure balanced coverage
    even for overwhelmingly positive or negative titles, in a single fast network pass (~0.8s).
    """
    half = max(num_reviews // 2, 25)
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_pos = executor.submit(_fetch_steam_review_batch, game_id, "positive", half)
        f_neg = executor.submit(_fetch_steam_review_batch, game_id, "negative", half)
        raw_pos = f_pos.result()
        raw_neg = f_neg.result()

    combined = raw_pos + raw_neg
    if not combined:
        return pd.DataFrame()

    records = []
    seen = set()
    for review in combined:
        r_id = review.get("recommendationid")
        if r_id and r_id in seen:
            continue
        if r_id:
            seen.add(r_id)

        votes_up = review.get("votes_up", 0)
        if votes_up >= min_votes:
            review_text = review.get("review", "")
            sentiment = analyzer.polarity_scores(review_text)
            records.append({
                "review": review_text[:300],
                "sentiment": sentiment["compound"],
                "votes_up": votes_up,
                "recommend": review.get("voted_up", True),
                "timestamp": review.get("timestamp_created", 0)
            })

    return pd.DataFrame(records[:num_reviews])


def _is_copypasta(text):
    """Detects ASCII checklist copypasta and non-substantive review templates."""
    if text.count("☐") > 1 or text.count("☑") > 1:
        return True
    if text.count("[ ]") > 1 or text.count("[x]") > 1 or text.count("[-]") > 1:
        return True
    letters = sum(1 for c in text if c.isalpha())
    if len(text) > 20 and (letters / len(text)) < 0.45:
        return True
    return False


def clean_reviews_for_prompt(reviews):
    """Strips personal pronouns, copypasta, and truncates reviews to keep prompt compact and fast."""
    if not reviews:
        return []
    cleaned_list = []
    for r in reviews:
        if _is_copypasta(r):
            continue
        # Remove personal pronouns to generalize feedback
        cleaned = re.sub(r'\b(I|my|me|we|our)\b', '', r, flags=re.IGNORECASE)
        cleaned = cleaned.strip()[:180]
        if len(cleaned) > 15:
            cleaned_list.append(cleaned)
    return cleaned_list


def generate_with_gemini_fallback(prompt, timeout=GEMINI_TIMEOUT):
    """
    Attempts generation with the primary model, falling back to alternate models
    if a model experiences a timeout or per-model rate limit (429).
    """
    primary = get_fast_gemini_model()
    candidates = [primary] + [m for m in PREFERRED_MODELS if m != primary]
    last_err = None

    for model_name in candidates:
        try:
            model = get_gemini_model(model_name)
            response = model.generate_content(
                prompt,
                request_options={"timeout": timeout},
                safety_settings=GEMINI_SAFETY_SETTINGS,
            )
            candidate = response.candidates[0] if response.candidates else None
            parts = getattr(getattr(candidate, "content", None), "parts", None)
            if candidate and parts:
                text = response.text.strip()
                if text:
                    return text
            finish_reason = getattr(candidate, "finish_reason", "NO_CANDIDATES")
            print(f"Gemini returned no parts on {model_name}: finish_reason={finish_reason}")
        except Exception as e:
            last_err = e
            print(f"Gemini error with {model_name}: {e}")
            continue

    raise last_err or Exception("All Gemini models failed")


def summarize_with_gemini(reviews, review_type="positive"):
    """
    Summarizes a single set of reviews (positive or negative) with Gemini.
    """
    cleaned = clean_reviews_for_prompt(reviews)
    if not cleaned:
        return "No data to summarize."

    prompt = f"""Analyze the following game reviews and summarize consistent {review_type} points:
{cleaned}
Rules:
- List at least 2-3 consistent {review_type} points ONLY
- No neutral/mixed feedback
- No "this review" or "one review" qualifiers
- No use of the phrase "subjectively"
- Each point must apply to at least 2 reviews
- Maximum 10 words per point
- Format as bullet points with '-'
- No introductory phrases
"""
    try:
        return generate_with_gemini_fallback(prompt)
    except Exception as e:
        print(f"Gemini error ({review_type}): {e}")
        err_msg = str(e).lower()
        if "429" in err_msg or "quota" in err_msg:
            return "Summary unavailable (API rate limit exceeded)."
        elif "timed out" in err_msg or "deadline" in err_msg:
            return "Summary unavailable (AI response timed out)."
        return "Summary unavailable."


def summarize_top_reviews_gemini(df, top_n=4):
    """
    Summarizes top positive and negative reviews using Gemini.
    Uses a single unified Gemini request when both sides are available to halve API calls and latency,
    avoiding rate limit exhaustion and eliminating unnecessary timeouts.
    """
    if df is None or df.empty:
        return {
            "positive_summary": "No data to summarize.",
            "negative_summary": "No data to summarize."
        }

    df_sorted = df.sort_values(by="votes_up", ascending=False)

    pos_df = df_sorted[df_sorted["recommend"] == True]
    if pos_df.empty:
        pos_df = df_sorted[df_sorted["sentiment"] > 0]
    positive = pos_df["review"].tolist()

    neg_df = df_sorted[df_sorted["recommend"] == False]
    if neg_df.empty:
        neg_df = df_sorted[df_sorted["sentiment"] < 0]
    negative = neg_df["review"].tolist()

    pos_clean = clean_reviews_for_prompt(positive)[:top_n]
    neg_clean = clean_reviews_for_prompt(negative)[:top_n]

    if not pos_clean and not neg_clean:
        return {
            "positive_summary": "No reviews available.",
            "negative_summary": "No reviews available."
        }

    if pos_clean and not neg_clean:
        pos_summary = summarize_with_gemini(pos_clean, "positive")
        return {
            "positive_summary": pos_summary,
            "negative_summary": "No negative reviews found. Community sentiment is overwhelmingly positive."
        }

    if neg_clean and not pos_clean:
        neg_summary = summarize_with_gemini(neg_clean, "negative")
        return {
            "positive_summary": "No positive reviews found. Community sentiment is overwhelmingly negative.",
            "negative_summary": neg_summary
        }

    prompt = f"""Analyze the following game reviews and summarize consistent community feedback:

POSITIVE REVIEWS:
{pos_clean}

NEGATIVE REVIEWS:
{neg_clean}

Rules:
- Under 'POSITIVE:', list 2-3 consistent positive points ONLY
- Under 'NEGATIVE:', list 2-3 consistent negative points ONLY
- Maximum 10 words per point
- Format each point as a bullet with '-'
- No introductory or concluding phrases
- Base points strictly on the provided reviews
"""
    try:
        raw_text = generate_with_gemini_fallback(prompt)
        pos_match = re.search(r'POSITIVE:\s*(.*?)(?=NEGATIVE:|$)', raw_text, re.DOTALL | re.IGNORECASE)
        neg_match = re.search(r'NEGATIVE:\s*(.*)', raw_text, re.DOTALL | re.IGNORECASE)

        pos_text = pos_match.group(1).strip() if pos_match else ""
        neg_text = neg_match.group(1).strip() if neg_match else ""

        if not pos_text and not neg_text:
            pos_text = summarize_with_gemini(pos_clean, "positive")
            neg_text = summarize_with_gemini(neg_clean, "negative")

        return {
            "positive_summary": pos_text or "Summary unavailable.",
            "negative_summary": neg_text or "Summary unavailable."
        }
    except Exception as e:
        print(f"Unified summarization error: {e}")
        err_msg = str(e).lower()
        if "429" in err_msg or "quota" in err_msg:
            msg = "Summary unavailable (API rate limit exceeded). Please try again shortly."
        elif "timed out" in err_msg or "deadline" in err_msg:
            msg = "Summary unavailable (AI response timed out). Please try again."
        else:
            msg = "Summary unavailable."
        return {
            "positive_summary": msg,
            "negative_summary": msg
        }


def get_game_reviews_and_summary(game_id):
    """
    Scrapes and summarizes reviews for a game_id with in-memory caching.
    Ensures repeated lookups respond immediately without burning Gemini rate limits.
    """
    if game_id in _analysis_cache:
        cached = _analysis_cache[game_id]
        if cached.get("positive_summary") != "Summary unavailable." or cached.get("negative_summary") != "Summary unavailable.":
            return cached

    df = scrape_steam_reviews(game_id)
    result = summarize_top_reviews_gemini(df)

    if result.get("positive_summary") != "Summary unavailable." or result.get("negative_summary") != "Summary unavailable.":
        _analysis_cache[game_id] = result

    return result


def format_bullet_points(text):
    """
    Formats bullet points with consistent styling (• per line).
    Strips raw markdown headers and extraneous bullet symbols.
    """
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[\*\-\•\d+\.]\s*', '', line).strip()
        if line.lower().startswith(('positive:', 'negative:')):
            continue
        if line:
            lines.append(f"• {line}")
    if lines:
        return "\n" + "\n".join(lines)
    return text


if __name__ == "__main__":
    game_name = "don't starve together"
    game_id = get_steam_app_id(game_name)

    if game_id:
        result = get_game_reviews_and_summary(game_id)
        print("Positive Summary:\n" + format_bullet_points(result["positive_summary"]))
        print("Negative Summary:\n" + format_bullet_points(result["negative_summary"]))
    else:
        print("Game not found.")
