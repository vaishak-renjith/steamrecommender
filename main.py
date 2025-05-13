import os
import re
import time
import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import google.generativeai as genai

# Set your Gemini API key securely
genai.configure(api_key="AIzaSyAG5y06JY1AJ_98vtiZqrdPcxIY00YdaKw")  # Set GEMINI_API_KEY in environment variables

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def get_steam_app_id(game_name):
    """
    Finds the Steam App ID for a given game name using Steam search page.
    """
    search_url = f"https://store.steampowered.com/search/?term={game_name.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(search_url, headers=headers)
    if response.status_code != 200:
        print("Error fetching search results.")
        return None

    match = re.search(r'data-ds-appid="(\d+)"', response.text)
    return match.group(1) if match else None

def scrape_steam_reviews(game_id, num_reviews=300, min_votes=0):
    """
    Scrapes recent Steam reviews using public API.
    """
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
                        "review": review["review"][:300],  # Truncate to reduce Gemini load
                        "sentiment": sentiment["compound"],
                        "votes_up": review["votes_up"],
                        "recommend": review["voted_up"],
                        "timestamp": review["timestamp_created"]
                    })
            
            cursor = data.get("cursor")
            if not cursor:
                break

        except Exception as e:
            print(f"Error during review fetch: {e}")
            time.sleep(1)
            continue
    
    return pd.DataFrame(all_reviews[:num_reviews])

def summarize_with_gemini(reviews, review_type="positive"):
    """
    Sends summarized prompt to Gemini API for either positive or negative review sets.
    """
    if not reviews:
        return "No data to summarize."

    # Remove personal pronouns to generalize content
    cleaned = [re.sub(r'\b(I|my|me|we|our)\b', '', r, flags=re.IGNORECASE) for r in reviews]

    # Truncate and batch reviews to minimize Gemini input size
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
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return "Summary unavailable."

def summarize_top_reviews_gemini(df, top_n=3):
    """
    Summarizes top positive and negative reviews using Gemini.
    """
    df_sorted = df.sort_values(by="votes_up", ascending=False)

    positive = df_sorted[df_sorted["sentiment"] > 0.5]["review"].head(top_n).tolist()
    negative = df_sorted[df_sorted["sentiment"] < -0.5]["review"].head(top_n).tolist()

    return {
        "positive_summary": summarize_with_gemini(positive, "positive"),
        "negative_summary": summarize_with_gemini(negative, "negative")
    }

def format_bullet_points(text):
    """
    Formats bullet points with consistent styling.
    """
    text = ' '.join(text.split())
    text = text.replace('*', '•')
    text = re.sub(r'•\s*', '\n• ', text).strip()
    if text.startswith('•') and not text.startswith('\n•'):
        text = '\n' + text
    return text

# Example run
if __name__ == "__main__":
    game_name = "don't starve together"
    game_id = get_steam_app_id(game_name)
    
    if game_id:
        reviews_df = scrape_steam_reviews(game_id, num_reviews=300)
        summaries = summarize_top_reviews_gemini(reviews_df)

        print(format_bullet_points("Positive Summary:\n" + summaries["positive_summary"]))
        print(format_bullet_points("Negative Summary:\n" + summaries["negative_summary"]))
    else:
        print("Game not found.")
