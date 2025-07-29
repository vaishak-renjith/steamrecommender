from flask import Flask, request, jsonify
from main import get_steam_app_id, scrape_steam_reviews, summarize_top_reviews_gemini

app = Flask(__name__)

@app.route("/api/recommendations", methods=["GET"])
def get_recommendations():
    game_name = request.args.get("game")
    if not game_name:
        return jsonify({"error": "Missing 'game' query parameter"}), 400

    game_id = get_steam_app_id(game_name)
    if not game_id:
        return jsonify({"error": "Game not found or API error"}), 404

    df = scrape_steam_reviews(game_id)
    result = summarize_top_reviews_gemini(df)

    return jsonify({
        "positive_summary": result["positive_summary"],
        "negative_summary": result["negative_summary"]
    })
