from flask import Flask, render_template, request, jsonify
from main import get_steam_app_id, scrape_steam_reviews, summarize_top_reviews_gemini, format_bullet_points

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    summaries = {"positive_summary": "", "negative_summary": ""}
    error = None
    if request.method == "POST":
        game_name = request.form.get("game_name", "")
        game_id = get_steam_app_id(game_name)
        if game_id:
            df = scrape_steam_reviews(game_id)
            result = summarize_top_reviews_gemini(df)
            summaries = {
                "positive_summary": format_bullet_points(result["positive_summary"]),
                "negative_summary": format_bullet_points(result["negative_summary"])
            }
        else:
            error = "Game not found or API error."
    return render_template("index.html", summaries=summaries, error=error)

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

if __name__ == "__main__":
    app.run(debug=True)
