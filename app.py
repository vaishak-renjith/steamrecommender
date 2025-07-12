from flask import Flask, render_template, request
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

if __name__ == "__main__":
    app.run(debug=True)
