"""
Automatic News Headline Generation -- Flask web application (Task 4).

Lets a user either paste article text directly, or upload a .txt file
(single article) or a .csv file (multiple articles, one per row) and get
back a generated headline for each article, plus the top-k beam-search
candidates.
"""
import io
import os
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
sys.path.insert(0, str(ROOT_DIR / "training"))

from infer import HeadlineGenerator  # noqa: E402

MODEL_DIR = os.environ.get("MODEL_DIR", str(ROOT_DIR / "models" / "transformer_scratch"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

_generator = None


def get_generator():
    global _generator
    if _generator is None:
        _generator = HeadlineGenerator(MODEL_DIR)
    return _generator


def generate_for_article(article_text, beam_size, top_k):
    gen = get_generator()
    greedy = gen.greedy_decode(article_text)
    beam_candidates = gen.beam_search_decode(article_text, beam_size=beam_size, top_k=top_k)
    return {
        "article": article_text,
        "article_preview": (article_text[:280] + "...") if len(article_text) > 280 else article_text,
        "greedy_headline": greedy,
        "beam_candidates": beam_candidates,
    }


def extract_articles_from_csv(file_storage, max_rows=25):
    df = pd.read_csv(file_storage)
    if df.empty or len(df.columns) == 0:
        return []
    col = None
    for candidate in ["article", "text", "body", "content"]:
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        col = df.columns[0]
    articles = df[col].astype(str).tolist()[:max_rows]
    return articles


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        beam_size = int(request.form.get("beam_size", 4))
        top_k = int(request.form.get("top_k", 3))
    except (TypeError, ValueError):
        return render_template("index.html", error="Beam size and top-k must be whole numbers."), 400
    if not 1 <= beam_size <= 8 or not 1 <= top_k <= 5:
        return render_template("index.html", error="Beam size must be 1–8 and top-k must be 1–5."), 400
    top_k = min(top_k, beam_size)

    articles = []
    error = None

    pasted_text = (request.form.get("article_text") or "").strip()
    upload = request.files.get("file")

    try:
        if upload and upload.filename:
            filename = upload.filename.lower()
            if filename.endswith(".csv"):
                articles = extract_articles_from_csv(upload)
            elif filename.endswith(".txt"):
                content = upload.read().decode("utf-8", errors="ignore")
                # Treat blank-line separated blocks as separate articles;
                # otherwise the whole file is a single article.
                blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
                articles = blocks if len(blocks) > 1 else [content.strip()]
            else:
                error = "Unsupported file type. Please upload a .txt or .csv file."
        elif pasted_text:
            articles = [pasted_text]
        else:
            error = "Please paste an article or upload a .txt/.csv file."
    except Exception as exc:  # pragma: no cover
        error = f"Could not read the uploaded file: {exc}"

    articles = [a for a in articles if a and len(a.strip()) > 0]

    results = []
    if not error and not articles:
        error = "No article text was found in the input."

    if not error:
        for article in articles:
            results.append(generate_for_article(article, beam_size, top_k))

    return render_template(
        "index.html",
        results=results,
        error=error,
        beam_size=beam_size,
        top_k=top_k,
        pasted_text=pasted_text,
    )


@app.route("/health")
def health():
    try:
        get_generator()
        return {"status": "ok", "model_dir": MODEL_DIR}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}, 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
