from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import random
import os

# -------------------- App Setup --------------------
app = Flask(__name__)
CORS(app)

# -------------------- Health / Home --------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "Mood Based Song Recommendation API",
        "endpoints": {
            "GET /api/moods": "Get available moods",
            "POST /api/recommend/mood": "Recommend songs by mood",
            "POST /api/recommend/song": "Recommend songs by song name"
        }
    })

# -------------------- Load Resources (SAFE PATHS) --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "mood_predictor_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "enhanced_song_dataset.csv")

model = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)
df = pd.read_csv(DATASET_PATH)

# -------------------- Preprocessing --------------------
df["mood"] = (
    df["mood"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.capitalize()
)

features = [
    "valence",
    "danceability",
    "energy",
    "tempo",
    "acousticness",
    "liveness"
]

available_moods = sorted(df["mood"].dropna().unique().tolist())

# -------------------- API: Get Moods --------------------
@app.route("/api/moods", methods=["GET"])
def get_available_moods():
    return jsonify({"available_moods": available_moods})

# -------------------- API: Recommend by Mood --------------------
@app.route("/api/recommend/mood", methods=["POST"])
def recommend_songs_by_mood():
    data = request.get_json(silent=True) or {}
    mood_input = data.get("mood", "").strip().capitalize()

    if not mood_input:
        return jsonify({"error": "Mood is required."}), 400

    if mood_input not in available_moods:
        return jsonify({"error": f"Mood '{mood_input}' not found."}), 404

    filtered_songs = df[df["mood"] == mood_input]

    if filtered_songs.empty:
        return jsonify({"error": f"No songs found for mood '{mood_input}'."}), 404

    count = min(random.randint(5, 8), len(filtered_songs))
    sampled_songs = filtered_songs.sample(count, random_state=42)

    return jsonify({
        "mood": mood_input,
        "songs": sampled_songs["song_name"].tolist()
    })

# -------------------- API: Recommend by Song --------------------
@app.route("/api/recommend/song", methods=["POST"])
def recommend_songs_by_song():
    data = request.get_json(silent=True) or {}
    song_input = data.get("song", "").strip()

    if not song_input:
        return jsonify({"error": "Song name is required."}), 400

    song_row = df[df["song_name"].str.lower() == song_input.lower()]
    if song_row.empty:
        return jsonify({"error": f"Song '{song_input}' not found."}), 404

    try:
        input_features = song_row[features].iloc[0].to_dict()
        input_df = pd.DataFrame([input_features], columns=features)

        predicted_mood_encoded = model.predict(input_df)[0]
        predicted_mood = (
            le.inverse_transform([predicted_mood_encoded])[0]
            .strip()
            .capitalize()
        )
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    mood_songs = df[
        (df["mood"] == predicted_mood) &
        (df["song_name"].str.lower() != song_input.lower())
    ]

    if mood_songs.empty:
        return jsonify({
            "error": f"No recommendations found for mood '{predicted_mood}'."
        }), 404

    count = min(random.randint(5, 8), len(mood_songs))
    recommendations = mood_songs.sample(count, random_state=42)

    return jsonify({
        "song_input": song_input,
        "predicted_mood": predicted_mood,
        "recommendations": recommendations["song_name"].tolist()
    })

# -------------------- Entry Point --------------------
if __name__ == "__main__":
    app.run()
