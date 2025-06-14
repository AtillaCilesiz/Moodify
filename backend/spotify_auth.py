import os
import base64
import requests
from flask import Flask, request, redirect, session, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from dataset_utils import initialize_dataset, get_audio_features_for_track
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Ortam Değişkenlerini Yükle ---
load_dotenv()

# --- Flask Uygulaması Başlat ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
app.secret_key = os.getenv("FLASK_SECRET_KEY", "defaultsecretkey")

# --- Spotify API Bilgileri ---
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")

# Eksik Değişken Kontrolü
if not all([SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI]):
    raise EnvironmentError("Spotify Client ID, Secret veya Redirect URI eksik!")

SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE_URL = "https://api.spotify.com/v1"
SCOPE = (
    "user-top-read "
    "user-read-recently-played "
    "user-library-read "
    "playlist-read-private "
    "user-read-private"
)

# --- Dataset Yükle ---
initialize_dataset()

# --- Yardımcı Fonksiyonlar ---
# --- 3) NLP ile duygu analizi için pipeline ---
sentiment_analyzer = pipeline("sentiment-analysis")

def detect_artist(text):
    """
    Metindeki sanatçı adını bulmaya çalışır.
    Bulursa artist objesini, yoksa None döner.
    """
    # Spotify Search API çağrısı
    res = requests.get(
        f"{SPOTIFY_API_BASE_URL}/search",
        headers=get_spotify_headers(),
        params={"q": text, "type": "artist", "limit": 1}
    ).json()
    items = res.get("artists", {}).get("items", [])
    if items and items[0]["name"].lower() in text.lower():
        return items[0]
    return None

def analyze_mood(text):
    """
    Metni NLP ile pozitif/negatif ayrımına sokar.
    Pozitifse 'energetic', negatifse 'sad' döner.
    """
    r = sentiment_analyzer(text)[0]
    return "energetic" if r["label"] == "POSITIVE" else "sad"

def recommend_by_mood(mood):
    """
    Mood'a göre Spotify Recommendations endpoint'ini kullanır.
    """
    if mood == "energetic":
        recs = requests.get(
            f"{SPOTIFY_API_BASE_URL}/recommendations",
            headers=get_spotify_headers(),
            params={
                "seed_genres": "pop,dance",
                "target_energy": 0.8,
                "target_valence": 0.9,
                "limit": 50
            }
        ).json()
    else:
        recs = requests.get(
            f"{SPOTIFY_API_BASE_URL}/recommendations",
            headers=get_spotify_headers(),
            params={
                "seed_genres": "acoustic,sad",
                "target_energy": 0.3,
                "target_valence": 0.2,
                "limit": 50
            }
        ).json()
    return recs.get("tracks", [])


def get_spotify_headers():
    access_token = session.get('access_token')
    if not access_token:
        return None
    return {"Authorization": f"Bearer {access_token}"}

def handle_unauthorized():
    return jsonify({"error": "Unauthorized", "message": "Please login first"}), 401

def simplify_track(track):
    return {
        "id": track.get("id"),
        "name": track.get("name"),
        "artists": [artist['name'] for artist in track.get("artists", [])],
        "album": track.get("album", {}).get("name"),
        "popularity": track.get("popularity")
    }

def simplify_artist(artist):
    return {
        "id": artist.get("id"),
        "name": artist.get("name"),
        "genres": artist.get("genres"),
        "popularity": artist.get("popularity")
    }

# --- Auth Routes ---

@app.route('/login')
def login():
    auth_url = (
        f"{SPOTIFY_AUTH_URL}"
        f"?client_id={SPOTIFY_CLIENT_ID}"
        f"&response_type=code"
        f"&redirect_uri={SPOTIFY_REDIRECT_URI}"
        f"&scope={SCOPE}"
    )
    return jsonify({"auth_url": auth_url})

@app.route('/callback')
def callback():
    code = request.args.get('code')
    if not code:
        return jsonify({"error": "No code provided"}), 400

    auth_header = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
    headers = {"Authorization": f"Basic {auth_header}"}
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": SPOTIFY_REDIRECT_URI
    }

    response = requests.post(SPOTIFY_TOKEN_URL, data=payload, headers=headers)
    if response.status_code != 200:
        return jsonify({"error": "Failed to get access token"}), response.status_code

    response_data = response.json()
    session['access_token'] = response_data.get('access_token')
    session['refresh_token'] = response_data.get('refresh_token')

    # Redirect to frontend home page
    return redirect('http://localhost:3000')

@app.route('/refresh_token')
def refresh_token():
    refresh_token = session.get('refresh_token')
    if not refresh_token:
        return jsonify({"error": "No refresh token found"}), 400

    auth_header = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
    headers = {"Authorization": f"Basic {auth_header}"}
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token
    }

    response = requests.post(SPOTIFY_TOKEN_URL, data=payload, headers=headers)
    response_data = response.json()

    session['access_token'] = response_data.get('access_token')

    return jsonify({"status": "Access token refreshed"})

# --- Spotify Data Routes ---

@app.route('/profile')
def profile():
    headers = get_spotify_headers()
    if headers is None:
        return redirect('/login')
    response = requests.get(f"{SPOTIFY_API_BASE_URL}/me", headers=headers)
    return jsonify(response.json())

@app.route('/top_tracks')
def top_tracks():
    headers = get_spotify_headers()
    if headers is None:
        return handle_unauthorized()
    
    spotify_response = requests.get(
        f"{SPOTIFY_API_BASE_URL}/me/top/tracks?limit=50",
        headers=headers
    )
    if spotify_response.status_code != 200:
        return jsonify({"error": "Spotify API Error", "details": spotify_response.text}), spotify_response.status_code

    data = spotify_response.json()
    simplified = [simplify_track(track) for track in data.get('items', [])]
    return jsonify(simplified)

@app.route('/top_artists')
def top_artists():
    headers = get_spotify_headers()
    if headers is None:
        return handle_unauthorized()
    
    spotify_response = requests.get(
        f"{SPOTIFY_API_BASE_URL}/me/top/artists?limit=50",
        headers=headers
    )
    if spotify_response.status_code != 200:
        return jsonify({"error": "Spotify API Error", "details": spotify_response.text}), spotify_response.status_code

    data = spotify_response.json()
    simplified = [simplify_artist(artist) for artist in data.get("items", [])]
    return jsonify(simplified)

@app.route('/recently_played')
def recently_played():
    headers = get_spotify_headers()
    if headers is None:
        return handle_unauthorized()
    
    spotify_response = requests.get(
        f"{SPOTIFY_API_BASE_URL}/me/player/recently-played?limit=50",
        headers=headers
    )
    if spotify_response.status_code != 200:
        return jsonify({"error": "Spotify API Error", "details": spotify_response.text}), spotify_response.status_code

    data = spotify_response.json()
    simplified = [simplify_track(item["track"]) for item in data.get("items", [])]
    return jsonify(simplified)

@app.route('/saved_tracks')
def saved_tracks():
    headers = get_spotify_headers()
    if headers is None:
        return handle_unauthorized()
    
    spotify_response = requests.get(
        f"{SPOTIFY_API_BASE_URL}/me/tracks?limit=50",
        headers=headers
    )
    if spotify_response.status_code != 200:
        return jsonify({"error": "Spotify API Error", "details": spotify_response.text}), spotify_response.status_code

    data = spotify_response.json()
    simplified = [simplify_track(item["track"]) for item in data.get("items", [])]
    return jsonify(simplified)

@app.route('/playlists')
def playlists():
    headers = get_spotify_headers()
    if headers is None:
        return redirect('/login')
    response = requests.get(f"{SPOTIFY_API_BASE_URL}/me/playlists", headers=headers)
    return jsonify(response.json())

# --- Audio Features/Analysis ---

@app.route('/audio_features/<track_id>')
def audio_features(track_id):
    headers = get_spotify_headers()
    if headers is None:
        return redirect('/login')
    response = requests.get(f"{SPOTIFY_API_BASE_URL}/audio-features/{track_id}", headers=headers)
    return jsonify(response.json())

@app.route('/audio_analysis/<track_id>')
def audio_analysis(track_id):
    headers = get_spotify_headers()
    if headers is None:
        return redirect('/login')
    response = requests.get(f"{SPOTIFY_API_BASE_URL}/audio-analysis/{track_id}", headers=headers)
    return jsonify(response.json())

@app.route('/recommend', methods=['POST'])
def recommend():
    text = request.json.get('text', '')
    # 1) Önce sanatçı isteği var mı?
    artist = detect_artist(text)
    if artist:
        # artist radio önerisi
        recs = requests.get(
            f"{SPOTIFY_API_BASE_URL}/recommendations",
            headers=get_spotify_headers(),
            params={"seed_artists": artist["id"], "limit": 50}
        ).json()
        return jsonify({
            "source": "artist_radio",
            "artist": artist["name"],
            "tracks": recs.get("tracks", [])
        })

    # 2) Sanatçı değilse duyguya (mood) göre öner
    mood = analyze_mood(text)
    tracks = recommend_by_mood(mood)
    return jsonify({
        "source": "mood",
        "mood": mood,
        "tracks": tracks
    })


# --- Frontend'e Özel Uyumlu Routes ---

@app.route('/get_top_tracks')
def get_top_tracks_new():
    return top_tracks()

@app.route('/get_liked_songs')
def get_liked_songs_new():
    return saved_tracks()

@app.route('/get_top_artists')
def get_top_artists_new():
    return top_artists()

@app.route('/get_recently_played')
def get_recently_played_new():
    return recently_played()

# --- App Runner ---

if __name__ == "__main__":
    app.run(debug=True)
