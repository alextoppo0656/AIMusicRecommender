# Music Recommender (Streamlit frontend + FastAPI backend)

## Overview
This small project uses your Spotify liked songs as seeds, expands them using Last.fm similar artists and top tracks, stores results in a CSV, and returns random recommendations. Streamlit provides the UI, FastAPI provides the backend.

## Files
- `fastapi_app.py` - FastAPI backend with `/api/expand` and `/api/recommend`
- `streamlit_app.py` - Streamlit frontend that handles Spotify OAuth and UI interactions
- `requirements.txt` - Python dependencies
- `data/expanded_dataset.csv` - generated dataset (created automatically)

## Setup (local)
1. Create a Python venv and install requirements:
```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

2. Edit `fastapi_app.py` and set your Last.fm API key/secret at top (replace placeholders). Also set Spotify client id/secret in the Streamlit file or as environment variables.

3. Run backend FastAPI:
```bash
uvicorn fastapi_app:app --reload --port 8000
```

4. Run Streamlit frontend (in a new terminal):
```bash
streamlit run streamlit_app.py
```

5. In Streamlit, click the Spotify login link, follow Spotify login, paste the redirect URL back into the text input. Then "Expand dataset" and "Get Recommendations".

## Notes & limitations
- This project expects you to run both backend and frontend locally.
- The Streamlit app uses the user's Spotify OAuth flow to obtain an access token and sends it to the backend. You can modify to exchange tokens in the backend if preferred.
- Playback is not implemented (embedding or preview), but Streamlit provides "Open in Spotify" search links for each recommended track.
- For production you'd want proper session handling, HTTPS redirect URIs, secure storage of API keys, and rate-limit handling.

Enjoy!