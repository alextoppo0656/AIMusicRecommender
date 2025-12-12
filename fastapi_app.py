from dotenv import load_dotenv
load_dotenv()

import os
import time
import random
import sqlite3
import pandas as pd
import requests
import logging
import unicodedata
import hashlib
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator, Field
from cachetools import TTLCache
import spotipy
from spotipy.exceptions import SpotifyException
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# ------------------ LOGGING ------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------ CONFIG ------------------
class Config:
    """Centralized configuration with validation"""
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    DB_PATH = os.path.join(DATA_DIR, "music.db")
    
    SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
    SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
    SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8501/callback")
    
    LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")
    LASTFM_API_SECRET = os.getenv("LASTFM_API_SECRET")
    
    MAX_EXPAND_PER_CALL = 30 # Cap for new Last.fm tracks per expansion run
    CACHE_TTL = 3600
    CACHE_MAX_SIZE = 1000
    
    LASTFM_RATE_LIMIT_CALLS = 5
    LASTFM_RATE_LIMIT_PERIOD = 1
    
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")
    
    @classmethod
    def validate(cls):
        required = {
            "SPOTIFY_CLIENT_ID": cls.SPOTIFY_CLIENT_ID,
            "SPOTIFY_CLIENT_SECRET": cls.SPOTIFY_CLIENT_SECRET,
            "LASTFM_API_KEY": cls.LASTFM_API_KEY,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            logger.warning(f"Missing environment variables: {', '.join(missing)}")
        else:
            logger.info("✅ Configuration validated successfully")

try:
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    Config.validate()
except Exception as e:
    logger.error(f"Configuration error: {e}")

# ------------------ FASTAPI APP ------------------
app = FastAPI(title="AI Music Recommender Backend", version="2.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ SESSION TRACKING ------------------
active_sessions: Dict[str, Dict[str, Any]] = {}

def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()

def validate_session(user_id: str, access_token: str) -> bool:
    """
    Handles concurrent access by updating the session hash if a new token is 
    presented by the same user ID (ensuring the latest token is always accepted).
    """
    token_hash = hash_token(access_token)
    now = datetime.utcnow()
    
    if user_id in active_sessions:
        stored_hash = active_sessions[user_id].get("token_hash")
        
        if stored_hash != token_hash:
            logger.warning(f"Token hash mismatch for user {user_id}. Overwriting session with new token.")
            active_sessions[user_id]["token_hash"] = token_hash
            active_sessions[user_id]["access_token_prefix"] = access_token[:10]
        
        active_sessions[user_id]["last_seen"] = now
        return True
    
    # Register new session
    active_sessions[user_id] = {
        "token_hash": token_hash,
        "last_seen": now,
        "access_token_prefix": access_token[:10]
    }
    logger.info(f"New session registered for user: {user_id}")
    return True

def cleanup_old_sessions():
    """Remove sessions older than 2 hours"""
    cutoff = datetime.utcnow() - timedelta(hours=2)
    to_remove = [
        uid for uid, session in active_sessions.items()
        if session.get("last_seen", datetime.min) < cutoff
    ]
    for uid in to_remove:
        del active_sessions[uid]
    if to_remove:
        logger.info(f"Cleaned up {len(to_remove)} old sessions")

# ------------------ RATE LIMITER ------------------
class RateLimiter:
    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self.timestamps: Dict[str, List[float]] = defaultdict(list)
    
    def is_allowed(self, key: str) -> bool:
        now = time.time()
        self.timestamps[key] = [t for t in self.timestamps[key] if now - t < self.period]
        
        if random.random() < 0.01: 
            cutoff = now - 3600
            keys_to_remove = [
                k for k, v in self.timestamps.items() 
                if not v or (v and max(v) < cutoff)
            ]
            for k in keys_to_remove:
                del self.timestamps[k]
        
        if len(self.timestamps[key]) < self.calls:
            self.timestamps[key].append(now)
            return True
        return False
    
    def wait_time(self, key: str) -> float:
        if not self.timestamps[key]:
            return 0.0
        oldest = min(self.timestamps[key])
        wait = self.period - (time.time() - oldest)
        return max(0.0, wait)

lastfm_limiter = RateLimiter(Config.LASTFM_RATE_LIMIT_CALLS, Config.LASTFM_RATE_LIMIT_PERIOD)

# ------------------ DATABASE ------------------
@contextmanager
def get_db():
    conn = sqlite3.connect(Config.DB_PATH, check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}", exc_info=True)
        raise
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        c = conn.cursor()
        
        c.execute("""
        CREATE TABLE IF NOT EXISTS expanded (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            track_name TEXT NOT NULL,
            artist TEXT NOT NULL,
            album TEXT,
            year TEXT,
            source TEXT,
            tags TEXT,
            artist_seed TEXT,
            genre_seed TEXT,
            spotify_id TEXT,
            album_image TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        c.execute("""
        CREATE TABLE IF NOT EXISTS liked (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            track_name TEXT NOT NULL,
            artist TEXT NOT NULL,
            album TEXT,
            year TEXT,
            spotify_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Ensure feedback table has all necessary fields
        c.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            track_name TEXT NOT NULL,
            artist TEXT NOT NULL,
            liked INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            album TEXT,
            year TEXT,
            spotify_id TEXT
        )
        """)
        
        c.execute("""
        CREATE TABLE IF NOT EXISTS user_sessions (
            user_id TEXT PRIMARY KEY,
            token_hash TEXT NOT NULL,
            last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            login_count INTEGER DEFAULT 1
        )
        """)
        
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_expanded_user_track_artist ON expanded(user_id, track_name, artist)")
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_liked_user_track_artist ON liked(user_id, track_name, artist)")
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_user_track_artist ON feedback(user_id, track_name, artist)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_expanded_user ON expanded(user_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_liked_user ON liked(user_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id)")
        
        logger.info("✅ Database initialized successfully")

init_db()

# ------------------ MODELS ------------------
class TokenPayload(BaseModel):
    access_token: str
    
    @validator('access_token')
    def validate_token(cls, v):
        if not v or not v.strip():
            raise ValueError("access_token cannot be empty")
        if len(v) > 500:
            raise ValueError("access_token too long")
        return v.strip()

class FeedbackPayload(BaseModel):
    access_token: str = Field(..., min_length=1, max_length=500)
    track_name: str = Field(..., min_length=1, max_length=500)
    artist: str = Field(..., min_length=1, max_length=300)
    liked: bool
    album: Optional[str] = Field(None, max_length=500)
    year: Optional[str] = Field(None, max_length=4)
    spotify_id: Optional[str] = Field(None, max_length=100)
    
    @validator('track_name', 'artist', 'access_token')
    def validate_required_fields(cls, v):
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()
    
    @validator('year')
    def validate_year(cls, v):
        if v and v.strip():
            v = v.strip()
            if not v.isdigit() or len(v) != 4:
                return None
            year_int = int(v)
            if year_int < 1900 or year_int > datetime.now().year + 1:
                return None
            return v
        return None

# ------------------ UTILITIES ------------------
def normalize_string(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize('NFKC', s)
    return ' '.join(s.lower().strip().split())

def sanitize_string(s: str) -> str:
    """Remove potentially problematic characters"""
    if not s:
        return ""
    s = ''.join(char for char in s if char.isprintable() or char.isspace())
    return s.strip()

def spotify_user_id(access_token: str) -> str:
    if not access_token or not access_token.strip():
        raise HTTPException(status_code=401, detail="Missing access token")
    
    # 1. Get user ID from Spotify
    try:
        sp = spotipy.Spotify(auth=access_token)
        me = sp.current_user()
        user_id = me.get("id")
        if not user_id:
            raise HTTPException(status_code=500, detail="Unable to retrieve user ID from Spotify")
    except SpotifyException as e:
        logger.error(f"Spotify API error during user ID retrieval: {e}")
        if e.http_status in (401, 403):
            raise HTTPException(status_code=401, detail="Invalid or expired Spotify token")
        raise HTTPException(status_code=502, detail=f"Spotify API error: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting user ID: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")
    
    # 2. Validate and register session (CRITICAL FOR MULTI-USER)
    validate_session(user_id, access_token)
    
    return user_id

def get_all_liked_songs(access_token: str) -> List[Dict[str, Any]]:
    """FIX: Paginate to grab ALL liked songs (not just first 50)"""
    sp = spotipy.Spotify(auth=access_token)
    tracks = []
    
    try:
        # Initial call: offset starts at 0, limit is 50
        results = sp.current_user_saved_tracks(limit=50, offset=0)
        
        while True:
            for item in results.get("items", []):
                track = item.get("track")
                if not track or not track.get("name"):
                    continue
                
                album = track.get("album") or {}
                artists = track.get("artists") or []
                artist_name = artists[0]["name"] if artists else "Unknown Artist"
                
                track_name = sanitize_string(track.get("name", ""))
                artist_name = sanitize_string(artist_name)
                
                if not track_name or not artist_name:
                    continue
                
                artist_genres = []
                try:
                    if artists and artists[0].get("id"):
                        # Only fetch artist data if we have an ID
                        artist_data = sp.artist(artists[0]["id"])
                        artist_genres = [g.lower() for g in artist_data.get('genres', [])]
                except Exception:
                    pass
                
                tracks.append({
                    "track_name": track_name,
                    "artist": artist_name,
                    "album": sanitize_string(album.get("name", "")),
                    "year": (album.get("release_date") or "")[:4],
                    "spotify_id": track.get("id"),
                    "tags": ", ".join(artist_genres)
                })
            
            # Check if there are more pages
            if results.get("next"):
                results = sp.next(results)
            else:
                break # Stop loop when 'next' is None
        
        logger.info(f"Retrieved {len(tracks)} liked songs from Spotify")
        return tracks
        
    except SpotifyException as e:
        logger.error(f"Error fetching liked songs: {e}")
        raise HTTPException(status_code=502, detail=f"Spotify API error: {str(e)}")

def _safe_lastfm_get(url: str, max_retries: int = 3, backoff: float = 0.8) -> Dict[str, Any]:
    rate_key = "lastfm_global"
    if not lastfm_limiter.is_allowed(rate_key):
        wait = lastfm_limiter.wait_time(rate_key)
        time.sleep(wait)
    
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                retry_after = int(r.headers.get('Retry-After', backoff * (attempt + 1)))
                logger.warning(f"Last.fm rate limit, waiting {retry_after}s")
                time.sleep(retry_after)
                continue
            else:
                last_err = f"HTTP {r.status_code}"
        except requests.Timeout:
            last_err = "Request timeout"
        except Exception as e:
            last_err = str(e)
        
        if attempt < max_retries - 1:
            time.sleep(backoff * (attempt + 1))
    
    logger.warning(f"Last.fm request failed: {last_err}")
    return {}

def get_similar_artists(artist_name: str) -> List[str]:
    if not artist_name or not artist_name.strip():
        return []
    
    url = (f"http://ws.audioscrobbler.com/2.0/?method=artist.getsimilar"
           f"&artist={requests.utils.quote(artist_name)}"
           f"&api_key={Config.LASTFM_API_KEY}&format=json&limit=6")
    
    data = _safe_lastfm_get(url)
    artists = data.get("similarartists", {}).get("artist", [])
    return [sanitize_string(a.get("name")) for a in artists if a.get("name")]

def get_tracks_from_artist(artist_name: str, seed_artist: str) -> List[Dict[str, Any]]:
    if not artist_name or not artist_name.strip():
        return []
    
    url = (f"http://ws.audioscrobbler.com/2.0/?method=artist.gettoptracks"
           f"&artist={requests.utils.quote(artist_name)}"
           f"&api_key={Config.LASTFM_API_KEY}&format=json&limit=5")
    
    data = _safe_lastfm_get(url)
    tracks = data.get("toptracks", {}).get("track", [])
    
    out = []
    for t in tracks:
        name = sanitize_string(t.get("name", ""))
        if not name:
            continue
        
        out.append({
            "track_name": name,
            "artist": sanitize_string(artist_name),
            "album": None,
            "year": None,
            "source": "artist_similarity",
            "tags": None,
            "artist_seed": sanitize_string(seed_artist), 
            "genre_seed": None,
            "spotify_id": None,
            "album_image": None
        })
    
    return out

def get_artist_tags(artist_name: str) -> List[str]:
    if not artist_name or not artist_name.strip():
        return []
    
    url = (f"http://ws.audioscrobbler.com/2.0/?method=artist.gettoptags"
           f"&artist={requests.utils.quote(artist_name)}"
           f"&api_key={Config.LASTFM_API_KEY}&format=json&limit=5")
    
    data = _safe_lastfm_get(url)
    tags = data.get("toptags", {}).get("tag", [])
    return [sanitize_string(t.get("name")) for t in tags if t.get("name") and sanitize_string(t.get("name")) not in ["seen live", "fm", ""]]

def get_tracks_from_tag(tag: str, seed_tag: str) -> List[Dict[str, Any]]:
    if not tag or not tag.strip():
        return []
    
    url = (f"http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks"
           f"&tag={requests.utils.quote(tag)}"
           f"&api_key={Config.LASTFM_API_KEY}&format=json&limit=5")
    
    data = _safe_lastfm_get(url)
    tracks = data.get("tracks", {}).get("track", [])
    
    out = []
    for t in tracks:
        name = sanitize_string(t.get("name", ""))
        artist_obj = t.get("artist")
        artist_name = sanitize_string(artist_obj.get("name") if isinstance(artist_obj, dict) else "")
        
        if not name or not artist_name:
            continue
        
        out.append({
            "track_name": name,
            "artist": artist_name,
            "album": None,
            "year": None,
            "source": "genre_similarity",
            "tags": sanitize_string(tag),
            "artist_seed": None,
            "genre_seed": sanitize_string(seed_tag),
            "spotify_id": None,
            "album_image": None
        })
    
    return out

# ------------------ RECOMMENDATION CACHE ------------------
recommendation_cache: TTLCache = TTLCache(maxsize=Config.CACHE_MAX_SIZE, ttl=Config.CACHE_TTL)

# ------------------ MIDDLEWARE ------------------
@app.middleware("http")
async def cleanup_middleware(request: Request, call_next):
    """Periodically cleanup old sessions"""
    if random.random() < 0.01: 
        cleanup_old_sessions()
    response = await call_next(request)
    return response

# ------------------ ENDPOINTS ------------------
@app.get("/")
def root():
    return {
        "message": "🎧 AI Music Recommender Backend v2.3.1",
        "status": "running",
        "active_users": len(active_sessions),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
def health():
    try:
        with get_db() as conn:
            conn.execute("SELECT 1").fetchone()
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "ok" if db_status == "healthy" else "degraded",
        "database": db_status,
        "cache_size": len(recommendation_cache),
        "active_sessions": len(active_sessions),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/logout")
def logout(payload: TokenPayload):
    """Explicitly log out a user and invalidate their session"""
    try:
        user_id = None
        try:
            sp = spotipy.Spotify(auth=payload.access_token)
            me = sp.current_user()
            user_id = me.get("id")
        except Exception:
            pass
        
        if user_id in active_sessions:
            del active_sessions[user_id]
            logger.info(f"User logged out: {user_id}")
        
        recommendation_cache.pop(user_id, None)
        
        return {"status": "success", "message": "Logged out successfully"}
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return {"status": "success", "message": "Logged out"}

@app.post("/api/expand")
def expand_dataset(payload: TokenPayload):
    logger.info("Expand dataset request received")
    
    try:
        user_id = spotify_user_id(payload.access_token)
        logger.info(f"Expanding dataset for user: {user_id}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User ID retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")
    
    try:
        with get_db() as conn:
            c = conn.cursor()
            
            # --- 1. IMPORT LIKED SONGS AND FEEDBACK (PAGINATED) ---
            logger.info("Fetching ALL Spotify liked songs (paginated)...")
            spotify_liked = get_all_liked_songs(payload.access_token)
            
            # Insert Spotify liked songs into the 'liked' table
            for song in spotify_liked:
                try:
                    c.execute("""
                    INSERT INTO liked (user_id, track_name, artist, album, year, spotify_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, track_name, artist) DO UPDATE SET
                        album=COALESCE(excluded.album, album),
                        year=COALESCE(excluded.year, year),
                        spotify_id=COALESCE(excluded.spotify_id, spotify_id)
                    """, (user_id, song["track_name"], song["artist"], 
                          song["album"], song["year"], song["spotify_id"]))
                except sqlite3.Error as e:
                    logger.warning(f"Failed to insert liked song: {e}")
                    continue
            
            # Insert liked from explicit feedback into the 'liked' table
            feedback_liked = c.execute("""
                SELECT DISTINCT track_name, artist, album, year, spotify_id
                FROM feedback 
                WHERE user_id=? AND liked=1
            """, (user_id,)).fetchall()
            
            for fb in feedback_liked:
                try:
                    c.execute("""
                    INSERT INTO liked (user_id, track_name, artist, album, year, spotify_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, track_name, artist) DO UPDATE SET
                        album=COALESCE(excluded.album, album),
                        year=COALESCE(excluded.year, year),
                        spotify_id=COALESCE(excluded.spotify_id, spotify_id)
                    """, (user_id, fb["track_name"], fb["artist"], fb["album"], fb["year"], fb["spotify_id"]))
                except sqlite3.Error as e:
                    logger.warning(f"Failed to insert feedback song into liked: {e}")
                    continue
            
            conn.commit()
            
            # --- FIX: COPY ALL LIKED SONGS INTO THE EXPANDED DATASET ---
            liked_tracks = c.execute(
                "SELECT track_name, artist, album, year, spotify_id FROM liked WHERE user_id=?",
                (user_id,)
            ).fetchall()

            initial_count = c.execute("SELECT COUNT(*) FROM expanded WHERE user_id=?", (user_id,)).fetchone()[0]
            
            for track in liked_tracks:
                try:
                    c.execute("""
                    INSERT OR IGNORE INTO expanded 
                    (user_id, track_name, artist, album, year, source, tags, 
                     artist_seed, genre_seed, spotify_id, album_image)
                    VALUES (?, ?, ?, ?, ?, 'spotify_liked', NULL, NULL, NULL, ?, NULL)
                    """, (
                        user_id, 
                        track["track_name"], track["artist"], 
                        track["album"], track["year"], 
                        track["spotify_id"]
                    ))
                except sqlite3.Error as e:
                    logger.warning(f"Failed to insert liked track into expanded: {e}")
                    continue
            
            conn.commit()

            # --- 2. GET SEEDS AND EXISTING TRACKS (RE-QUERY AFTER INSERTS) ---
            liked_artists = [r[0] for r in c.execute(
                "SELECT DISTINCT artist FROM liked WHERE user_id=? AND artist IS NOT NULL AND artist != ''", 
                (user_id,)
            ).fetchall()]
            random.shuffle(liked_artists)
            
            existing_pairs = set()
            for r in c.execute(
                "SELECT track_name, artist FROM expanded WHERE user_id=?", 
                (user_id,)
            ).fetchall():
                normalized = (normalize_string(r["track_name"]), normalize_string(r["artist"]))
                existing_pairs.add(normalized)
            
            logger.info(f"Found {len(liked_artists)} artists as seeds. Existing expanded tracks (after liked import): {len(existing_pairs)}")
            
            expanded_tracks: List[Dict[str, Any]] = []
            
            # --- 3. LAST.FM EXPANSION LOGIC (CAPPED) ---
            
            for artist in liked_artists:
                
                # Check if we have gathered enough POTENTIAL tracks before making more API calls
                if len(expanded_tracks) >= Config.MAX_EXPAND_PER_CALL * 2:
                    break
                
                # 3a. Artist Similarity Expansion
                similar_artists = get_similar_artists(artist)
                for sim_artist in similar_artists[:5]:
                    if not sim_artist:
                        continue
                    for track in get_tracks_from_artist(sim_artist, artist):
                        normalized_key = (
                            normalize_string(track["track_name"]), 
                            normalize_string(track["artist"])
                        )
                        if normalized_key not in existing_pairs and normalized_key[0] and normalized_key[1]:
                            expanded_tracks.append(track)
                            existing_pairs.add(normalized_key)
                
                # 3b. Genre Similarity Expansion
                artist_tags = get_artist_tags(artist)
                for tag in artist_tags:
                    
                    for track in get_tracks_from_tag(tag, tag):
                        normalized_key = (
                            normalize_string(track["track_name"]), 
                            normalize_string(track["artist"])
                        )
                        if normalized_key not in existing_pairs and normalized_key[0] and normalized_key[1]:
                            expanded_tracks.append(track)
                            existing_pairs.add(normalized_key)
                
            
            random.shuffle(expanded_tracks)
            added_from_lastfm = 0
            
            # --- 4. INSERT EXPANDED TRACKS (FINAL CAP OF 30) ---
            for track in expanded_tracks:
                if added_from_lastfm >= Config.MAX_EXPAND_PER_CALL: # Apply final cap here (30)
                    break
                
                if not track.get("track_name") or not track.get("artist"):
                    continue
                
                try:
                    c.execute("""
                    INSERT OR IGNORE INTO expanded 
                    (user_id, track_name, artist, album, year, source, tags, 
                     artist_seed, genre_seed, spotify_id, album_image)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        user_id, track["track_name"], track["artist"], 
                        track["album"], track["year"], track["source"],
                        track["tags"], track["artist_seed"], track["genre_seed"], 
                        track["spotify_id"], track["album_image"]
                    ))
                    
                    if c.rowcount > 0:
                        added_from_lastfm += 1
                        
                except sqlite3.IntegrityError:
                    continue
                except sqlite3.Error as e:
                    logger.warning(f"Failed to insert track: {e}")
                    continue
            
            conn.commit()
            
            total = c.execute(
                "SELECT COUNT(*) FROM expanded WHERE user_id=?", 
                (user_id,)
            ).fetchone()[0]
            
            logger.info(f"✅ Added {added_from_lastfm} new tracks from Last.fm (capped at {Config.MAX_EXPAND_PER_CALL}), total: {total}")
            
            recommendation_cache.pop(user_id, None)
            
            return {
                "expanded_added": added_from_lastfm, 
                "total_rows": total,
                "status": "success"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Expand dataset failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Expansion failed: {str(e)}")

@app.post("/api/stats")
def stats(payload: TokenPayload):
    try:
        user_id = spotify_user_id(payload.access_token)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    try:
        with get_db() as conn:
            total_expanded = conn.execute(
                "SELECT COUNT(*) FROM expanded WHERE user_id=?", 
                (user_id,)
            ).fetchone()[0]
            
            total_liked = conn.execute(
                "SELECT COUNT(*) FROM liked WHERE user_id=?", 
                (user_id,)
            ).fetchone()[0]
            
            total_feedback = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE user_id=?", 
                (user_id,)
            ).fetchone()[0]
        
        return {
            "total_songs": total_expanded,
            "total_liked": total_liked,
            "total_feedback": total_feedback,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Stats query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommend")
def recommend(payload: TokenPayload):
    try:
        user_id = spotify_user_id(payload.access_token)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommend auth failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    try:
        with get_db() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM expanded WHERE user_id=?", 
                conn, 
                params=(user_id,)
            )
            fb = pd.read_sql_query(
                "SELECT * FROM feedback WHERE user_id=?", 
                conn, 
                params=(user_id,)
            )
            total_liked_songs = conn.execute(
                "SELECT COUNT(*) FROM liked WHERE user_id=?", 
                (user_id,)
            ).fetchone()[0]
        
        if df.empty:
            logger.info(f"No expanded tracks for user {user_id}")
            recommendation_cache[user_id] = {"ranked": [], "index": 0, "mode": "Random"}
            return {"recommendations": [], "mode": "Random", "status": "success"}
        
        df = df.replace([pd.NA, None, np.nan], "")
        
        # --- ML Mode Condition Check: Liked >= 50 ---
        if total_liked_songs < 50:
            logger.info(f"Using Random mode for user {user_id} (Total Liked: {total_liked_songs} < 50)")
            df_shuffled = df.sample(frac=1, random_state=None)
            ranked_list = df_shuffled.to_dict(orient="records")
            
            recommendation_cache[user_id] = {
                "ranked": ranked_list, 
                "index": 10, 
                "mode": "Random"
            }
            
            return {
                "recommendations": ranked_list[:10], 
                "mode": "Random",
                "status": "success"
            }
        
        logger.info(f"Using ML mode for user {user_id} (Total Liked: {total_liked_songs} >= 50)")
        
        # --- ML Training Prep ---
        df_merged = df.copy()
        # Merge all expanded tracks with feedback
        df_merged = df_merged.merge(fb[["track_name", "artist", "liked"]], on=["track_name", "artist"], how="left")
        
        # NOTE: Downcasting warning is inevitable here due to pandas typing when using fillna on objects.
        # This is fine for execution but will show the FutureWarning until you explicitly set the pandas option.
        df_merged["liked"] = df_merged["liked"].fillna(0)
        df_merged["liked"] = df_merged["liked"].astype(int)
        
        df_train = df_merged
        y = df_train["liked"]
        
        # Check if we have two classes (Liked=1 and Skipped/Un-feedbacked=0)
        if len(set(y)) < 2:
            logger.info("Only one class in feedback, falling back to Random mode")
            df_shuffled = df.sample(frac=1, random_state=None)
            ranked_list = df_shuffled.to_dict(orient="records")
            
            recommendation_cache[user_id] = {
                "ranked": ranked_list, 
                "index": 10, 
                "mode": "Random"
            }
            
            return {
                "recommendations": ranked_list[:10], 
                "mode": "Random",
                "status": "success"
            }
        
        # Feature Engineering (Artist and Tags)
        le_artist = LabelEncoder()
        df_train["artist_encoded"] = le_artist.fit_transform(df_train["artist"].astype(str))
        
        le_tags = LabelEncoder()
        df_train["tags"] = df_train["tags"].fillna("None")
        df_train["tags_encoded"] = le_tags.fit_transform(df_train["tags"].astype(str))
        
        X = df_train[["artist_encoded", "tags_encoded"]]
        
        # Model Training
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
        model.fit(X, y)
        
        logger.info(f"Trained ML model with {len(X)} samples using Artist and Tags features.")
        
        # --- Prediction ---
        def safe_encode(le: LabelEncoder, value: str, default_val: int) -> int:
            value = value if value else "None"
            if value in le.classes_:
                return int(le.transform([value])[0])
            return default_val
        
        df_predict = df.copy()
        
        default_artist_enc = df_train["artist_encoded"].mode()[0] if not df_train["artist_encoded"].empty else 0
        default_tag_enc = df_train["tags_encoded"].mode()[0] if not df_train["tags_encoded"].empty else 0

        df_predict["artist_encoded"] = df_predict["artist"].astype(str).apply(
            lambda x: safe_encode(le_artist, x, default_artist_enc)
        )
        df_predict["tags"] = df_predict["tags"].fillna("None")
        df_predict["tags_encoded"] = df_predict["tags"].astype(str).apply(
            lambda x: safe_encode(le_tags, x, default_tag_enc)
        )

        X_predict = df_predict[["artist_encoded", "tags_encoded"]]
        
        probs = model.predict_proba(X_predict)
        like_prob = probs[:, 1]
        df_predict["like_prob"] = like_prob
        
        # Filter out already-feedbacked songs
        feedbacked_keys = set(tuple(fb[["track_name", "artist"]].values))
        df_predict['key'] = df_predict.apply(lambda row: (row['track_name'], row['artist']), axis=1)
        df_predict = df_predict[~df_predict['key'].isin(feedbacked_keys)].drop(columns=['key'])

        df_ranked = df_predict.sort_values("like_prob", ascending=False)
        df_ranked = df_ranked.replace([pd.NA, None, np.nan], "")
        ranked_list = df_ranked.to_dict(orient="records")
        
        recommendation_cache[user_id] = {
            "ranked": ranked_list, 
            "index": 10, 
            "mode": "ML"
        }
        
        logger.info(f"✅ Generated {len(ranked_list)} ML recommendations.")
        
        return {
            "recommendations": ranked_list[:10], 
            "mode": "ML",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}", exc_info=True)
        # Fallback to random if ML fails
        try:
             df_shuffled = df.sample(frac=1, random_state=None)
             ranked_list = df_shuffled.to_dict(orient="records")
             recommendation_cache[user_id] = {"ranked": ranked_list, "index": 10, "mode": "Random"}
             return {"recommendations": ranked_list[:10], "mode": "Random", "status": "success"}
        except Exception:
             raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.post("/api/next_recommendations")
def next_recommendations(payload: TokenPayload):
    try:
        user_id = spotify_user_id(payload.access_token)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Next recommendations auth failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    try:
        cache = recommendation_cache.get(user_id)
        
        if not cache or not cache.get("ranked"):
            return recommend(payload)
        
        start = cache["index"]
        
        start_index = start
        end_index = start_index + 10
        total = len(cache["ranked"])
        
        if total == 0:
            return {"recommendations": [], "status": "success"}
        
        if start_index >= total:
            cache["index"] = 0
            start_index, end_index = 0, min(10, total)
        
        next_batch = cache["ranked"][start_index:end_index]
        cache["index"] = end_index if end_index < total else 0
        
        logger.info(f"Served batch {start_index}-{end_index} for user {user_id}")
        
        return {
            "recommendations": next_batch,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Next recommendations failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/api/previous_recommendations")
def previous_recommendations(payload: TokenPayload):
    try:
        user_id = spotify_user_id(payload.access_token)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Previous recommendations auth failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    try:
        cache = recommendation_cache.get(user_id)
        
        if not cache or not cache.get("ranked"):
            return recommend(payload)
        
        current_index = cache["index"]
        
        start_index = max(0, current_index - 20)
        end_index = max(10, current_index - 10)
        
        if end_index == 10 and current_index <= 10:
             start_index = 0
             end_index = 10

        previous_batch = cache["ranked"][start_index:end_index]
        cache["index"] = start_index + len(previous_batch)
        
        logger.info(f"Served previous batch {start_index}-{end_index} for user {user_id}")
        
        return {
            "recommendations": previous_batch,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Previous recommendations failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
def feedback(data: FeedbackPayload):
    try:
        user_id = spotify_user_id(data.access_token)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback auth failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    try:
        with get_db() as conn:
            c = conn.cursor()
            
            # --- 1. Update Feedback Table ---
            c.execute("""
            INSERT INTO feedback (user_id, track_name, artist, liked, timestamp, album, year, spotify_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, track_name, artist) DO UPDATE SET
                liked=excluded.liked,
                timestamp=excluded.timestamp,
                album=COALESCE(excluded.album, album),
                year=COALESCE(excluded.year, year),
                spotify_id=COALESCE(excluded.spotify_id, spotify_id)
            """, (
                user_id, 
                data.track_name, 
                data.artist, 
                int(data.liked), 
                datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                data.album,
                data.year,
                data.spotify_id
            ))
            
            # --- 2. Update/Insert into Liked Table (if Liked) ---
            if data.liked:
                c.execute("""
                INSERT INTO liked (user_id, track_name, artist, album, year, spotify_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, track_name, artist) DO UPDATE SET
                    album=COALESCE(excluded.album, album),
                    year=COALESCE(excluded.year, year),
                    spotify_id=COALESCE(excluded.spotify_id, spotify_id)
                """, (
                    user_id, 
                    data.track_name, 
                    data.artist, 
                    data.album, 
                    data.year, 
                    data.spotify_id
                ))
                
                # --- 3. Update Expanded Source (for ML/Tracking) ---
                # NOTE: We insert into expanded here if it was a user feedback like.
                c.execute("""
                INSERT OR IGNORE INTO expanded 
                (user_id, track_name, artist, album, year, source, tags, 
                 artist_seed, genre_seed, spotify_id, album_image)
                VALUES (?, ?, ?, ?, ?, 'user_liked', NULL, NULL, NULL, ?, NULL)
                """, (
                    user_id, 
                    data.track_name, data.artist, 
                    data.album, data.year, 
                    data.spotify_id
                ))
            
            conn.commit()
            
            logger.info(f"Feedback recorded for {user_id}: {data.track_name} - {'liked' if data.liked else 'skipped'}")
        
        recommendation_cache.pop(user_id, None)
        
        return {
            "status": "success",
            "message": f"Feedback recorded: {'liked' if data.liked else 'skipped'}"
        }
        
    except sqlite3.Error as e:
        logger.error(f"Database error in feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Feedback failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting AI Music Recommender Backend v2.3.1...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
