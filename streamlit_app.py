from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import time
import requests
import os
import logging
import hashlib
from typing import Optional, Dict, Any
from spotipy.oauth2 import SpotifyOAuth
import urllib.parse

# ------------------ CONFIGURATION ------------------
BACKEND_URL = os.getenv("BACKEND_URL")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")
SCOPE = "user-library-read"

if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
    st.error("⚠️ Missing Spotify credentials! Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables.")
    st.stop()

# ------------------ LOGGING ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Music Recommender",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------ CSS STYLING ------------------
st.markdown("""
<style>
@keyframes gradient {
  0% {background-position: 0% 50%;}
  50% {background-position: 100% 50%;}
  100% {background-position: 0% 50%;}
}

.stApp {
  background: linear-gradient(135deg, #1e3c72, #2a5298, #0f2027, #203a43, #2c5364);
  background-size: 400% 400%;
  animation: gradient 12s ease infinite;
  color: white;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

h1, h2, h3 {
  color: #1DB954;
  font-weight: 700;
}

.info-badge {
  margin-top: 8px;
  display: inline-block;
  padding: 8px 16px;
  border-radius: 10px;
  background: rgba(255,255,255,0.15);
  border: 1px solid rgba(255,255,255,0.25);
  color: #fff;
  font-size: 14px;
  font-weight: 600;
}

.stat-container {
  background: rgba(255,255,255,0.1);
  border-radius: 12px;
  padding: 15px;
  margin: 10px 0;
  backdrop-filter: blur(8px);
}

.mode-badge {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 20px;
  font-weight: 600;
  font-size: 13px;
}

.mode-ml {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.mode-random {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
  color: white;
}

.stButton > button {
  border-radius: 10px !important;
  font-weight: 600 !important;
  transition: all 0.3s ease !important;
}

.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
}

.stButton > button[kind="primary"] {
  background-color: #2ecc71 !important;
  border-color: #27ae60 !important;
  color: white !important;
}

.stButton > button[kind="primary"]:hover {
  background-color: #27ae60 !important;
}

.stButton > button[kind="secondary"] {
  background-color: #e74c3c !important;
  border-color: #c0392b !important;
  color: white !important;
}

.stButton > button[kind="secondary"]:hover {
  background-color: #c0392b !important;
}

.user-badge {
  background: rgba(29, 185, 84, 0.2);
  border: 1px solid rgba(29, 185, 84, 0.4);
  padding: 8px 16px;
  border-radius: 20px;
  display: inline-block;
  font-size: 14px;
  font-weight: 600;
  color: #1DB954;
}
</style>
""", unsafe_allow_html=True)

# ------------------ SESSION STATE INITIALIZATION ------------------
if "active_user_id" not in st.session_state:
    st.session_state["active_user_id"] = None

if "token_info" not in st.session_state:
    st.session_state["token_info"] = None

if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

if "user_display_name" not in st.session_state:
    st.session_state["user_display_name"] = None

if "session_id" not in st.session_state:
    st.session_state["session_id"] = hashlib.md5(str(time.time()).encode()).hexdigest()[:16]

if "recommendations" not in st.session_state:
    st.session_state["recommendations"] = []

if "feedback" not in st.session_state:
    st.session_state["feedback"] = {}

if "current_mode" not in st.session_state:
    st.session_state["current_mode"] = None

if "stats" not in st.session_state:
    st.session_state["stats"] = {}

# ------------------ UTILITY FUNCTIONS ------------------
def get_valid_token() -> Optional[str]:
    """Get valid access token, handling refresh if needed"""
    if not st.session_state["token_info"]:
        return None
    
    token_info = st.session_state["token_info"]
    
    sp_oauth = SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SCOPE,
        cache_path=None
    )
    
    if sp_oauth.is_token_expired(token_info):
        logger.info("Token expired, attempting refresh...")
        try:
            refresh_token = token_info.get('refresh_token')
            if not refresh_token:
                logger.error("No refresh token available")
                st.session_state.clear()
                st.error("🔒 Session expired. Please log in again.")
                st.rerun()
                return None
            
            token_info = sp_oauth.refresh_access_token(refresh_token)
            if not token_info:
                raise Exception("Token refresh returned None")
                
            st.session_state["token_info"] = token_info
            logger.info("✅ Token refreshed successfully")
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            st.session_state.clear()
            st.error("🔒 Session expired. Please log in again.")
            st.rerun()
            return None
    
    return token_info['access_token']

def call_backend(endpoint: str, payload: Dict[str, Any], timeout: int = 30, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """Make API call to backend with proper error handling and retries"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Calling backend: {endpoint} (attempt {attempt + 1}/{max_retries})")
            
            response = requests.post(
                f"{BACKEND_URL}{endpoint}",
                json=payload,
                timeout=timeout,
                headers={"Content-Type": "application/json"}
            )
            
            # Handle 401/403 (Invalid token or Backend Session Mismatch)
            if response.status_code in (401, 403):
                error_detail = response.json().get("detail", "Session expired or mismatch")
                st.error(f"🔒 Error: {error_detail}. Please log out and log in again.")
                st.session_state.clear()
                st.rerun()
                return None
            
            if response.status_code == 500:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Server error, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    error_data = response.json() if response.text else {}
                    error_msg = error_data.get("detail", "Internal server error")
                    st.error(f"❌ Server error: {error_msg}")
                    return None
            
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"✅ Backend call successful: {endpoint}")
            return data
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                logger.warning(f"Timeout, retrying...")
                time.sleep(2 ** attempt)
                continue
            st.error("⏱️ Request timed out after multiple attempts. Please try again.")
            return None
            
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                logger.warning(f"Connection error, retrying...")
                time.sleep(2 ** attempt)
                continue
            st.error("🔌 Cannot connect to backend server. Make sure it's running.")
            return None
            
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ HTTP Error {e.response.status_code}")
            return None
            
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Unexpected error, retrying: {e}")
                time.sleep(2 ** attempt)
                continue
            st.error(f"❌ Unexpected error: {str(e)}")
            return None
    
    return None

def validate_recommendation(rec: Dict[str, Any]) -> bool:
    """Validate recommendation has required fields"""
    required = ["track_name", "artist"]
    return all(rec.get(field) and str(rec.get(field)).strip() for field in required)

def sanitize_string(s: str, max_length: int = 200) -> str:
    """Sanitize and truncate user input strings"""
    if not s:
        return ""
    s = ' '.join(str(s).split())
    return s[:max_length]

def load_stats():
    """Load user statistics from backend"""
    access_token = get_valid_token()
    if not access_token:
        return
    
    data = call_backend("/api/stats", {"access_token": access_token}, timeout=10)
    if data and data.get("status") == "success":
        st.session_state["stats"] = data

def perform_logout():
    """Perform complete logout and aggressively clear state"""
    access_token = get_valid_token()
    
    if access_token:
        try:
            call_backend("/api/logout", {"access_token": access_token}, timeout=5)
        except Exception as e:
            logger.error(f"Logout API call failed (non-critical): {e}")
    
    logger.info(f"User logging out: {st.session_state.get('user_id', 'unknown')}")
    
    st.session_state.clear()
    
    # Add final client-side safety net (clears browser's local storage)
    st.markdown("""
        <script>
        localStorage.clear();
        sessionStorage.clear();
        </script>
    """, unsafe_allow_html=True)
    
    st.session_state["logged_out"] = True
    st.session_state["session_id"] = hashlib.md5(str(time.time()).encode()).hexdigest()[:16]
    
    st.query_params.clear()
    
    st.rerun()

# ------------------ HEADER ------------------
st.title("🎧 AI-Powered Music Recommender")
st.caption("Discover new tracks powered by Spotify + Last.fm + Machine Learning")

# ------------------ SPOTIFY AUTHENTICATION ------------------
st.header("1️⃣ Connect Your Spotify Account")

if st.session_state.get("logged_out"):
    st.info("👋 You've been logged out. Click the button below to log in with a different account.")
    st.session_state.pop("logged_out", None)

# Create OAuth handler with no caching
sp_oauth = SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope=SCOPE,
    open_browser=False,
    show_dialog=True
)

# Handle OAuth callback
query_params = st.query_params
auth_code = query_params.get("code")

if isinstance(auth_code, list):
    auth_code = auth_code[0] if auth_code else None

if not st.session_state["token_info"]:
    if auth_code:
        
        # AGGRESSIVE STATE CLEAR ON NEW LOGIN
        if st.session_state.get("user_id"):
            logger.warning(f"Stale user ID ({st.session_state['user_id']}) detected during new auth flow. Clearing all session state.")
            
            st.session_state.clear()
            st.session_state["session_id"] = hashlib.md5(str(time.time()).encode()).hexdigest()[:16]
            st.rerun() 
            
        with st.spinner("🔐 Authenticating with Spotify..."):
            try:
                token_info = sp_oauth.get_access_token(auth_code, as_dict=True)
                st.session_state["token_info"] = token_info
                
                access_token = token_info.get('access_token')
                if access_token:
                    import spotipy
                    sp = spotipy.Spotify(auth=access_token)
                    user_info = sp.current_user()
                    user_id = user_info.get('id')
                    user_display_name = user_info.get('display_name', user_id)
                    
                    st.session_state["user_id"] = user_id
                    st.session_state["user_display_name"] = user_display_name
                    
                    logger.info(f"✅ User authenticated: {user_id} (session: {st.session_state['session_id']})")
                    st.success(f"✅ Successfully logged in as **{user_display_name}**!")
                else:
                    raise Exception("No access token received")
                
                st.query_params.clear()
                time.sleep(0.5)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Authentication failed: {str(e)}")
                logger.error(f"Auth error: {e}")
                st.session_state.clear()
                st.rerun()
    else:
        auth_url = sp_oauth.get_authorize_url()
        
        # --- CRITICAL FIX: Ensure robust URL construction with all necessary parameters ---
        
        import urllib.parse
        parsed_url = urllib.parse.urlparse(auth_url)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        
        query_params['show_dialog'] = ['true']
        query_params['state'] = [st.session_state['session_id']]
        query_params['prompt'] = ['login'] # Force login screen for multi-user fix
        
        new_query = urllib.parse.urlencode(query_params, doseq=True)
        auth_url_with_params = parsed_url._replace(query=new_query).geturl()
        
        # ----------------------------------------------------------------------------------

        # FIX: Implement JavaScript redirect to avoid X-Frame-Options denial.
        js_redirect = f"window.top.location.href = '{auth_url_with_params}';"
        
        if st.button("🔗 Login with Spotify (Force New User)", type="primary", use_container_width=True):
            st.markdown(f"<script>{js_redirect}</script>", unsafe_allow_html=True)
        
        st.info("👆 Click above and enter your credentials to ensure a clean multi-user login.")
        st.warning("⚠️ **Multi-User Support**: Each login creates a new session. Your data is isolated from other users.")
        st.stop()

else:
    # User is logged in
    col_status, col_logout = st.columns([3, 1])
    
    with col_status:
        user_display = st.session_state.get("user_display_name", "Unknown User")
        user_id = st.session_state.get("user_id", "Unknown")
        
        st.markdown(f"""
        <div class='user-badge'>
            👤 {user_display}
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"User ID: {user_id} | Session: {st.session_state.get('session_id', 'N/A')[:8]}")
    
    with col_logout:
        if st.button("🚪 Logout", type="secondary", use_container_width=True):
            perform_logout()
            st.stop() 

st.markdown("---")

# Verify user info is loaded
if st.session_state.get("token_info") and not st.session_state.get("user_id"):
    try:
        access_token = get_valid_token()
        if access_token:
            import spotipy
            sp = spotipy.Spotify(auth=access_token)
            user_info = sp.current_user()
            st.session_state["user_id"] = user_info.get('id')
            st.session_state["user_display_name"] = user_info.get('display_name')
    except Exception as e:
        logger.error(f"Failed to get user info: {e}")
        st.session_state.clear()
        st.rerun()

# ------------------ DATASET EXPANSION ------------------
st.header("2️⃣ Expand Your Music Dataset")
st.caption("Fetch your liked songs from Spotify and discover similar tracks from Last.fm")

col_expand, col_stats = st.columns([1, 1])

with col_expand:
    if st.button("🚀 Expand Dataset", type="primary", use_container_width=True):
        access_token = get_valid_token()
        if not access_token:
            st.error("Please log in first.")
        else:
            with st.spinner("🔄 Expanding dataset... This may take a minute. Grabbing all liked songs in chunks."):
                data = call_backend(
                    "/api/expand",
                    {"access_token": access_token},
                    timeout=180
                )
                
                if data and data.get("status") == "success":
                    added = data.get("expanded_added", 0)
                    total = data.get("total_rows", 0)
                    st.success(f"✅ Added {added} new tracks! Total dataset: {total} tracks 🎶")
                    
                    load_stats()
                    st.rerun()

with col_stats:
    if not st.session_state["stats"]:
        load_stats()
    
    stats = st.session_state["stats"]
    if stats:
        st.markdown(f"""
        <div class='stat-container'>
            <div style='font-size: 14px; color: #aaa;'>Dataset Size</div>
            <div style='font-size: 28px; font-weight: 700; color: #1DB954;'>
                {stats.get('total_songs', 0):,}
            </div>
            <div style='font-size: 12px; color: #ccc; margin-top: 5px;'>
                📚 {stats.get('total_liked', 0)} liked | 
                💬 {stats.get('total_feedback', 0)} feedback
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='stat-container'>
            <div style='font-size: 14px; color: #aaa;'>Dataset Size</div>
            <div style='font-size: 28px; font-weight: 700; color: #888;'>---</div>
            <div style='font-size: 12px; color: #ccc;'>Click Expand to get started</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ------------------ RECOMMENDATIONS ------------------
st.header("3️⃣ Get Personalized Recommendations")

col_rec, col_prev, col_next = st.columns([1, 1, 1])

with col_rec:
    if st.button("🎵 Generate Recommendations", type="primary", use_container_width=True):
        access_token = get_valid_token()
        if not access_token:
            st.error("Please log in first.")
        else:
            with st.spinner("🔮 Generating personalized recommendations..."):
                data = call_backend(
                    "/api/recommend",
                    {"access_token": access_token},
                    timeout=30
                )
                
                if data and data.get("status") == "success":
                    recs = data.get("recommendations", [])
                    mode = data.get("mode", "Random")
                    
                    st.session_state["recommendations"] = recs
                    st.session_state["current_mode"] = mode
                    
                    if recs:
                        st.success(f"✅ Generated {len(recs)} recommendations!")
                        st.rerun()
                    else:
                        st.warning("No recommendations available. Try expanding your dataset first.")

with col_prev:
    if st.button("⏪ Previous 10 Songs", use_container_width=True, disabled=(st.session_state.get('recommendations') is None)):
        access_token = get_valid_token()
        if not access_token:
            st.error("Please log in first.")
        else:
            with st.spinner("Loading previous batch..."):
                data = call_backend(
                    "/api/previous_recommendations",
                    {"access_token": access_token},
                    timeout=20
                )
                
                if data and data.get("status") == "success":
                    prev_recs = data.get("recommendations", [])
                    
                    if prev_recs:
                        st.session_state["recommendations"] = prev_recs
                        st.success("✅ Loaded previous batch!")
                        st.rerun()
                    else:
                        st.info("⏮️ Reached the beginning of the list.")

with col_next:
    if st.button("⏭️ Next 10 Songs", use_container_width=True, disabled=(st.session_state.get('recommendations') is None)):
        access_token = get_valid_token()
        if not access_token:
            st.error("Please log in first.")
        else:
            with st.spinner("Loading next batch..."):
                data = call_backend(
                    "/api/next_recommendations",
                    {"access_token": access_token},
                    timeout=20
                )
                
                if data and data.get("status") == "success":
                    next_recs = data.get("recommendations", [])
                    
                    if next_recs:
                        st.session_state["recommendations"] = next_recs
                        st.success("✅ Loaded next batch!")
                        st.rerun()
                    else:
                        st.info("🔄 No more recommendations - looping to start")

# Display current mode
if st.session_state["current_mode"]:
    mode = st.session_state["current_mode"]
    badge_class = "mode-ml" if mode == "ML" else "mode-random"
    icon = "🧠" if mode == "ML" else "🎲"
    
    st.markdown(f"""
    <div style='margin: 15px 0;'>
        <span class='mode-badge {badge_class}'>
            {icon} {mode} Mode Active
        </span>
        <span style='color: #aaa; font-size: 13px; margin-left: 10px;'>
            {'ML-powered predictions based on your feedback' if mode == 'ML' else 'Random selection from your dataset'}
        </span>
    </div>
    """, unsafe_allow_html=True)

# ------------------ DISPLAY RECOMMENDATIONS ------------------
if st.session_state["recommendations"]:
    st.markdown("### 🎼 Your Recommendations")
    
    valid_recs = 0
    for i, rec in enumerate(st.session_state["recommendations"]):
        if not validate_recommendation(rec):
            logger.warning(f"Skipping invalid recommendation at index {i}")
            continue
        
        try:
            track = sanitize_string(rec.get("track_name", "Unknown"))
            artist = sanitize_string(rec.get("artist", "Unknown"))
            album = sanitize_string(rec.get("album", ""))
            year = sanitize_string(rec.get("year", ""), max_length=4)
            source = sanitize_string(rec.get("source", ""))
            
            if not track or track == "Unknown" or not artist or artist == "Unknown":
                continue
            
            key = f"{track}-{artist}"
            status = st.session_state["feedback"].get(key)
            
            search_query = f"{track} {artist}"
            spotify_url = f"https://open.spotify.com/search/{requests.utils.quote(search_query)}"
            
            status_indicator = ""
            if status == "liked":
                status_indicator = "💚 LIKED"
            elif status == "skipped":
                status_indicator = "💔 SKIPPED"
            
            is_expanded = (status != "skipped")
            
            valid_recs += 1
            
            with st.expander(
                f"**{valid_recs}. {track}** by {artist} {('• ' + status_indicator) if status_indicator else ''}", 
                expanded=is_expanded
            ):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if album:
                        st.write(f"💿 **Album:** {album}")
                    if year:
                        st.write(f"📅 **Year:** {year}")
                    if source:
                        st.write(f"🔍 **Source:** {source}")
                    
                    st.markdown(f"[🎧 Open in Spotify →]({spotify_url})")
                
                with col2:
                    if st.button("👍", key=f"like_{i}", use_container_width=True, help="Like this track", type="primary"):
                        access_token = get_valid_token()
                        if access_token:
                            with st.spinner("Saving..."):
                                st.session_state["feedback"][key] = "liked"
                                
                                feedback_data = {
                                    "access_token": access_token,
                                    "track_name": track,
                                    "artist": artist,
                                    "liked": True,
                                    "album": album if album else None,
                                    "year": year if year else None,
                                    "spotify_id": rec.get("spotify_id")
                                }
                                
                                result = call_backend("/api/feedback", feedback_data, timeout=10)
                                
                                if result and result.get("status") == "success":
                                    st.toast(f"💚 Liked: {track}", icon="✅")
                                    time.sleep(0.5)
                                    st.rerun()
                                else:
                                    st.error("Failed to save feedback.")
                    
                    if st.button("👎", key=f"skip_{i}", use_container_width=True, help="Skip this track", type="secondary"):
                        access_token = get_valid_token()
                        if access_token:
                            with st.spinner("Saving..."):
                                st.session_state["feedback"][key] = "skipped"
                                
                                feedback_data = {
                                    "access_token": access_token,
                                    "track_name": track,
                                    "artist": artist,
                                    "liked": False
                                }
                                
                                result = call_backend("/api/feedback", feedback_data, timeout=10)
                                
                                if result and result.get("status") == "success":
                                    st.toast(f"💔 Skipped: {track}", icon="⏭️")
                                    time.sleep(0.5)
                                    st.rerun()
                                else:
                                    st.error("Failed to save feedback.")
        
        except Exception as e:
            logger.error(f"Error displaying recommendation {i}: {e}", exc_info=True)
            continue
    
    if valid_recs == 0:
        st.warning("⚠️ No valid recommendations to display. Try generating new ones.")

else:
    st.info("👆 Click 'Generate Recommendations' to get started!")

# ------------------ FOOTER ------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.caption("💚 Powered by Streamlit + FastAPI + SQLite + Spotify + Last.fm | Built with ❤️")

# Add helpful tips in sidebar
with st.sidebar:
    st.header("👤 Current User")
    if st.session_state.get("user_id"):
        st.markdown(f"""
        <div style='background: rgba(29, 185, 84, 0.1); padding: 10px; border-radius: 8px; border: 1px solid rgba(29, 185, 84, 0.3);'>
            <strong style='color: #1DB954;'>{st.session_state.get('user_display_name', 'User')}</strong><br>
            <small style='color: #aaa;'>ID: {st.session_state.get('user_id', 'N/A')}</small><br>
            <small style='color: #aaa;'>Session: {st.session_state.get('session_id', 'N/A')[:8]}</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Not logged in")
    
    st.markdown("---")
    st.header("💡 Tips")
    st.markdown("""
    **Getting Started:**
    1. Connect your Spotify account
    2. Expand your dataset (imports liked songs)
    3. Generate recommendations
    
    **Understanding Modes:**
    - 🎲 **Random**: < 50 liked songs
    - 🧠 **ML Mode**: $\ge 50$ liked songs + enough data
    
    **Pro Tips:**
    - Like/skip songs to train ML model
    - More feedback = better recommendations
    - Expand multiple times for variety
    - Each user has isolated data
    """)
    
    st.markdown("---")
    
    st.header("📊 Session Info")
    if st.session_state["recommendations"]:
        st.metric("Current Recommendations", len(st.session_state["recommendations"]))
    
    feedback_count = len(st.session_state["feedback"])
    if feedback_count > 0:
        liked_count = sum(1 for v in st.session_state["feedback"].values() if v == "liked")
        skipped_count = sum(1 for v in st.session_state["feedback"].values() if v == "skipped")
        st.metric("Total Feedback", feedback_count)
        st.write(f"👍 Liked: {liked_count} | 👎 Skipped: {skipped_count}")
    
    if st.button("🔄 Refresh Stats", use_container_width=True):
        load_stats()
        st.rerun()
    
    if st.button("🗑️ Clear Session Data", use_container_width=True):
        st.session_state["recommendations"] = []
        st.session_state["feedback"] = {}
        st.session_state["current_mode"] = None
        st.success("Session data cleared!")
        st.rerun()