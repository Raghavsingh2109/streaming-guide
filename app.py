# importing all the stuff i need for this project
import streamlit as st
import pandas as pd
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

st.set_page_config(page_title="StreamIQ", page_icon="🎬", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    background-color: #0a0a0f !important;
    color: #e8e0d0 !important;
    font-family: 'Playfair Display', Georgia, serif !important;
}
.stApp { background-color: #0a0a0f !important; }
h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: #00d4d4 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
.stTextInput > div > div > input {
    background-color: #12121a !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 8px !important;
    color: #e8e0d0 !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 15px !important;
    padding: 0.85rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #00d4d4 !important;
    box-shadow: none !important;
}
.stButton > button {
    background-color: #00d4d4 !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    width: 100% !important;
    padding: 0.75rem !important;
    margin-top: 0.5rem !important;
}
.stButton > button:hover { background-color: #00b3b3 !important; }
.stSelectbox > div > div {
    background-color: #12121a !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 8px !important;
    color: #e8e0d0 !important;
}
.stSlider > div > div > div > div { background-color: #00d4d4 !important; }
.stTabs [data-baseweb="tab-list"] {
    background-color: #0d0d14 !important;
    border-bottom: 1px solid #2a2a3a !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #666 !important;
}
.stTabs [aria-selected="true"] {
    color: #00d4d4 !important;
    border-bottom: 2px solid #00d4d4 !important;
}
.stSuccess {
    background-color: #12121a !important;
    border: 1px solid #1a3a3a !important;
    border-left: 3px solid #00d4d4 !important;
    border-radius: 8px !important;
}
.stSpinner > div { border-top-color: #00d4d4 !important; }
.stToolbar { visibility: hidden; }
[data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stDecoration"] { visibility: hidden; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='padding:1.5rem 0 1rem; border-bottom:0.5px solid #2a2a3a; margin-bottom:1.5rem;'>
    <div style='font-family:Playfair Display,serif; font-size:2.2rem; color:#00d4d4; font-weight:700; letter-spacing:-0.5px;'>
        Stream<span style='color:#e8e0d0; font-weight:300;'>IQ</span>
    </div>
    <div style='font-family:Space Mono,monospace; font-size:11px; color:#555; letter-spacing:2px; text-transform:uppercase; margin-top:6px;'>
        India's smart streaming guide
    </div>
</div>
""", unsafe_allow_html=True)

if not GROQ_API_KEY:
    st.error("Groq API key not found. Please set it in your environment secrets.")
    st.stop()

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'genre_result' not in st.session_state:
    st.session_state.genre_result = None
if 'genre_matches' not in st.session_state:
    st.session_state.genre_matches = None
if 'rag_result' not in st.session_state:
    st.session_state.rag_result = None
if 'rag_matches' not in st.session_state:
    st.session_state.rag_matches = None

MOOD_MAP = {
    "Any mood": "",
    "Intense & gripping": "crime thriller drama",
    "Light & fun": "comedy romance",
    "Feel good": "family animation feel-good",
    "Romantic": "romance love",
    "Scary": "horror supernatural",
    "Action packed": "action adventure",
    "Thought provoking": "documentary biography drama"
}

# platform colors and badges
PLATFORM_COLORS = {
    "Netflix": "#E50914",
    "Prime Video": "#00A8E0",
    "Hotstar": "#1F80E0"
}

PLATFORM_BADGES = {
    "Netflix": "🔴 Netflix",
    "Prime Video": "🔵 Prime Video",
    "Hotstar": "🟣 Hotstar"
}

@st.cache_data
def load_data():
    try:
        prime = pd.read_csv("primevideo_india_movies_and_shows.csv")
        netflix = pd.read_csv("netflix_india_shows_and_movies.csv")
        hotstar = pd.read_csv("hotstar.csv")

        prime_clean = prime[['name', 'type', 'genre', 'release_year', 'synopsis', 'imdb_rating']].copy()
        prime_clean.columns = ['title', 'type', 'genre', 'release_year', 'description', 'imdb_rating']
        prime_clean['platform'] = 'Prime Video'

        netflix_clean = netflix[['name', 'type', 'genre', 'release_year', 'description']].copy()
        netflix_clean.columns = ['title', 'type', 'genre', 'release_year', 'description']
        netflix_clean['imdb_rating'] = None
        netflix_clean['platform'] = 'Netflix'

        hotstar_clean = hotstar[['title', 'type', 'genre', 'year', 'description']].copy()
        hotstar_clean.columns = ['title', 'type', 'genre', 'release_year', 'description']
        hotstar_clean['imdb_rating'] = None
        hotstar_clean['platform'] = 'Hotstar'

        combined = pd.concat([prime_clean, netflix_clean, hotstar_clean], ignore_index=True)
        combined = combined.dropna(subset=['title', 'genre'])
        combined['description'] = combined['description'].fillna('')
        combined['imdb_rating'] = pd.to_numeric(combined['imdb_rating'], errors='coerce')
        combined['search_text'] = combined['title'] + ' ' + combined['genre'] + ' ' + combined['description']
        combined = combined.reset_index(drop=True)
        return combined

    except KeyError as e:
        st.error(f"Column missing in one of the CSV files: {e}")
        st.stop()
    except FileNotFoundError as e:
        st.error(f"Couldnt find one of the dataset files: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Something went wrong loading the data: {e}")
        st.stop()

@st.cache_resource
def build_vectorizer(combined):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(combined['search_text'])
    return vectorizer, tfidf_matrix

def get_genre_recommendation(genre_input, combined, client, min_rating=6.0):
    genre_input = genre_input.lower().strip()
    matches = combined[combined['genre'].str.lower().str.contains(genre_input, na=False)]

    rated = matches[matches['imdb_rating'].notna()]
    unrated = matches[matches['imdb_rating'].isna()]
    rated = rated[rated['imdb_rating'] >= min_rating]
    matches = pd.concat([rated, unrated], ignore_index=True)

    if matches.empty:
        return None, None

    platform_counts = matches['platform'].value_counts()
    summary = f"The user likes {genre_input} content. Here are some matches:\n\n"
    for platform in platform_counts.index:
        summary += f"{platform}: {platform_counts[platform]} titles available\n"
        top = matches[matches['platform'] == platform].head(3)
        for _, row in top.iterrows():
            summary += f"  - {row['title']} ({row['type']})\n"
        summary += "\n"

    chat = client.chat.completions.create(
        messages=[{
            "role": "system",
            "content": "You are a friendly streaming guide for Indian users. Give a short friendly recommendation telling the user which platform suits them best and what to watch first. Keep it conversational and under 100 words."
        }, {"role": "user", "content": summary}],
        model="llama-3.3-70b-versatile",
    )
    return chat.choices[0].message.content, matches

def get_rag_recommendation(user_query, combined, vectorizer, tfidf_matrix, client):
    query_vec = vectorizer.transform([user_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-10:][::-1]
    top_matches = combined.iloc[top_indices]

    context = f"User asked: {user_query}\n\nMost relevant shows and movies found:\n\n"
    for _, row in top_matches.iterrows():
        context += f"Title: {row['title']}\nPlatform: {row['platform']}\nGenre: {row['genre']}\nType: {row['type']}\n"
        if row['description']:
            context += f"Description: {row['description'][:150]}\n"
        context += "\n"

    chat = client.chat.completions.create(
        messages=[{
            "role": "system",
            "content": "You are a friendly streaming guide for Indian users. Give personalized recommendations based on the context. Be conversational and always mention which platform the show is on. Keep it under 120 words."
        }, {"role": "user", "content": context}],
        model="llama-3.3-70b-versatile",
    )
    return chat.choices[0].message.content, top_matches

# colored bar chart using plotly so each platform gets its own color
def show_platform_chart(matches):
    import plotly.graph_objects as go
    
    platform_counts = matches['platform'].value_counts().reset_index()
    platform_counts.columns = ['Platform', 'Titles']
    
    colors = [PLATFORM_COLORS.get(p, "#00d4d4") for p in platform_counts['Platform']]
    
    fig = go.Figure(go.Bar(
        x=platform_counts['Platform'],
        y=platform_counts['Titles'],
        marker_color=colors,
        text=platform_counts['Titles'],
        textposition='outside',
        textfont=dict(color='#e8e0d0', size=12)
    ))
    
    fig.update_layout(
        paper_bgcolor='#0d0d14',
        plot_bgcolor='#0d0d14',
        font=dict(color='#e8e0d0', family='Space Mono'),
        margin=dict(t=30, b=10, l=10, r=10),
        showlegend=False,
        bargap=0.4,
        xaxis=dict(
            showgrid=False,
            showline=False,
            color='#555',
            tickfont=dict(size=11, color='#555')
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            zeroline=False,
            visible=False
        ),
        height=220
    )

    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': False
    })

# empty state shown when no results are found
def show_empty_state(search_term):
    st.markdown(f"""
    <div style='text-align:center; padding:3rem 1rem; border:0.5px dashed #2a2a3a; border-radius:12px; margin:1.5rem 0;'>
        <div style='font-size:2.5rem; margin-bottom:1rem;'>🎬</div>
        <div style='font-family:Space Mono,monospace; font-size:13px; color:#00d4d4; letter-spacing:2px; text-transform:uppercase; margin-bottom:0.75rem;'>
            No results found
        </div>
        <div style='font-family:Playfair Display,serif; font-size:14px; color:#555; margin-bottom:1.5rem;'>
            We couldn't find anything matching <span style='color:#e8e0d0;'>"{search_term}"</span>
        </div>
        <div style='font-family:Space Mono,monospace; font-size:10px; color:#444; letter-spacing:1px; line-height:1.8;'>
            Try a broader genre like Action, Drama or Comedy<br>
            Or lower the IMDb rating filter<br>
            Or try the AI Search tab and describe what you want
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_results(result, matches, tab_prefix=""):
    st.success(result)
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # colored platform chart
    show_platform_chart(matches)

    st.markdown("### Top Picks")
    for platform in matches['platform'].unique():
        color = PLATFORM_COLORS.get(platform, "#00d4d4")
        badge = PLATFORM_BADGES.get(platform, platform)

        # platform section header with its color
        st.markdown(f"""
        <div style='border-left:3px solid {color}; padding-left:10px; margin:1.2rem 0 0.6rem;'>
            <span style='font-family:Space Mono,monospace; font-size:11px; color:{color}; letter-spacing:2px; text-transform:uppercase;'>{badge}</span>
        </div>
        """, unsafe_allow_html=True)

        top = matches[matches['platform'] == platform].head(5).reset_index(drop=True)
        for idx, row in top.iterrows():
            col1, col2 = st.columns([4, 1])
            with col1:
                # platform colored dot next to each title
                st.markdown(f"""
                <div style='padding:6px 0;'>
                    <span style='color:{color}; font-size:10px;'>●</span>
                    <span style='color:#e8e0d0; font-size:14px; margin-left:8px;'>{row['title']}</span>
                    <span style='color:#444; font-size:11px; margin-left:6px;'>{row['type']}</span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                already_saved = row['title'] in st.session_state.watchlist
                if already_saved:
                    st.markdown(f"<span style='color:{color}; font-size:20px;'>♥</span>", unsafe_allow_html=True)
                else:
                    button_key = f"{tab_prefix}_{idx}_{platform}_{row['title'][:8]}"
                    if st.button("♥", key=button_key):
                        st.session_state.watchlist.append(row['title'])

combined = load_data()
vectorizer, tfidf_matrix = build_vectorizer(combined)
client = Groq(api_key=GROQ_API_KEY)

if st.session_state.watchlist:
    st.markdown(f"""
    <div style='background:#0d0d14; border:0.5px solid #1a3a3a; border-radius:8px; padding:0.75rem 1rem; margin-bottom:1rem; font-family:Space Mono,monospace; font-size:11px; color:#00d4d4; letter-spacing:1px;'>
        ♥ WATCHLIST — {len(st.session_state.watchlist)} saved: {", ".join(st.session_state.watchlist[:3])}{"..." if len(st.session_state.watchlist) > 3 else ""}
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎭  By Genre", "🤖  AI Search", "♥  My Watchlist"])

with tab1:
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    genre = st.text_input("Enter your favourite genre", placeholder="e.g. Action, Drama, Comedy, Horror", key="genre_input")
    mood = st.selectbox("What's your mood?", list(MOOD_MAP.keys()), key="mood_select")
    min_rating = st.slider("Minimum IMDb rating", 1.0, 10.0, 6.0, 0.5, key="rating_slider")

    if st.button("Find My Shows", key="genre_btn"):
        search_term = genre.strip()
        if MOOD_MAP[mood]:
            search_term = search_term + " " + MOOD_MAP[mood] if search_term else MOOD_MAP[mood]
        if search_term == "":
            st.warning("Please enter a genre or select a mood!")
        else:
            with st.spinner("Finding the best content for you..."):
                result, matches = get_genre_recommendation(search_term, combined, client, min_rating)
            if result is None:
                st.session_state.genre_result = "empty"
                st.session_state.genre_matches = None
                st.session_state.genre_search_term = search_term
            else:
                st.session_state.genre_result = result
                st.session_state.genre_matches = matches
                st.session_state.genre_search_term = search_term

    if st.session_state.genre_result == "empty":
        show_empty_state(st.session_state.get("genre_search_term", ""))
    elif st.session_state.genre_result is not None:
        show_results(st.session_state.genre_result, st.session_state.genre_matches, tab_prefix="genre")

with tab2:
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    user_query = st.text_input("What are you in the mood for?", placeholder="e.g. I loved Mirzapur, suggest something similar", key="rag_input")

    if st.button("Get Recommendations", key="rag_btn"):
        if user_query.strip() == "":
            st.warning("Please tell me what you're looking for!")
        else:
            with st.spinner("Finding the best content for you..."):
                result, matches = get_rag_recommendation(user_query, combined, vectorizer, tfidf_matrix, client)
            st.session_state.rag_result = result
            st.session_state.rag_matches = matches

    if st.session_state.rag_result is not None:
        show_results(st.session_state.rag_result, st.session_state.rag_matches, tab_prefix="rag")

with tab3:
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    if not st.session_state.watchlist:
        st.markdown("""
        <div style='text-align:center; padding:3rem 1rem; border:0.5px dashed #2a2a3a; border-radius:12px; margin:1rem 0;'>
            <div style='font-size:2rem; margin-bottom:1rem;'>♥</div>
            <div style='font-family:Space Mono,monospace; font-size:12px; color:#444; letter-spacing:1px;'>
                Nothing saved yet<br>
                <span style='font-size:10px; color:#333;'>Hit the ♥ on any recommendation to save it here</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='font-family:Space Mono,monospace; font-size:11px; color:#00d4d4; letter-spacing:2px; margin-bottom:1rem;'>{len(st.session_state.watchlist)} TITLES SAVED</div>", unsafe_allow_html=True)
        for i, title in enumerate(st.session_state.watchlist):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"<span style='color:#e8e0d0; font-size:14px;'>{title}</span>", unsafe_allow_html=True)
            with col2:
                if st.button("Remove", key=f"remove_{i}"):
                    st.session_state.watchlist.remove(title)
                    st.rerun()

        if st.button("Clear Watchlist", key="clear_watchlist"):
            st.session_state.watchlist = []
            st.rerun()
