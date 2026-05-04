# importing all the stuff i need for this project
import streamlit as st
import pandas as pd
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# grabbing the api key from environment so it doesnt show up in the code
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

st.set_page_config(page_title="StreamIQ", page_icon="🎬", layout="centered")

# all the custom styling to make it look cinematic and dark
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
.stDataFrame {
    background-color: #12121a !important;
    border: 1px solid #2a2a3a !important;
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

# app header with the StreamIQ branding
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

# stop the app early if the api key isnt set up properly
if not GROQ_API_KEY:
    st.error("Groq API key not found. Please set it in your environment secrets.")
    st.stop()

# using session state to keep the watchlist alive while the app is running
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

# mood to genre mapping - makes it easier for users who dont know genre names
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

# loading and combining all three platform datasets into one
# wrapped in try/except so if a column name is slightly different it shows a proper error
@st.cache_data
def load_data():
    try:
        prime = pd.read_csv("primevideo_india_movies_and_shows.csv")
        netflix = pd.read_csv("netflix_india_shows_and_movies.csv")
        hotstar = pd.read_csv("hotstar.csv")

        # keeping only the columns i need and renaming them to match
        prime_clean = prime[['name', 'type', 'genre', 'release_year', 'synopsis']].copy()
        prime_clean['platform'] = 'Prime Video'
        prime_clean.columns = ['title', 'type', 'genre', 'release_year', 'description', 'platform']

        netflix_clean = netflix[['name', 'type', 'genre', 'release_year', 'description']].copy()
        netflix_clean['platform'] = 'Netflix'
        netflix_clean.columns = ['title', 'type', 'genre', 'release_year', 'description', 'platform']

        hotstar_clean = hotstar[['title', 'type', 'genre', 'year', 'description']].copy()
        hotstar_clean['platform'] = 'Hotstar'
        hotstar_clean.columns = ['title', 'type', 'genre', 'release_year', 'description', 'platform']

        # combining all platforms into one big dataframe
        st.write("Prime columns:", prime.columns.tolist())
        st.write("Netflix columns:", netflix.columns.tolist())
        st.write("Hotstar columns:", hotstar.columns.tolist())
        combined = pd.concat([prime_clean, netflix_clean, hotstar_clean], ignore_index=True)
        combined = combined.dropna(subset=['title', 'genre'])
        combined['description'] = combined['description'].fillna('')

        # this is what the recommendation model will search through
        combined['search_text'] = combined['title'] + ' ' + combined['genre'] + ' ' + combined['description']
        combined = combined.reset_index(drop=True)
        return combined

    except KeyError as e:
        st.error(f"Looks like a column is missing in one of the CSV files: {e}")
        st.stop()
    except FileNotFoundError as e:
        st.error(f"Couldnt find one of the dataset files: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Something went wrong while loading the data: {e}")
        st.stop()

# building the tfidf vectorizer - this is the core of the recommendation system
# cache_resource means it only builds once and reuses it
@st.cache_resource
def build_vectorizer(combined):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(combined['search_text'])
    return vectorizer, tfidf_matrix

# genre based search - filters by genre and optionally by rating
def get_genre_recommendation(genre_input, combined, client, min_rating=6.0):
    genre_input = genre_input.lower().strip()
    matches = combined[combined['genre'].str.lower().str.contains(genre_input, na=False)]

    # only apply rating filter if the dataset actually has a rating column
    if 'rating' in combined.columns:
        matches = matches[matches['rating'] >= min_rating]

    matches = matches.reset_index(drop=True)

    if matches.empty:
        return None, None

    # building a summary to send to groq
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

# rag based search - uses cosine similarity to find closest matches to what user typed
def get_rag_recommendation(user_query, combined, vectorizer, tfidf_matrix, client):
    query_vec = vectorizer.transform([user_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # getting top 10 most similar titles
    top_indices = similarities.argsort()[-10:][::-1]
    top_matches = combined.iloc[top_indices]

    # building context to pass into groq
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

# reusable function to display results so i dont repeat the same code in both tabs
def show_results(result, matches):
   def show_results(result, matches, tab_prefix=""):
    st.success(result)
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # bar chart showing how many titles each platform has
    pc = matches['platform'].value_counts().reset_index()
    pc.columns = ['Platform', 'Titles']
    st.bar_chart(pc.set_index('Platform'))

    st.markdown("### Top Picks")
    for platform in matches['platform'].unique():
        st.markdown(f"**{platform}**")
        top = matches[matches['platform'] == platform].head(5)[['title', 'type', 'genre']]
        for idx, row in top.iterrows():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"<span style='color:#e8e0d0; font-size:14px;'>{row['title']} <span style='color:#555; font-size:12px;'>({row['type']})</span></span>", unsafe_allow_html=True)
            with col2:
                # unique key using tab prefix + index + title + platform
                button_key = f"{tab_prefix}_{idx}_{row['title'][:10]}_{platform}"
                if st.button("♥", key=button_key):
                    if row['title'] not in st.session_state.watchlist:
                        st.session_state.watchlist.append(row['title'])
                        st.toast(f"Added {row['title']} to watchlist!")
                    else:
                        st.toast(f"Already in watchlist!")

# loading data and building the model when the app starts
combined = load_data()
vectorizer, tfidf_matrix = build_vectorizer(combined)
client = Groq(api_key=GROQ_API_KEY)

# showing watchlist summary at the top if user has saved anything
if st.session_state.watchlist:
    st.markdown(f"""
    <div style='background:#0d0d14; border:0.5px solid #1a3a3a; border-radius:8px; padding:0.75rem 1rem; margin-bottom:1rem; font-family:Space Mono,monospace; font-size:11px; color:#00d4d4; letter-spacing:1px;'>
        ♥ WATCHLIST — {len(st.session_state.watchlist)} saved: {", ".join(st.session_state.watchlist[:3])}{"..." if len(st.session_state.watchlist) > 3 else ""}
    </div>
    """, unsafe_allow_html=True)

# three tabs - genre search, ai search, and watchlist
tab1, tab2, tab3 = st.tabs(["🎭  By Genre", "🤖  AI Search", "♥  My Watchlist"])

with tab1:
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    genre = st.text_input("Enter your favourite genre", placeholder="e.g. Action, Drama, Comedy, Horror", key="genre_input")
    mood = st.selectbox("What's your mood?", list(MOOD_MAP.keys()), key="mood_select")

    # slider value gets passed into the function to filter results
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
                st.error("No matches found. Try a different genre or mood!")
            else:
                show_results(result, matches)

with tab2:
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    user_query = st.text_input("What are you in the mood for?", placeholder="e.g. I loved Mirzapur, suggest something similar", key="rag_input")

    if st.button("Get Recommendations", key="rag_btn"):
        if user_query.strip() == "":
            st.warning("Please tell me what you're looking for!")
        else:
            with st.spinner("Finding the best content for you..."):
                result, matches = get_rag_recommendation(user_query, combined, vectorizer, tfidf_matrix, client)
            show_results(result, matches)

with tab3:
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    if not st.session_state.watchlist:
        st.markdown("<div style='color:#555; font-family:Space Mono,monospace; font-size:12px; letter-spacing:1px;'>No shows saved yet — hit the ♥ Save button on any recommendation!</div>", unsafe_allow_html=True)
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

        # clear everything button at the bottom
        if st.button("Clear Watchlist", key="clear_watchlist"):
            st.session_state.watchlist = []
            st.rerun()
