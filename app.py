import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# page setup
st.set_page_config(page_title="Smart Streaming Guide", page_icon="🎬")

# load datasets
@st.cache_data
def load_data():
    base = "https://raw.githubusercontent.com/Raghavsingh2109/streaming-guide/main/"
    
    prime = pd.read_csv(base + "primevideo_india_movies_and_shows.csv")
    netflix = pd.read_csv(base + "netflix_india_shows_and_movies.csv")
    hotstar = pd.read_csv(base + "hotstar.csv")

    prime_clean = prime[['name', 'type', 'genre', 'imdb_rating', 'synopsis', 'release_year']].copy()
    prime_clean['platform'] = 'Prime Video'
    prime_clean.columns = ['title', 'type', 'genre', 'rating', 'description', 'release_year', 'platform']

    netflix_clean = netflix[['name', 'type', 'genre', 'release_year', 'description']].copy()
    netflix_clean['rating'] = None
    netflix_clean['platform'] = 'Netflix'
    netflix_clean.columns = ['title', 'type', 'genre', 'release_year', 'description', 'rating', 'platform']

    hotstar_clean = hotstar[['title', 'type', 'genre', 'year', 'description', 'age_rating']].copy()
    hotstar_clean['platform'] = 'Hotstar'
    hotstar_clean.columns = ['title', 'type', 'genre', 'release_year', 'description', 'rating', 'platform']

    combined = pd.concat([prime_clean, netflix_clean, hotstar_clean], ignore_index=True)
    combined = combined.dropna(subset=['title', 'genre'])
    combined['description'] = combined['description'].fillna('')
    combined = combined.reset_index(drop=True)
    combined['content'] = combined['genre'] + ' ' + combined['description']

    return combined

# build tfidf model
@st.cache_resource
def build_model(combined):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(combined['content'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# get ai recommendation
def get_recommendation(genre_input, combined, client):
    genre_input = genre_input.lower()
    matches = combined[combined['genre'].str.lower().str.contains(genre_input, na=False)]

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
        messages=[
            {
                "role": "user",
                "content": f"""You are a friendly streaming guide for Indian users.
Based on these results, give a short friendly recommendation telling the user
which platform suits them best and what to watch first. Keep it simple and conversational.

{summary}"""
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    return chat.choices[0].message.content, matches

# main app
st.title("🎬 Smart Streaming Guide - India")
st.write("Find the best platform and shows based on your favourite genre!")

combined = load_data()
build_model(combined)

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    groq_api_key = st.text_input("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", type="password")
    if not groq_api_key:
        st.warning("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")
        st.stop()

client = Groq(api_key=groq_api_key)
genre = st.text_input("Enter your favourite genre", placeholder="e.g. Action, Drama, Comedy, Horror")

if st.button("Get Recommendations"):
    if genre.strip() == "":
        st.warning("Please enter a genre first!")
    else:
        with st.spinner("Finding the best content for you..."):
            result, matches = get_recommendation(genre, combined, client)

        if result is None:
            st.error(f"No matches found for '{genre}'. Try another genre!")
        else:
            st.success("Here's what we found!")
            st.write(result)

            st.subheader("📊 Titles available by platform")
            platform_counts = matches['platform'].value_counts().reset_index()
            platform_counts.columns = ['Platform', 'Titles']
            st.bar_chart(platform_counts.set_index('Platform'))

            st.subheader("🎥 Top picks")
            for platform in matches['platform'].unique():
                st.write(f"**{platform}**")
                top = matches[matches['platform'] == platform].head(3)[['title', 'type']]
                st.dataframe(top, hide_index=True)
