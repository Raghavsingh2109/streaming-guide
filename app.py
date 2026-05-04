import streamlit as st
import pandas as pd
from groq import Groq

st.set_page_config(page_title="Smart Streaming Guide", page_icon="🎬")

@st.cache_data
def load_data():
    base = "https://raw.githubusercontent.com/Raghavsingh2109/streaming-guide/main/"
    
    prime = pd.read_csv(base + "primevideo_india_movies_and_shows.csv")
    netflix = pd.read_csv(base + "netflix_india_shows_and_movies.csv")
    hotstar = pd.read_csv(base + "hotstar.csv")

    prime_clean = prime[['name', 'type', 'genre', 'release_year']].copy()
    prime_clean['platform'] = 'Prime Video'
    prime_clean.columns = ['title', 'type', 'genre', 'release_year', 'platform']

    netflix_clean = netflix[['name', 'type', 'genre', 'release_year']].copy()
    netflix_clean['platform'] = 'Netflix'
    netflix_clean.columns = ['title', 'type', 'genre', 'release_year', 'platform']

    hotstar_clean = hotstar[['title', 'type', 'genre', 'year']].copy()
    hotstar_clean['platform'] = 'Hotstar'
    hotstar_clean.columns = ['title', 'type', 'genre', 'release_year', 'platform']

    combined = pd.concat([prime_clean, netflix_clean, hotstar_clean], ignore_index=True)
    combined = combined.dropna(subset=['title', 'genre'])
    combined = combined.reset_index(drop=True)
    return combined

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
        messages=[{"role": "user", "content": f"""You are a friendly streaming guide for Indian users.
Based on these results, give a short friendly recommendation telling the user
which platform suits them best and what to watch first. Keep it simple and conversational.
{summary}"""}],
        model="llama-3.3-70b-versatile",
    )
    return chat.choices[0].message.content, matches

st.title("🎬 Smart Streaming Guide - India")
st.write("Find the best platform and shows based on your favourite genre!")

combined = load_data()

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    groq_api_key = st.text_input("Enter your Groq API key", type="password")
    if not groq_api_key:
        st.warning("Please enter your Groq API key to use the app")
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
            st.subheader("📊 Titles by platform")
            pc = matches['platform'].value_counts().reset_index()
            pc.columns = ['Platform', 'Titles']
            st.bar_chart(pc.set_index('Platform'))
            st.subheader("🎥 Top picks")
            for platform in matches['platform'].unique():
                st.write(f"**{platform}**")
                top = matches[matches['platform'] == platform].head(3)[['title', 'type']]
                st.dataframe(top, hide_index=True)
