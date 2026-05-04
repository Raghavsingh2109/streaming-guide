import streamlit as st
import pandas as pd
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="Smart Streaming Guide", page_icon="🎬")
st.markdown("""<style>st.markdown("""
<style>
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='padding: 1.5rem 0 0.5rem;'>
    <div style='font-family: Playfair Display, serif; font-size: 2rem; color: #00d4d4; font-weight: 700;'>Stream<span style='color:#e8e0d0; font-weight:300;'>IQ</span></div>
    <div style='font-family: Space Mono, monospace; font-size: 11px; color: #555; letter-spacing: 2px; text-transform: uppercase; margin-top: 4px;'>India's smart streaming guide</div>
</div>
""", unsafe_allow_html=True)</style>""", unsafe_allow_html=True)


GROQ_API_KEY = "••••••••••••••••••••••••••••••••••••••••••••••••••••••••"

@st.cache_data
def load_data():
    prime = pd.read_csv("primevideo_india_movies_and_shows.csv")
    netflix = pd.read_csv("netflix_india_shows_and_movies.csv")
    hotstar = pd.read_csv("hotstar.csv")

    prime_clean = prime[['name', 'type', 'genre', 'release_year', 'synopsis']].copy()
    prime_clean['platform'] = 'Prime Video'
    prime_clean.columns = ['title', 'type', 'genre', 'release_year', 'description', 'platform']

    netflix_clean = netflix[['name', 'type', 'genre', 'release_year', 'description']].copy()
    netflix_clean['platform'] = 'Netflix'
    netflix_clean.columns = ['title', 'type', 'genre', 'release_year', 'description', 'platform']

    hotstar_clean = hotstar[['title', 'type', 'genre', 'year', 'description']].copy()
    hotstar_clean['platform'] = 'Hotstar'
    hotstar_clean.columns = ['title', 'type', 'genre', 'release_year', 'description', 'platform']

    combined = pd.concat([prime_clean, netflix_clean, hotstar_clean], ignore_index=True)
    combined = combined.dropna(subset=['title', 'genre'])
    combined['description'] = combined['description'].fillna('')
    combined['search_text'] = combined['title'] + ' ' + combined['genre'] + ' ' + combined['description']
    combined = combined.reset_index(drop=True)
    return combined

@st.cache_resource
def build_vectorizer(combined):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(combined['search_text'])
    return vectorizer, tfidf_matrix

def get_genre_recommendation(genre_input, combined, client):
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
        messages=[{
            "role": "system",
            "content": "You are a friendly streaming guide for Indian users. Give a short friendly recommendation telling the user which platform suits them best and what to watch first. Keep it simple and conversational."
        },
        {
            "role": "user",
            "content": summary
        }],
        model="llama-3.3-70b-versatile",
    )
    return chat.choices[0].message.content, matches

def get_rag_recommendation(user_query, combined, vectorizer, tfidf_matrix, client):
    query_vec = vectorizer.transform([user_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-10:][::-1]
    top_matches = combined.iloc[top_indices]

    context = f"User asked: {user_query}\n\nHere are the most relevant shows and movies found:\n\n"
    for _, row in top_matches.iterrows():
        context += f"Title: {row['title']}\n"
        context += f"Platform: {row['platform']}\n"
        context += f"Genre: {row['genre']}\n"
        context += f"Type: {row['type']}\n"
        if row['description']:
            context += f"Description: {row['description'][:150]}\n"
        context += "\n"

    chat = client.chat.completions.create(
        messages=[{
            "role": "system",
            "content": "You are a friendly streaming guide for Indian users. Based on the context provided, give personalized recommendations. Be conversational, helpful and specific. Always mention which platform the show is on."
        },
        {
            "role": "user",
            "content": context
        }],
        model="llama-3.3-70b-versatile",
    )
    return chat.choices[0].message.content, top_matches

# --- UI ---
st.title("🎬 Smart Streaming Guide - India")
st.write("Find the best platform and shows based on what you love!")

combined = load_data()
vectorizer, tfidf_matrix = build_vectorizer(combined)
client = Groq(api_key=GROQ_API_KEY)

tab1, tab2 = st.tabs(["🎭 Search by Genre", "🤖 AI Personalised Search"])

# --- TAB 1: Genre Search ---
with tab1:
    st.subheader("Search by Genre")
    st.write("Enter a genre and we'll find the best platform for you!")

    genre = st.text_input(
        "Enter your favourite genre",
        placeholder="e.g. Action, Drama, Comedy, Horror",
        key="genre_input"
    )

    if st.button("Get Recommendations", key="genre_btn"):
        if genre.strip() == "":
            st.warning("Please enter a genre first!")
        else:
            with st.spinner("Finding the best content for you..."):
                result, matches = get_genre_recommendation(genre, combined, client)
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
                    top = matches[matches['platform'] == platform].head(3)[['title', 'type', 'genre']]
                    st.dataframe(top, hide_index=True)

# --- TAB 2: RAG AI Search ---
with tab2:
    st.subheader("AI Personalised Search")
    st.write("Tell me what you're in the mood for and I'll find the best match!")

    user_query = st.text_input(
        "What are you in the mood for?",
        placeholder="e.g. I loved Mirzapur, suggest something similar",
        key="rag_input"
    )

    if st.button("Get Recommendations", key="rag_btn"):
        if user_query.strip() == "":
            st.warning("Please tell me what you're looking for!")
        else:
            with st.spinner("Finding the best content for you..."):
                result, matches = get_rag_recommendation(user_query, combined, vectorizer, tfidf_matrix, client)
            st.success("Here's what we found!")
            st.write(result)
            st.subheader("📊 Titles by platform")
            pc = matches['platform'].value_counts().reset_index()
            pc.columns = ['Platform', 'Titles']
            st.bar_chart(pc.set_index('Platform'))
            st.subheader("🎥 Top picks")
            for platform in matches['platform'].unique():
                st.write(f"**{platform}**")
                top = matches[matches['platform'] == platform].head(3)[['title', 'type', 'genre']]
                st.dataframe(top, hide_index=True)
