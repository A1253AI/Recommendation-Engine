# Import libraries
import streamlit as st
import pandas as pd
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Override torch classes module to avoid errors
class TorchClassesOverride:
    def __getattr__(self, name):
        if name == "__path__":
            return None
        return None

import sys
if "torch.classes" in sys.modules:
    sys.modules["torch.classes"] = TorchClassesOverride()

# Load dataset from CSV file only
@st.cache_data
def load_data():
    file_path = r"C:\Users\user\Desktop\recommdender_system\anime.csv"  # Path to the data file
    df = pd.read_csv(file_path, encoding="utf-8")
    return df

df = load_data()

# Load SentenceTransformer Model
@st.cache_resource
def load_embedding_model():
    load = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return load

model = load_embedding_model()

# Create  descriptions including episodes and votes
def create_enhanced_descriptions(df):
    # Assuming column 3 is episodes and column 4 is votes
    episodes_info = "episodes: " + df.iloc[:, 3].astype(str)
    votes_info = "votes: " + df.iloc[:, 4].astype(str)
    
    # Combine anime title, episodes, votes, and genre information
    anime_descriptions = (df["anime"] + " " + 
                         episodes_info + " " + 
                         votes_info + " " + 
                         df.iloc[:, 12:].astype(str).apply(lambda x: " ".join(x), axis=1))
    
    return anime_descriptions

# Create FAISS Index for Semantic Search with enhanced descriptions
@st.cache_resource
def build_faiss_index(df):
    anime_descriptions = create_enhanced_descriptions(df)
    anime_embeddings = model.encode(anime_descriptions.tolist(), convert_to_numpy=True)
    
    index = faiss.IndexFlatL2(anime_embeddings.shape[1])
    index.add(anime_embeddings)
    
    return index, anime_embeddings

faiss_index, anime_vectors = build_faiss_index(df)

# TF-IDF Vectorizer with enhanced descriptions
@st.cache_resource
def build_tfidf(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    anime_descriptions = create_enhanced_descriptions(df)
    tfidf_matrix = vectorizer.fit_transform(anime_descriptions)
    return vectorizer, tfidf_matrix

tfidf_vectorizer, tfidf_matrix = build_tfidf(df)

# Simplified Anime Recommendation Function
def recommend_anime(user_input, top_n=5):
    results = []
    
    try:
        # Semantic search using FAISS
        query_embedding = model.encode([user_input], convert_to_numpy=True)
        _, faiss_indices = faiss_index.search(query_embedding, top_n)
        faiss_results = [df.iloc[i] for i in faiss_indices[0]]
        results.extend(faiss_results)
    except Exception as e:
        st.error(f"Error during semantic search: {e}")
    
    try:
        # Keyword matching using TF-IDF
        query_tfidf = tfidf_vectorizer.transform([user_input])
        cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
        tfidf_results = [df.iloc[i] for i in cosine_similarities.argsort()[-top_n:][::-1]]
        results.extend(tfidf_results)
    except Exception as e:
        st.error(f"Error during TF-IDF recommendation: {e}")
    
    # Remove duplicates and limit to top_n
    unique_results = list({anime["anime"]: anime for anime in results}.values())[:top_n]
    
    return unique_results

# Streamlit UI
st.title("AniFlix, your go to Anime Recommendation store ")

user_input = st.text_input("Enter an anime name or describe what you're looking for:", 
                       placeholder="E.g., 'Action anime with many episodes' or 'Popular anime with few episodes'")

if st.button("Get Recommendations", type="primary"):
    if user_input:
        with st.spinner("Finding anime recommendations..."):
            recommended_anime = recommend_anime(user_input, top_n=5)
        
        if recommended_anime:
            st.success(f"Found {len(recommended_anime)} recommendations for you!")
            
            for i, anime in enumerate(recommended_anime, 1):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    try:
                        st.image(anime['anime_img'], width=150)
                    except:
                        st.image("https://via.placeholder.com/150x225?text=No+Image", width=150)
                
                with col2:
                    st.markdown(f"### {i}. {anime['anime']}")
                    if 'anime_url' in anime:
                        st.markdown(f"[View Details]({anime['anime_url']})")
                    
                    # Highlight episodes and votes specifically
                    episodes_col = df.columns[3]  # Assuming column 3 is episodes
                    votes_col = df.columns[4]     # Assuming column 4 is votes
                    
                    st.write(f"**Episodes:** {anime[episodes_col]} | **Votes:** {anime[votes_col]} | **Rating:** {anime['rate']}")
                    
                    # Only show genres that this anime has (value=1)
                    genres = [genre for genre, has_genre in anime.iloc[12:].items() if has_genre == 1]
                    if genres:
                        st.write(f"**Genres:** {', '.join(genres)}")
                
                st.markdown("---")
        else:
            st.warning("No recommendations found. Try a different query.")
    else:
        st.warning("Please enter an anime name or preference.")

st.caption("Anime Recommendation System | Data from local CSV file")