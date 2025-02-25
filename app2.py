# Import libraries
import streamlit as st
import pandas as pd
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch


class TorchClassesOverride:
    def __getattr__(self, name):
        if name == "__path__":
            return None
        return None

import sys
if "torch.classes" in sys.modules:
    sys.modules["torch.classes"] = TorchClassesOverride()

# Column configuration
ANIME_NAME_COL = "anime"
IMAGE_COL = "anime_img"
EPISODES_COL_IDX = 3
RATING_COL_IDX = 4
GENRE_START_COL_IDX = 5

# Load data
@st.cache_data
def load_data():
    file_path = r"C:\Users\user\Desktop\recommdender_system\anime.csv"
    df = pd.read_csv(file_path, encoding="utf-8")
    
    # Validate columns
    required_cols = [ANIME_NAME_COL, IMAGE_COL] + df.columns[3:5].tolist()
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Clean data
   df.iloc[:, EPISODES_COL_IDX] = df.iloc[:, EPISODES_COL_IDX].fillna(0).astype(int)
   df.iloc[:, RATING_COL_IDX] = df.iloc[:, RATING_COL_IDX].fillna(0).astype(float)
    
    return df

df = load_data()

# Create enhanced descriptions
def create_enhanced_descriptions(df):
    def get_genres(row):
        genre_cols = df.columns[GENRE_START_COL_IDX:]
        return ' '.join([col for col, val in zip(genre_cols, row[GENRE_START_COL_IDX:]) if val == 1])
    
    episodes = df.iloc[:, EPISODES_COL_IDX].astype(str)
    ratings = df.iloc[:, RATING_COL_IDX].astype(str)
    genres = df.apply(get_genres, axis=1)
    
    return (
        df[ANIME_NAME_COL] + " " +
        "Episodes:" + episodes + " " +
        "Rating:" + ratings + " " +
        "Genres:" + genres
    )

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()

# Build FAISS index
@st.cache_resource
def build_faiss_index():
    descriptions = create_enhanced_descriptions(df)
    embeddings = model.encode(descriptions.tolist())
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return index

faiss_index = build_faiss_index()

# Build TF-IDF vectors
@st.cache_resource
def build_tfidf():
    descriptions = create_enhanced_descriptions(df)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    return vectorizer, tfidf_matrix

tfidf_vectorizer, tfidf_matrix = build_tfidf()

# Natural language filter parsing
def parse_natural_filters(query):
    filters = {}
    patterns = [
        (r"(rating|score|rated)\s+(above|over|more than|at least|greater than)\s+(\d+\.?\d*)", ">", "rating"),
        (r"(rating|score|rated)\s+(below|under|less than|at most|lower than)\s+(\d+\.?\d*)", "<", "rating"),
        (r"(episodes|length)\s+(over|more than|above)\s+(\d+)", ">", "episodes"),
        (r"(episodes|length)\s+(under|below|less than)\s+(\d+)", "<", "episodes"),
        (r"(\d+\.?\d*)\+ rating", ">=", "rating"),
        (r"rating under (\d+\.?\d*)", "<", "rating"),
        (r"at least (\d+\.?\d*) rating", ">=", "rating")
    ]

    for pattern, operator, col in patterns:
        matches = re.finditer(pattern, query, re.IGNORECASE)
        for match in matches:
            value = float(match.group(match.lastindex))
            filters[col] = (operator, value)
            query = query.replace(match.group(0), "")

    return query.strip(), filters

def apply_numeric_filters(df, filters):
    filtered_df = df.copy()
    for col, (op, val) in filters.items():
        try:
            filtered_df = filtered_df.query(f"`{col}` {op} {val}")
        except:
            continue
    return filtered_df

# Recommendation function

def recommend_anime(user_input, top_n=5, faiss_weight=0.6, tfidf_weight=0.4):
    # Parse natural language filters
    clean_query, filters = parse_natural_filters(user_input)
    
    # Semantic search
    query_embedding = model.encode([clean_query])
    faiss_distances, faiss_indices = faiss_index.search(query_embedding, top_n*2)
    faiss_scores = 1 / (1 + faiss_distances[0])
    
    # TF-IDF search
    query_tfidf = tfidf_vectorizer.transform([clean_query])
    tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    tfidf_indices = tfidf_scores.argsort()[-top_n*2:][::-1]
    
    # Create combined results with scores
    combined = pd.DataFrame({
        'index': np.concatenate([faiss_indices[0], tfidf_indices]),
        'score': np.concatenate([
            faiss_scores * faiss_weight,
            tfidf_scores[tfidf_indices] * tfidf_weight
        ])
    }).drop_duplicates('index')
    
    # Merge scores with original data
    results_with_scores = df.iloc[combined['index']].copy()
    results_with_scores['score'] = combined['score'].values
    print(results_with_scores)
    
    # Apply numeric filters
    filtered_results = apply_numeric_filters(results_with_scores, filters)
    
    # Return top N results with highest scores
    return filtered_results.nlargest(top_n, 'score')
