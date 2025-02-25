import streamlit as st
import pandas as pd
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Column configuration
ANIME_NAME_COL = "anime"
IMAGE_COL = "anime_img"
EPISODES_COL = "episodes"
RATE_COL = "rate" 
GENRE_START_COL_IDX = 5

# Load data
@st.cache_data
def load_data():
    file_path = r"C:\Users\user\Desktop\recommdender_system\anime.csv"
    df = pd.read_csv(file_path, encoding="utf-8")
    
    # check required columns exist
    required_cols = [ANIME_NAME_COL, IMAGE_COL, EPISODES_COL, RATE_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Fill missing values
    df[EPISODES_COL] = df[EPISODES_COL].fillna(0).astype(int)
    df[RATE_COL] = df[RATE_COL].fillna(0).astype(float)
    
    return df

df = load_data()

# Create enhanced descriptions
def create_enhanced_descriptions(df):
    def get_genres(row):
        genre_cols = df.columns[GENRE_START_COL_IDX:]
        return ' '.join([col for col, val in zip(genre_cols, row[GENRE_START_COL_IDX:]) if val == 1])
    
    return (
        df[ANIME_NAME_COL] + " " +
        "Episodes:" + df[EPISODES_COL].astype(str) + " " +
        "Rate:" + df[RATE_COL].astype(str) + " " +
        "Genres:" + df.apply(get_genres, axis=1)
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
    embeddings = model.encode(descriptions.tolist()).astype(np.float32)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
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

# Parse natural language filters
def parse_natural_filters(query):
    filters = {}
    patterns = [
        (r"(rating|rate|score|rated)\s+(above|over|more than|at least|greater than)\s+(\d+\.?\d*)", ">", RATE_COL),
        (r"(rating|rate|score|rated)\s+(below|under|less than|at most|lower than)\s+(\d+\.?\d*)", "<", RATE_COL),
        (r"(episodes|length)\s+(over|more than|above)\s+(\d+)", ">", EPISODES_COL),
        (r"(episodes|length)\s+(under|below|less than)\s+(\d+)", "<", EPISODES_COL),
        (r"(\d+\.?\d*)\+ rating", ">=", RATE_COL),
        (r"rating under (\d+\.?\d*)", "<", RATE_COL),
        (r"at least (\d+\.?\d*) rating", ">=", RATE_COL)
    ]

    for pattern, operator, col in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            value = float(match.group(match.lastindex))
            filters[col] = (operator, value)
            query = query.replace(match.group(0), "")

    return query.strip(), filters

# Apply numeric filters
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
    query_embedding = model.encode([clean_query]).astype(np.float32)
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
    
    # Apply numeric filters
    filtered_results = apply_numeric_filters(results_with_scores, filters)
    
    # Return top N results
    return filtered_results.nlargest(top_n, 'score')
