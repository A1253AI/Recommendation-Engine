# Recommendation-Engine
AI Powered Recommendation Engine.
Developed a Hybrid FAISS + TF-IDF Recommendation System
Simplified Recommendation Logic: 
The recommendation function now simply:
Gets semantic search results using FAISS
Gets keyword matching results using TF-IDF
Combines them, removes duplicates, and returns the top N

The load_data function reads the anime dataset from a CSV file, ensuring that required columns (anime, anime_img, episodes, rate) exist. It also handles missing values by filling episodes with 0 (as integers) and rate with 0.0 (as floats) to maintain data consistency.
