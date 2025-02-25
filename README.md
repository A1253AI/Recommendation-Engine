# Recommendation-Engine
AI Powered Recommendation Engine.
Developed a Hybrid FAISS + TF-IDF Recommendation System
Simplified Recommendation Logic: 
The recommendation function now simply:
Gets semantic search results using FAISS
Gets keyword matching results using TF-IDF
Combines them, removes duplicates, and returns the top N

The load_data function reads the anime dataset from a CSV file, ensuring that required columns (anime, anime_img, episodes, rate) exist. It also handles missing values by filling episodes with 0 (as integers) and rate with 0.0 (as floats) to maintain data consistency.
                                             
The create_enhanced_descriptions function generates detailed descriptions for each anime by combining its name, number of episodes, rating, and genres into a single string. This enriched text representation helps improve the effectiveness of search and recommendation models.

The load_model() function loads the SentenceTransformer model (all-MiniLM-L6-v2) for generating text embeddings. This model is used to convert anime descriptions into numerical vectors for efficient similarity comparisons in the recommendation system.

The encode() function from SentenceTransformer converts input text into a high-dimensional numerical vector, also called an embedding. This embedding captures the semantic meaning of the text, allowing for similarity comparisons.

The build_faiss_index() function creates a FAISS (Facebook AI Similarity Search) index, which enables fast nearest-neighbor search for anime recommendations.
Generate Enhanced Descriptions – Combines anime details (title, episodes, rating, genres) into a single descriptive text.
Compute Sentence Embeddings – Uses SentenceTransformer to encode each description into a high-dimensional vector.
Build FAISS Index – Stores these vectors in a FAISS IndexFlatL2, which enables efficient similarity searches.
Add Vectors to FAISS – Converts embeddings to float32 and adds them to the index.

The build_tfidf() function constructs a TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer and generates a matrix representation of the anime descriptions.
Creates text descriptions of each anime (combining name, genres, rating, episodes).
Initializes TfidfVectorizer to convert text into numerical feature vectors.
Fits and transforms descriptions into a sparse matrix where:
Each row is an anime.Each column is a unique word (excluding stopwords like "the", "is", etc.).
The values represent the importance of a word in a description (TF-IDF score).

eg - ["Naruto Episodes:220 Rating:8.3 Genres:Action Adventure",  

 "Attack on Titan Episodes:75 Rating:9.1 Genres:Action Fantasy"]

Adding ,

The parse_natural_filters(query) function helps in FAISS and TF-IDF search by ensuring that only the relevant text is used for similarity matching while applying numeric filters separately.

FAISS and TF-IDF compare text embeddings to find the most similar anime descriptions to the user's query.
However, if the query contains numeric conditions like "rate above 8.5", these don't contribute to text similarity.
The function removes these numeric constraints from the query before encoding it for FAISS/TF-IDF.

Before Parsing: "action anime with rate above 8.5 and episodes under 50"

After Parsing: "action anime with" -> Used for FAISS & TF-IDF

Filters Applied Later: {"rate": (">", 8.5), "episodes": ("<", 50)}

Without this step, FAISS and TF-IDF would try to match unnecessary words like "above" or "under," which are not useful for recommendations.

Once FAISS and TF-IDF retrieve a list of similar anime, the extracted numeric filters are applied to remove results that don’t meet the conditions.

Function --> recommend_anime(user_input, top_n=5, faiss_weight=0.6, tfidf_weight=0.4):

This function takes a user query, retrieves the most relevant anime using FAISS (semantic search) and TF-IDF (keyword-based search), and then filters the results based on numeric conditions.

Extract numeric constraints (e.g., "rate above 8.5" becomes {"rate": (">", 8.5)})
Clean query for text-based search ("action anime with")
faiss_scores = 1 / (1 + faiss_distances[0])
Convert query into vector embedding.
Find similar anime descriptions in FAISS index.
Compute similarity scores --> Lower distance = Higher similarity.

Convert query into TF-IDF representation.
Compute cosine similarity between query & dataset.
Retrieve top matches.

Combine results from FAISS (weighted 60%) and TF-IDF (weighted 40%).
Remove duplicates, keeping the highest score.

Remove anime that don’t meet numeric conditions (e.g., "rate > 8.5").
Return Final Recommendations

return filtered_results.nlargest(top_n, 'score')
Sort by final scores.
Return top N most relevant anime.
recommend_anime("fantasy anime with rate above 8.0 and less than 30 episodes")


Finds fantasy anime using FAISS & TF-IDF.
Filters out results where rate < 8.0 or episodes > 30.
Returns the best-matching anime.
Result: Highly relevant, personalized anime recommendations!


