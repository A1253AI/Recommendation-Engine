import streamlit as st
import app2 
from app2 import df

# Streamlit UI
st.title("AniFlix - Natural Language Anime Recommender")

# Query examples
st.markdown("""
**Example queries:**
- "Action anime with rating above 4.5"
- "Emotional dramas under 50 episodes"
- "Popular shows with more than 100 episodes"
- "Highly rated movies (at least 4.7)"
- "Classic anime rated over 4.0"
""")

# User inputs
user_input = st.text_input("Describe what you're looking for:", 
                         placeholder="e.g., 'Epic fantasy with rating above 4.3'")
col1, col2 = st.columns(2)
with col1:
    top_n = st.slider("Number of recommendations", 3, 10, 5)
with col2:
    method = st.selectbox("Search Type", ["Hybrid", "Semantic", "Keyword"])

if st.button("Get Recommendations", type="primary"):
    if user_input:
        with st.spinner("Finding perfect matches..."):
            # Select recommendation method
            if method == "Hybrid":
                results = app2.recommend_anime(user_input, top_n)
            elif method == "Semantic":
                results = app2.recommend_anime(user_input, top_n, faiss_weight=1, tfidf_weight=0)
            else:
                results = app2.recommend_anime(user_input, top_n, faiss_weight=0, tfidf_weight=1)
            
            # Display results
            if not results.empty:
                st.success(f"Found {len(results)} matching anime")
                for _, row in results.iterrows():
                    with st.expander(f"{row[app2.ANIME_NAME_COL]}  {row.iloc[app2.RATING_COL_IDX]:.1f}"):
                        cols = st.columns([1, 3])
                        with cols[0]:
                            try:
                                st.image(row[app2.IMAGE_COL], width=150, 
                                       caption=row[app2.ANIME_NAME_COL])
                            except:
                                st.image("https://via.placeholder.com/200x300?text=Image+Not+Found",
                                       width=200)
                        with cols[1]:
                            st.markdown(f"**Episodes:** {row.iloc[app2.EPISODES_COL_IDX]}")
                            genres = ', '.join([col for col in df.columns[app2.GENRE_START_COL_IDX:] 
                                              if row[col] == 1])
                            st.markdown(f"**Genres:** {genres}")
                            if 'anime_url' in df.columns:
                                st.markdown(f"[More Info]({row['anime_url']})")
                st.markdown("---")
            else:
                st.warning("No anime found matching your criteria. Try different filters.")
    else:
        st.warning("Please describe what you're looking for")

st.caption("Smart Anime Recommender Designed for Best Experience")
