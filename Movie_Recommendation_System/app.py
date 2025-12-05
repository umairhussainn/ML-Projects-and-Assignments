import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Loading datasets
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

# Merging datasets
movie_ratings = pd.merge(ratings, movies, on='movieId')

# Pivot table (users x movies)
ratings_matrix = movie_ratings.pivot_table(index='userId', columns='title', values='rating')

# Filling the missing ratings with 0
ratings_matrix_filled = ratings_matrix.fillna(0)

# Computing cosine similarity
cosine_sim = cosine_similarity(ratings_matrix_filled.T)
cosine_sim_df = pd.DataFrame(cosine_sim, index=ratings_matrix.columns, columns=ratings_matrix.columns)

# Streamlit UI
st.title(" Movie Recommendation System")
st.write("Discover movies similar to your favorites!")

# Dropdown for movie selection
movie_list = movies['title'].sort_values().unique()
selected_movie = st.selectbox("Select a movie to get recommendations:", movie_list)

# Number of recommendations
num = st.slider("Number of recommendations:", 5, 20, 10)

# Recommend movies
if st.button("Recommend "):
    if selected_movie not in cosine_sim_df.columns:
        st.error("Movie not found in similarity matrix.")
    else:
        similar_movies = cosine_sim_df[selected_movie].sort_values(ascending=False)[1:num+1]
        st.subheader(f"Movies similar to '{selected_movie}':")
        for i, (movie, score) in enumerate(similar_movies.items(), start=1):
            st.write(f"{i}. {movie} — **Similarity Score:** {score:.3f}")
