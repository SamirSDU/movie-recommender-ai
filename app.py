import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- Загрузка данных ---
df = pd.read_csv("full_movie_dataset.csv")
movie_df = df[['movie_id', 'title', 'genres']].drop_duplicates().reset_index(drop=True)

# --- Обработка жанров: TF-IDF ---
tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
tfidf_matrix = tfidf.fit_transform(movie_df['genres'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# --- Функция рекомендаций ---
def get_recommendations(title, top_n=10):
    idx = movie_df[movie_df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movie_df['title'].iloc[movie_indices]

# --- Streamlit UI ---
st.title("🎬 Movie Recommendation System")
st.write("Выберите фильм, и мы подберем похожие!")

selected_movie = st.selectbox("Выберите фильм:", movie_df['title'].sort_values().unique())

if st.button("Показать рекомендации"):
    recs = get_recommendations(selected_movie)
    st.subheader("Вам может понравиться:")
    for rec in recs:
        st.write("👉", rec)
