import streamlit as st
from knn_model import get_recommendations, get_track_options, df

st.title("🎧 Spotify Song Recommender")

# Song dropdown
track_options = get_track_options()
selected = st.selectbox("Choose a song:", track_options)

# Slider for number of results
top_n = st.slider("Number of recommendations", 1, 5, 5)

# Checkbox to filter by genre
genre_filter = st.checkbox("Match same genre only")

# Extract track ID from selected option
track_id = selected.split(" | ")[0]

# Get recommendations
results = get_recommendations(track_id, top_n=top_n, genre_filter=genre_filter)

# Display results
if results:
    st.subheader("🔁 Recommendations:")
    for rec in results:
        st.markdown(f"**{rec['track_name']}** by *{rec['artist_name']}*  \n🎼 Genre: {rec['genre']}  \n🔗 Similarity: {rec['similarity']}%")
        st.markdown("---")
else:
    st.warning("No similar tracks found.")
