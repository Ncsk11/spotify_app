import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("SpotifyFeatures.csv")

# Convert categorical to numerical
df['key'] = df['key'].map({'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
                           'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11})
df['mode'] = df['mode'].map({'Minor': 0, 'Major': 1})
df['time_signature'] = df['time_signature'].str.extract(r'(\d+)').astype(float)

features = ['acousticness', 'danceability', 'duration_ms', 'energy',
            'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
            'speechiness', 'tempo', 'time_signature', 'valence', 'popularity']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Fit KNN
knn = NearestNeighbors(n_neighbors=10, metric='cosine')
knn.fit(X_scaled)

# Recommendation logic
def get_recommendations(track_id, top_n=5, genre_filter=False):
    try:
        idx = df[df['track_id'] == track_id].index[0]
    except IndexError:
        return []

    base_genre = df.loc[idx, 'genre']
    query = X_scaled[idx].reshape(1, -1)
    distances, indices = knn.kneighbors(query, n_neighbors=top_n + 10)

    seen_ids = set()
    results = []

    for dist, i in zip(distances[0], indices[0]):
        row = df.iloc[i]
        if row['track_id'] == track_id or row['track_id'] in seen_ids:
            continue
        if genre_filter and row['genre'] != base_genre:
            continue

        results.append({
            'track_id': row['track_id'],
            'track_name': row['track_name'],
            'artist_name': row['artist_name'],
            'genre': row['genre'],
            'similarity': round((1 - dist) * 100, 2)
        })
        seen_ids.add(row['track_id'])

        if len(results) >= top_n:
            break

    return results

# Dropdown options
def get_track_options():
    return df['track_id'] + ' | ' + df['track_name'] + ' by ' + df['artist_name']
