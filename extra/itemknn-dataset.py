import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests
import zipfile
import io

# --- Step 1: Download and Load Data ---
# This part handles downloading the MovieLens 100K dataset automatically.
def download_movielens_100k():
    """Downloads and extracts the MovieLens 100K dataset."""
    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    print("Downloading MovieLens 100K dataset...")
    try:
        r = requests.get(url)
        r.raise_for_status()  # Raise an exception for bad status codes
        z = zipfile.ZipFile(io.BytesIO(r.content))
        print("Extracting files...")
        z.extractall()
        print("Dataset downloaded and extracted to 'ml-100k' directory.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        exit()
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file.")
        exit()

try:
    # Try loading data assuming it's already downloaded
    ratings_df = pd.read_csv(
        'ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
    movies_df = pd.read_csv(
        'ml-100k/u.item', sep='|', encoding='latin-1', header=None,
        usecols=[0, 1], names=['movie_id', 'title']
    )
except FileNotFoundError:
    # If not found, download it
    download_movielens_100k()
    ratings_df = pd.read_csv(
        'ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
    movies_df = pd.read_csv(
        'ml-100k/u.item', sep='|', encoding='latin-1', header=None,
        usecols=[0, 1], names=['movie_id', 'title']
    )

# --- Step 2: Prepare the User-Item Matrix ---
print("Preparing user-item matrix...")
num_users = ratings_df['user_id'].nunique()
num_items = ratings_df['movie_id'].nunique()

# Create a mapping from original user/movie IDs to matrix indices
user_map = {uid: i for i, uid in enumerate(ratings_df['user_id'].unique())}
movie_map = {mid: i for i, mid in enumerate(ratings_df['movie_id'].unique())}

# Create inverse mappings to get original IDs back
idx_to_user = {i: uid for uid, i in user_map.items()}
idx_to_movie = {i: mid for mid, i in movie_map.items()}

# Create the user-item matrix
R = np.zeros((num_users, num_items))
for _, row in ratings_df.iterrows():
    user_idx = user_map[row['user_id']]
    movie_idx = movie_map[row['movie_id']]
    R[user_idx, movie_idx] = row['rating']

print(f"Created a user-item matrix of shape: {R.shape}")

# --- Step 3: Define the ItemKNNRecommender Class ---
# This is the same class as defined in the previous step.
class ItemKNNRecommender:
    def __init__(self, k=20, min_support=5, shrinkage=100.0):
        self.k = k
        self.min_support = min_support
        self.shrinkage = shrinkage
        self.similarity_matrix_ = None
        self.train_data_ = None
        self.user_means_ = None

    def fit(self, R):
        self.train_data_ = R
        num_items = R.shape[1]
        self.similarity_matrix_ = np.zeros((num_items, num_items))

        self.user_means_ = np.true_divide(R.sum(axis=1), (R != 0).sum(axis=1))
        self.user_means_[np.isnan(self.user_means_)] = 0

        R_centered = R - np.where(R != 0, self.user_means_[:, np.newaxis], 0)
        self.similarity_matrix_ = cosine_similarity(R_centered.T)

        co_rated_counts = (self.train_data_ > 0).astype(float).T @ (self.train_data_ > 0).astype(float)

        if self.min_support > 0:
            self.similarity_matrix_[co_rated_counts < self.min_support] = 0

        if self.shrinkage > 0:
            shrinkage_factor = co_rated_counts / (co_rated_counts + self.shrinkage)
            self.similarity_matrix_ *= shrinkage_factor

        np.fill_diagonal(self.similarity_matrix_, 0)
        return self

    def predict_for_user(self, user_ratings):
        predictions = np.zeros_like(user_ratings, dtype=float)
        rated_indices = np.where(user_ratings > 0)[0]

        for item_to_predict in range(len(user_ratings)):
            if user_ratings[item_to_predict] == 0:
                sims_to_rated_items = self.similarity_matrix_[item_to_predict, rated_indices]
                ratings_of_rated_items = user_ratings[rated_indices]

                if self.k > 0 and len(sims_to_rated_items) > self.k:
                    top_neighbors_indices = np.argsort(-np.abs(sims_to_rated_items))[:self.k]
                    sims_to_rated_items = sims_to_rated_items[top_neighbors_indices]
                    ratings_of_rated_items = ratings_of_rated_items[top_neighbors_indices]

                numerator = sims_to_rated_items @ ratings_of_rated_items
                denominator = np.sum(np.abs(sims_to_rated_items))

                if denominator > 1e-8:
                    predictions[item_to_predict] = numerator / denominator

        return predictions

# --- Step 4: Train the Model ---
print("\nTraining ItemKNN model...")
# Using common hyperparameters for ml-100k
item_knn = ItemKNNRecommender(k=20, min_support=5, shrinkage=100.0)
item_knn.fit(R)
print("Training complete.")

# --- Step 5: Make and Display Recommendations ---
def get_recommendations_for_user(user_id, top_n=10):
    """Generates and prints top N recommendations for a given user_id."""
    try:
        # Map original user_id to matrix index
        user_idx = user_map[user_id]

        # Get user's rating vector from the matrix
        user_ratings = R[user_idx, :]

        # Get predicted scores for this user
        predicted_scores = item_knn.predict_for_user(user_ratings)

        # Find items the user has NOT rated
        unrated_item_indices = np.where(user_ratings == 0)[0]

        # Create a list of (movie_id, score) for unrated items
        recommendations = sorted(
            [(idx_to_movie[i], predicted_scores[i]) for i in unrated_item_indices],
            key=lambda x: x[1],
            reverse=True
        )

        # --- Display Results ---
        print(f"\n--- Recommendations for User {user_id} ---")

        # Display user's top-rated movies for context
        rated_movie_indices = np.where(user_ratings >= 4)[0]
        print("User's highly-rated movies:")
        if len(rated_movie_indices) > 0:
            for idx in rated_movie_indices[:5]: # Show up to 5
                movie_id = idx_to_movie[idx]
                title = movies_df.loc[movies_df['movie_id'] == movie_id, 'title'].iloc[0]
                print(f"  - {title} (Rated: {user_ratings[idx]})")
        else:
            print("  (No movies rated >= 4 stars)")

        # Display top N recommendations
        print(f"\nTop {top_n} recommendations:")
        for movie_id, score in recommendations[:top_n]:
            title = movies_df.loc[movies_df['movie_id'] == movie_id, 'title'].iloc[0]
            print(f"  - {title} (Predicted Score: {score:.4f})")

    except KeyError:
        print(f"Error: User with ID {user_id} not found.")

# Get recommendations for a sample user
# User 1 is a good example as they have rated many popular movies
get_recommendations_for_user(user_id=1, top_n=10)
get_recommendations_for_user(user_id=25, top_n=10)