import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import requests
import zipfile
import io

# --- Step 1: Download and Load Data ---
def download_movielens_100k():
    """Downloads and extracts the MovieLens 100K dataset."""
    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    print("Downloading MovieLens 100K dataset...")
    r = requests.get(url)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()
    print("Dataset downloaded and extracted to 'ml-100k' directory.")

try:
    ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
except FileNotFoundError:
    download_movielens_100k()
    ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

# --- Step 2: Create a Dense Subset of Data ---
# To ensure the neighborhood models have enough data to work with, we'll filter
# down to the most active users and most rated movies.
print("Creating and filtering data to a dense subset...")

full_pivot_df = ratings_df.pivot(
    index='user_id',
    columns='movie_id',
    values='rating'
).fillna(0)

# Filter to top N users and M movies
num_users_to_select = 610
num_movies_to_select = 2000

# Find the most rated movies
movie_counts = (full_pivot_df > 0).sum(axis=0)
top_movies_ids = movie_counts.nlargest(num_movies_to_select).index
df_filtered_movies = full_pivot_df[top_movies_ids]

# Find the most active users within that subset
user_counts = (df_filtered_movies > 0).sum(axis=1)
top_users_ids = user_counts.nlargest(num_users_to_select).index
data_to_use = df_filtered_movies.loc[top_users_ids]

print(f"Using a dense subset of the data: {data_to_use.shape[0]} users and {data_to_use.shape[1]} movies.")

# Unpivot the dense data back to a long format for splitting
filtered_ratings_df = data_to_use.stack().reset_index()
filtered_ratings_df.columns = ['user_id', 'movie_id', 'rating']
filtered_ratings_df = filtered_ratings_df[filtered_ratings_df['rating'] > 0]

# --- Step 3: Split Data and Create Matrices ---
print("Splitting filtered data into training and test sets (80/20)...")
train_df, test_df = train_test_split(filtered_ratings_df, test_size=0.2, random_state=42)

# Create mappings for user and movie IDs to matrix indices
all_user_ids = sorted(filtered_ratings_df['user_id'].unique())
all_movie_ids = sorted(filtered_ratings_df['movie_id'].unique())
num_users = len(all_user_ids)
num_items = len(all_movie_ids)

user_map = {uid: i for i, uid in enumerate(all_user_ids)}
movie_map = {mid: i for i, mid in enumerate(all_movie_ids)}

def create_rating_matrix(df, n_users, n_items, u_map, m_map):
    """Creates a user-item rating matrix from a dataframe."""
    R = np.zeros((n_users, n_items))
    for _, row in df.iterrows():
        if row['user_id'] in u_map and row['movie_id'] in m_map:
            user_idx = u_map[row['user_id']]
            movie_idx = m_map[row['movie_id']]
            R[user_idx, movie_idx] = row['rating']
    return R

R_train = create_rating_matrix(train_df, num_users, num_items, user_map, movie_map)
R_test = create_rating_matrix(test_df, num_users, num_items, user_map, movie_map)


# --- Step 4: Define UserKNNRecommender Class ---
class UserKNNRecommender:
    def __init__(self, k=20, similarity_metric='adjusted_cosine'):
        self.k = k
        self.similarity_metric = similarity_metric
        self.similarity_matrix_ = None
        self.train_data_ = None
        self.user_means_ = None

    def fit(self, R):
        self.train_data_ = R

        if self.similarity_metric == 'adjusted_cosine' or self.similarity_metric == 'pearson':
            # For user-based CF, Pearson and Adjusted Cosine are effectively the same.
            self.user_means_ = np.true_divide(R.sum(axis=1), (R != 0).sum(axis=1))
            self.user_means_[np.isnan(self.user_means_)] = 0
            R_centered = R - np.where(R != 0, self.user_means_[:, np.newaxis], 0)
            self.similarity_matrix_ = cosine_similarity(R_centered)
        elif self.similarity_metric == 'cosine':
            # Calculate user means for prediction, even if not used for similarity
            self.user_means_ = np.true_divide(R.sum(axis=1), (R != 0).sum(axis=1))
            self.user_means_[np.isnan(self.user_means_)] = 0
            self.similarity_matrix_ = cosine_similarity(R)

        np.fill_diagonal(self.similarity_matrix_, 0)
        return self

    def predict(self):
        num_users, num_items = self.train_data_.shape
        predictions = np.zeros_like(self.train_data_, dtype=float)

        for u in range(num_users):
            # Find top k neighbors (excluding the user themselves)
            similar_user_indices = np.argsort(-self.similarity_matrix_[u, :])[1:self.k+1]

            for i in range(num_items):
                # Predict only for items the user has not rated in the training set
                if self.train_data_[u, i] == 0:
                    numerator = 0
                    denominator = 0

                    # Consider only neighbors who have rated this item
                    for neighbor_idx in similar_user_indices:
                        if self.train_data_[neighbor_idx, i] > 0:
                            sim = self.similarity_matrix_[u, neighbor_idx]
                            neighbor_rating = self.train_data_[neighbor_idx, i]
                            neighbor_mean = self.user_means_[neighbor_idx]

                            numerator += sim * (neighbor_rating - neighbor_mean)
                            denominator += np.abs(sim)

                    if denominator > 1e-8:
                        pred = self.user_means_[u] + (numerator / denominator)
                        predictions[u, i] = np.clip(pred, 1, 5) # Clip to valid rating range
                    else:
                        # Fallback to user's mean if no neighbors rated the item
                        predictions[u, i] = self.user_means_[u]

        return predictions

# --- Step 5: Train the Model and Evaluate ---
print("\nTraining UserKNN model on the DENSE SUBSET...")
# Using adjusted_cosine as it generally performs better by accounting for rating biases
user_knn = UserKNNRecommender(k=20, similarity_metric='adjusted_cosine')
user_knn.fit(R_train)
print("Training complete.")

print("\nGenerating predictions for the test set...")
# Predictions are generated for all items, then we'll filter to the test set
predictions_matrix = user_knn.predict()

# Extract only the predictions for items in the test set
test_indices = R_test.nonzero()
actual_ratings = R_test[test_indices]
predicted_ratings = predictions_matrix[test_indices]

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
mae = mean_absolute_error(actual_ratings, predicted_ratings)

print("\n--- Evaluation Metrics on DENSE Test Set ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE):     {mae:.4f}")