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
    try:
        r = requests.get(url)
        r.raise_for_status()
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
    ratings_df = pd.read_csv(
        'ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
except FileNotFoundError:
    download_movielens_100k()
    ratings_df = pd.read_csv(
        'ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp']
    )

# --- Step 2: Split Data into Training and Test Sets ---
print("Splitting data into training and test sets (80/20)...")
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

# --- Step 3: Prepare User-Item Matrices ---
print("Preparing user-item matrices for train and test sets...")
# Get all unique users and movies from the original dataset to ensure consistent matrix shapes
all_user_ids = sorted(ratings_df['user_id'].unique())
all_movie_ids = sorted(ratings_df['movie_id'].unique())
num_users = len(all_user_ids)
num_items = len(all_movie_ids)

# Create mappings
user_map = {uid: i for i, uid in enumerate(all_user_ids)}
movie_map = {mid: i for i, mid in enumerate(all_movie_ids)}

def create_rating_matrix(df, n_users, n_items, u_map, m_map):
    """Creates a user-item rating matrix from a dataframe."""
    R = np.zeros((n_users, n_items))
    for _, row in df.iterrows():
        user_idx = u_map[row['user_id']]
        movie_idx = m_map[row['movie_id']]
        R[user_idx, movie_idx] = row['rating']
    return R

R_train = create_rating_matrix(train_df, num_users, num_items, user_map, movie_map)
R_test = create_rating_matrix(test_df, num_users, num_items, user_map, movie_map)
print(f"Train matrix shape: {R_train.shape}, Test matrix shape: {R_test.shape}")


# --- Step 4: Define and Update the ItemKNNRecommender Class ---
class ItemKNNRecommender:
    def __init__(self, k=20, min_support=5, shrinkage=100.0):
        self.k = k
        self.min_support = min_support
        self.shrinkage = shrinkage
        self.similarity_matrix_ = None
        self.train_data_ = None

    def fit(self, R):
        self.train_data_ = R
        num_items = R.shape[1]

        user_means = np.true_divide(R.sum(axis=1), (R != 0).sum(axis=1))
        user_means[np.isnan(user_means)] = 0
        R_centered = R - np.where(R != 0, user_means[:, np.newaxis], 0)

        self.similarity_matrix_ = cosine_similarity(R_centered.T)

        co_rated_counts = (self.train_data_ > 0).astype(float).T @ (self.train_data_ > 0).astype(float)

        if self.min_support > 0:
            self.similarity_matrix_[co_rated_counts < self.min_support] = 0

        if self.shrinkage > 0:
            shrinkage_factor = co_rated_counts / (co_rated_counts + self.shrinkage)
            self.similarity_matrix_ *= shrinkage_factor

        np.fill_diagonal(self.similarity_matrix_, 0)
        return self

    def predict(self, R_input):
        """Generates a full matrix of predictions."""
        num_users, num_items = R_input.shape
        R_predicted = np.zeros_like(R_input, dtype=float)

        for user_idx in range(num_users):
            user_ratings = R_input[user_idx, :]
            rated_indices = np.where(user_ratings > 0)[0]

            for item_to_predict in range(num_items):
                # We can predict for all items, not just unrated ones, for evaluation purposes
                sims_to_rated_items = self.similarity_matrix_[item_to_predict, rated_indices]
                ratings_of_rated_items = user_ratings[rated_indices]

                if self.k > 0 and len(sims_to_rated_items) > self.k:
                    top_neighbors_indices = np.argsort(-np.abs(sims_to_rated_items))[:self.k]
                    sims_to_rated_items = sims_to_rated_items[top_neighbors_indices]
                    ratings_of_rated_items = ratings_of_rated_items[top_neighbors_indices]

                numerator = sims_to_rated_items @ ratings_of_rated_items
                denominator = np.sum(np.abs(sims_to_rated_items))

                if denominator > 1e-8:
                    R_predicted[user_idx, item_to_predict] = numerator / denominator

        return R_predicted

# --- Step 5: Train the Model and Evaluate ---
print("\nTraining ItemKNN model on the training set...")
item_knn = ItemKNNRecommender(k=40, min_support=2, shrinkage=0.0)
item_knn.fit(R_train)
print("Training complete.")

print("\nGenerating predictions for the test set...")
# The model uses the training data (R_train) to make predictions for test users
predictions_matrix = item_knn.predict(R_train)

# Get the indices of the ratings in the test set
test_indices = R_test.nonzero()
# Extract the actual ratings from the test set
actual_ratings = R_test[test_indices]
# Extract the corresponding predicted ratings
predicted_ratings = predictions_matrix[test_indices]

# Calculate metrics
rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
mae = mean_absolute_error(actual_ratings, predicted_ratings)

print("\n--- Evaluation Metrics on Test Set ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE):     {mae:.4f}")