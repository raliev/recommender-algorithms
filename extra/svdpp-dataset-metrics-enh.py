import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import requests
import zipfile
import io
import sys

# --- Step 1: Download and Load Data ---
def download_movielens_100k():
    """Downloads and extracts the MovieLens 100K dataset."""
    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    print("Downloading MovieLens 100K dataset...")
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
        print("Dataset downloaded and extracted to 'ml-100k' directory.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)

try:
    ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
except FileNotFoundError:
    download_movielens_100k()
    ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

print("Data loaded successfully.")

# --- Step 2: Split Data into Training and Test Sets ---
print("Splitting data into training and test sets (80/20)...")
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")


# --- Step 3: Prepare User-Item Matrices ---
print("Preparing user-item matrices for train and test sets...")
# Get all unique users and movies from the original dataset for consistent matrix shapes
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


# --- Step 4: Define the SVDppRecommender Class ---
class SVDppRecommender:
    def __init__(self, k, iterations=20, learning_rate=0.005, lambda_reg=0.02):
        self.k = k
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None
        self.P = None # User factors
        self.Q = None # Item factors

    def fit(self, R):
        num_users, num_items = R.shape
        self.P = np.random.normal(scale=1./self.k, size=(num_users, self.k))
        self.Q = np.random.normal(scale=1./self.k, size=(num_items, self.k))
        self.user_bias = np.zeros(num_users)
        self.item_bias = np.zeros(num_items)
        self.global_mean = R[R > 0].mean()

        # Get the non-zero indices once to iterate over
        training_indices = np.argwhere(R > 0)

        for i in range(self.iterations):
            np.random.shuffle(training_indices) # Shuffle for stochasticity
            for u, item_idx in training_indices:
                # Predict the rating
                pred = (self.global_mean + self.user_bias[u] +
                        self.item_bias[item_idx] +
                        self.P[u, :] @ self.Q[item_idx, :].T)
                error = R[u, item_idx] - pred

                # Update biases and factors
                bu = self.user_bias[u]
                bi = self.item_bias[item_idx]
                pu = self.P[u, :]
                qi = self.Q[item_idx, :]

                self.user_bias[u] += self.learning_rate * (error - self.lambda_reg * bu)
                self.item_bias[item_idx] += self.learning_rate * (error - self.lambda_reg * bi)
                self.P[u, :] += self.learning_rate * (error * qi - self.lambda_reg * pu)
                self.Q[item_idx, :] += self.learning_rate * (error * pu - self.lambda_reg * qi)

            sys.stdout.write(f"\rTraining SVD++: Iteration {i+1}/{self.iterations}")
            sys.stdout.flush()
        print("\nTraining complete.")
        return self

    def predict(self):
        user_bias_matrix = np.repeat(self.user_bias[:, np.newaxis], self.Q.shape[0], axis=1)
        item_bias_matrix = np.repeat(self.item_bias[np.newaxis, :], self.P.shape[0], axis=0)

        return (self.global_mean + user_bias_matrix +
                item_bias_matrix + (self.P @ self.Q.T))

# --- Step 5: Train the Model and Evaluate ---
print("\nTraining SVD++ model on the training set...")
svdpp_rec = SVDppRecommender(k=30, iterations=25, learning_rate=0.007, lambda_reg=0.05)
svdpp_rec.fit(R_train)

print("\nGenerating predictions for the test set...")
predictions_matrix = svdpp_rec.predict()

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