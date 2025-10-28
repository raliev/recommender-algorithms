import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.exceptions import ConvergenceWarning
import requests
import zipfile
import io
import warnings
import sys
from collections import defaultdict

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
    except requests.RequestException as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)

try:
    ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
except FileNotFoundError:
    download_movielens_100k()
    ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

print("Data loaded successfully.")

# --- Step 2: Preprocess Data for Implicit Feedback Task ---

# Convert explicit ratings to implicit feedback
# We consider ratings of 4 or 5 as a positive interaction (1)
print("Converting ratings to implicit feedback (ratings >= 4 are considered positive)...")
implicit_df = ratings_df[ratings_df['rating'] >= 4].copy()
implicit_df['interaction'] = 1

# Create user-item interaction matrix from implicit data
all_user_ids = sorted(ratings_df['user_id'].unique())
all_movie_ids = sorted(ratings_df['movie_id'].unique())
num_users = len(all_user_ids)
num_items = len(all_movie_ids)

user_map = {uid: i for i, uid in enumerate(all_user_ids)}
movie_map = {mid: i for i, mid in enumerate(all_movie_ids)}

R_implicit = np.zeros((num_users, num_items))
for _, row in implicit_df.iterrows():
    if row['user_id'] in user_map and row['movie_id'] in movie_map:
        user_idx = user_map[row['user_id']]
        movie_idx = movie_map[row['movie_id']]
        R_implicit[user_idx, movie_idx] = 1

def train_test_split_matrix(ratings_matrix, test_size=0.2, random_state=42):
    """Splits an implicit interaction matrix into training and testing sets."""
    rng = np.random.default_rng(random_state)
    test = np.zeros(ratings_matrix.shape)
    train = ratings_matrix.copy()

    for user in range(ratings_matrix.shape[0]):
        interactions = ratings_matrix[user, :].nonzero()[0]
        if len(interactions) > 1:
            test_indices = rng.choice(
                interactions,
                size=int(len(interactions) * test_size),
                replace=False
            )
            train[user, test_indices] = 0.
            test[user, test_indices] = 1.

    assert(np.all((train * test) == 0))
    return train, test

print("Splitting implicit data into training and test sets (80/20)...")
R_train, R_test = train_test_split_matrix(R_implicit, test_size=0.2, random_state=42)
print(f"Training set contains {int(R_train.sum())} interactions.")
print(f"Test set contains {int(R_test.sum())} interactions.")


# --- Step 3: Define SLIMRecommender Class ---
class SLIMRecommender:
    def __init__(self, l1_reg=0.001, l2_reg=0.0001):
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.similarity_matrix_ = None

    def fit(self, R):
        num_items = R.shape[1]
        self.similarity_matrix_ = np.zeros((num_items, num_items))

        alpha = self.l1_reg + self.l2_reg
        l1_ratio = self.l1_reg / alpha if alpha > 0 else 0.0

        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            positive=True,
            fit_intercept=False,
            copy_X=False,
            max_iter=200,
            tol=1e-4
        )
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        for j in range(num_items):
            y = R[:, j]
            X = np.delete(R, j, axis=1)
            model.fit(X, y)
            self.similarity_matrix_[j, :] = np.insert(model.coef_, j, 0)

            sys.stdout.write(f"\rTraining SLIM: Item {j+1}/{num_items}")
            sys.stdout.flush()

        warnings.resetwarnings()
        print("\nTraining complete.")
        return self

    def predict(self, R_input):
        """Generates ranking scores for all items, excluding already seen ones."""
        R_scores = R_input @ self.similarity_matrix_
        R_scores[R_input.nonzero()] = 0 # Exclude already interacted items
        return R_scores

# --- Step 4: Define Evaluation Metrics for Ranking ---
def evaluate_ranking(model, R_train, R_test, k=10):
    """Calculates Precision@k and Recall@k for a trained model."""
    print(f"\nEvaluating Precision@{k} and Recall@{k}...")

    R_scores = model.predict(R_train)

    precisions = []
    recalls = []

    num_users_with_test_items = 0

    for user_idx in range(R_train.shape[0]):
        true_test_items = R_test[user_idx, :].nonzero()[0]

        # Skip users with no items in the test set
        if len(true_test_items) == 0:
            continue

        num_users_with_test_items += 1

        user_scores = R_scores[user_idx, :]
        top_k_items = np.argsort(-user_scores)[:k]

        hits = np.isin(top_k_items, true_test_items).sum()

        precision = hits / k
        recall = hits / len(true_test_items)

        precisions.append(precision)
        recalls.append(recall)

    mean_precision = np.mean(precisions) if precisions else 0
    mean_recall = np.mean(recalls) if recalls else 0

    print(f"Evaluated on {num_users_with_test_items} users with items in the test set.")
    return mean_precision, mean_recall


# --- Step 5: Train the Model and Evaluate ---
print("\nTraining SLIM model on implicit data...")
slim_rec = SLIMRecommender(l1_reg=0.005, l2_reg=0.005)
slim_rec.fit(R_train)

# Evaluate using ranking metrics
precision_at_10, recall_at_10 = evaluate_ranking(slim_rec, R_train, R_test, k=10)

print("\n--- Evaluation Metrics for Ranking ---")
print(f"Precision@10: {precision_at_10:.4f}")
print(f"Recall@10:    {recall_at_10:.4f}")