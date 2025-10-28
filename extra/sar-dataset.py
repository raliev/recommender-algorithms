import numpy as np
import pandas as pd
import requests
import zipfile
import io
from collections import defaultdict

# --- Step 1: Download and Load Data ---
# (This part is the same as before)
def download_movielens_100k():
    """Downloads and extracts the MovieLens 100K dataset."""
    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    print("Downloading MovieLens 100K dataset...")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    print("Extracting files...")
    z.extractall()
    print("Dataset downloaded and extracted to 'ml-100k' directory.")

try:
    ratings_df = pd.read_csv(
        'ml-100k/u.data', sep='\t',
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
    movies_df = pd.read_csv(
        'ml-100k/u.item', sep='|', encoding='latin-1', header=None,
        names=['movie_id', 'title'] + [f'col{i}' for i in range(22)]
    )[['movie_id', 'title']]
except FileNotFoundError:
    download_movielens_100k()
    ratings_df = pd.read_csv(
        'ml-100k/u.data', sep='\t',
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
    movies_df = pd.read_csv(
        'ml-100k/u.item', sep='|', encoding='latin-1', header=None,
        names=['movie_id', 'title'] + [f'col{i}' for i in range(22)]
    )[['movie_id', 'title']]

# --- Step 2: Create Implicit Feedback Data ---
# Convert ratings >= 4 to positive interactions (1)
implicit_df = ratings_df[ratings_df['rating'] >= 4].copy()
implicit_df['interacted'] = 1

# --- Step 3: Train-Test Split ---
# We need to split the data to evaluate the model properly.
# For each user, we'll use some interactions for training and some for testing.
def train_test_split_by_user(df, test_size=0.2):
    """Splits interactions for each user into train and test sets."""
    train_list = []
    test_list = []
    for user_id, group in df.groupby('user_id'):
        n_test = int(len(group) * test_size)
        if n_test > 0:
            test_indices = np.random.choice(group.index, size=n_test, replace=False)
            test_set = group.loc[test_indices]
            train_set = group.drop(test_indices)
            train_list.append(train_set)
            test_list.append(test_set)
        else: # If user has too few interactions, use all for training
            train_list.append(group)

    return pd.concat(train_list), pd.concat(test_list)

print("Splitting data into training and testing sets...")
train_df, test_df = train_test_split_by_user(implicit_df, test_size=0.2)
print(f"Training set size: {len(train_df)}")
print(f"Testing set size: {len(test_df)}")


# --- Step 4: Prepare User-Item Matrix for Training ---
# Create the matrix ONLY from the training data
all_users = sorted(implicit_df['user_id'].unique())
all_items = sorted(implicit_df['movie_id'].unique())
user_map = {user_id: i for i, user_id in enumerate(all_users)}
item_map = {movie_id: i for i, movie_id in enumerate(all_items)}

R_train = np.zeros((len(all_users), len(all_items)))
for _, row in train_df.iterrows():
    if row['user_id'] in user_map and row['movie_id'] in item_map:
        R_train[user_map[row['user_id']], item_map[row['movie_id']]] = 1

print(f"Created training matrix with shape: {R_train.shape}")


# --- Step 5: Define SARRecommender Class ---
# (This class is the same as before)
class SARRecommender:
    def __init__(self, threshold=2):
        self.threshold = threshold
        self.similarity_matrix_ = None

    def fit(self, R):
        print("Fitting the SAR model...")
        co_occurrence_matrix = R.T @ R
        item_counts = np.diag(co_occurrence_matrix)
        denominator = (item_counts[:, np.newaxis] + item_counts -
                       co_occurrence_matrix)
        denominator[denominator == 0] = 1e-8
        similarity_matrix = co_occurrence_matrix / denominator
        similarity_matrix[co_occurrence_matrix < self.threshold] = 0
        np.fill_diagonal(similarity_matrix, 0)
        self.similarity_matrix_ = similarity_matrix
        print("Fit complete.")
        return self

    def predict(self, R_input):
        return R_input @ self.similarity_matrix_

# --- Step 6: Train Model and Generate Scores ---
sar = SARRecommender(threshold=5)
sar.fit(R_train)
# Scores are predicted based on training data interactions
R_scores = sar.predict(R_train)


# --- Step 7: Define and Calculate Evaluation Metrics ---
def precision_at_k(recommended_items, relevant_items, k):
    """Calculates Precision@k."""
    intersection = len(set(recommended_items[:k]) & set(relevant_items))
    return intersection / k

def recall_at_k(recommended_items, relevant_items, k):
    """Calculates Recall@k."""
    intersection = len(set(recommended_items[:k]) & set(relevant_items))
    return intersection / len(relevant_items) if relevant_items else 0

def evaluate_model(scores_matrix, test_data, train_data, user_map, item_map, k=10):
    """Calculates average Precision@k and Recall@k for all test users."""
    total_precision = 0
    total_recall = 0
    num_users = 0

    # Group test data by user to get relevant items for each
    test_user_items = test_data.groupby('user_id')['movie_id'].apply(list).to_dict()

    for user_id, relevant_items in test_user_items.items():
        if user_id not in user_map: continue

        user_idx = user_map[user_id]
        user_scores = scores_matrix[user_idx]

        # Get top-k recommendations
        top_item_indices = np.argsort(-user_scores)

        # Map indices back to movie_ids
        # Need to create an inverse map for this
        idx_to_item = {i: item_id for item_id, i in item_map.items()}
        recommended_items = [idx_to_item[i] for i in top_item_indices]

        total_precision += precision_at_k(recommended_items, relevant_items, k)
        total_recall += recall_at_k(recommended_items, relevant_items, k)
        num_users += 1

    avg_precision = total_precision / num_users if num_users > 0 else 0
    avg_recall = total_recall / num_users if num_users > 0 else 0

    return avg_precision, avg_recall

print("\n--- Evaluating Model Performance ---")
K = 10
avg_precision, avg_recall = evaluate_model(R_scores, test_df, train_df, user_map, item_map, k=K)

print(f"Average Precision@{K}: {avg_precision:.4f}")
print(f"Average Recall@{K}:    {avg_recall:.4f}")


# --- Step 8: Get Example Recommendations (Qualitative Check) ---
def get_recommendations(user_id, scores_matrix, movies_info_df, user_map, item_map, top_n=10):
    """Formats and prints top-N recommendations for a single user."""
    if user_id not in user_map:
        print(f"User {user_id} not in the model.")
        return

    user_idx = user_map[user_id]
    user_scores = scores_matrix[user_idx]

    top_item_indices = np.argsort(-user_scores)[:top_n]

    idx_to_item = {i: item_id for item_id, i in item_map.items()}
    recommended_movie_ids = [idx_to_item[i] for i in top_item_indices]

    recs_df = movies_info_df[movies_info_df['movie_id'].isin(recommended_movie_ids)]

    print(f"\n--- Top-{top_n} Recommendations for User {user_id} ---")
    print(recs_df[['title']].to_string(index=False))

# Get qualitative recommendations for the same example user
example_user_id = 50
get_recommendations(example_user_id, R_scores, movies_df, user_map, item_map)