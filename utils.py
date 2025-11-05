import datetime
import glob
import os
import shutil
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error
)
import requests
import zipfile
import io

@st.cache_data
def download_and_load_movielens():
    url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    try:
        response = requests.get(url)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open('ml-latest-small/ratings.csv') as f:
                ratings_df = pd.read_csv(f)
            with z.open('ml-latest-small/movies.csv') as f:
                movies_df = pd.read_csv(f)

    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading data: {e}")
        return None, None

    ratings_pivot_df = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    movies_df.set_index('movieId', inplace=True)
    return ratings_pivot_df, movies_df

@st.cache_data
def load_synthetic_data():
    """
    Loads the 20x20 synthetic dataset, including ratings, movies, and user profiles.
    """
    try:
        ratings_path = "datasets/20x20-synthetic/ratings.csv"
        movies_path = "datasets/20x20-synthetic/movies.csv"
        user_profiles_path = "datasets/20x20-synthetic/ground_truth_user_profiles.csv"

        ratings_df = pd.read_csv(ratings_path)
        movies_df = pd.read_csv(movies_path)
        user_profiles_df = pd.read_csv(user_profiles_path, index_col=0)

    except FileNotFoundError as e:
        st.error(f"Could not find a synthetic dataset file: {e.fileName}. Please ensure all files are in the `datasets/20x20-synthetic/` directory.")
        return None, None, None

    # Pivot ratings to create the user-item matrix
    ratings_pivot_df = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    movies_df.set_index('movieId', inplace=True)

    return ratings_pivot_df, movies_df, user_profiles_df


def split_data(df):
    train_df = df.copy()
    test_df = pd.DataFrame(np.zeros(df.shape), columns=df.columns, index=df.index)

    for user in range(df.shape[0]):
        user_ratings_indices = df.iloc[user, :].to_numpy().nonzero()[0]
        if len(user_ratings_indices) > 1:
            test_indices = np.random.choice(
                user_ratings_indices, size=int(len(user_ratings_indices) * 0.2), replace=False
            )
            if len(test_indices) == 0 and len(user_ratings_indices) > 1:
                test_indices = np.random.choice(user_ratings_indices, size=1, replace=False)

            train_df.iloc[user, test_indices] = 0
            test_df.iloc[user, test_indices] = df.iloc[user, test_indices]

    return train_df, test_df


def calculate_regression_metrics(predicted_df, test_df):
    test_indices = test_df.to_numpy().nonzero()
    preds = predicted_df.to_numpy()[test_indices]
    actuals = test_df.to_numpy()[test_indices]
    if actuals.size == 0:
        return {'rmse': 0, 'mae': 0, 'r2': 0, 'mape': 0, 'explained_variance': 0}
    epsilon = 1e-8
    actuals_for_mape = actuals + epsilon
    return {
        'rmse': np.sqrt(mean_squared_error(actuals, preds)),
        'mae': mean_absolute_error(actuals, preds),
        'r2': r2_score(actuals, preds),
        'mape': np.mean(np.abs((actuals - preds) / actuals_for_mape)) * 100,
        'explained_variance': explained_variance_score(actuals, preds)
    }

def precision_recall_at_k(predicted_scores_df, test_df, train_df, k=10):
    precisions = []
    recalls = []
    for user_id in test_df.index:
        test_ratings = test_df.loc[user_id]
        relevant_items = set(test_ratings[test_ratings > 0].index)
        if not relevant_items:
            continue
        predicted_scores = predicted_scores_df.loc[user_id]
        train_ratings_row = train_df.loc[user_id]
        seen_items = set(train_ratings_row[train_ratings_row > 0].index)
        scores_for_unseen = predicted_scores.drop(seen_items, errors='ignore')

        top_k_items = set(scores_for_unseen.nlargest(k).index)

        hits = len(top_k_items.intersection(relevant_items))

        precision = hits / k if k > 0 else 0
        recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
    return np.mean(precisions), np.mean(recalls)

def cleanup_old_reports(max_age_hours=24):
    """
    Deletes report and visualization directories older than max_age_hours.
    """
    st.toast(f"Running cleanup for files older than {max_age_hours} hours...")
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    deleted_count = 0

    # Directories to scan
    report_dirs = glob.glob('reports/report-*')
    viz_dirs = glob.glob('visuals/*/*')

    all_dirs = report_dirs + viz_dirs

    if not all_dirs:
        return 0

    for dir_path in all_dirs:
        try:
            # Check modification time
            mod_time = os.path.getmtime(dir_path)
            if datetime.fromtimestamp(mod_time) < cutoff:
                if os.path.isdir(dir_path):
                    shutil.rmtree(dir_path)
                    deleted_count += 1
        except FileNotFoundError:
            pass # Already deleted
        except Exception as e:
            # This might fail on Windows if a file is in use
            print(f"Could not delete {dir_path}: {e}")

    return deleted_count

@st.cache_data
def load_generated_data(dataset_name):
    """
    Loads a specific generated dataset from datasets/generated/[dataset_name]
    """
    base_path = os.path.join("datasets", "generated", dataset_name)
    try:
        ratings_path = os.path.join(base_path, "ratings.csv")
        movies_path = os.path.join(base_path, "movies.csv")
        user_profiles_path = os.path.join(base_path, "ground_truth_user_profiles.csv")
        item_features_path = os.path.join(base_path, "ground_truth_item_features.csv") # --- ADDED ---

        # Load ratings - it's already pivoted
        ratings_pivot_df = pd.read_csv(ratings_path, index_col=0)

        movies_df = pd.read_csv(movies_path, index_col=0)

        user_profiles_df = pd.read_csv(user_profiles_path, index_col=0)

        item_features_df = pd.read_csv(item_features_path, index_col=0) # --- ADDED ---

    except FileNotFoundError as e:
        st.error(f"Could not find dataset file: {e}. Please ensure '{dataset_name}' exists in `datasets/generated/`.")
        return None, None, None, None # --- MODIFIED ---
    except Exception as e:
        st.error(f"Error loading generated dataset '{dataset_name}': {e}")
        return None, None, None, None # --- MODIFIED ---

    return ratings_pivot_df, movies_df, user_profiles_df, item_features_df # --- MODIFIED ---

def get_generated_datasets():
    """
    Scans the datasets/generated/ directory and returns a list of dataset names.
    """
    base_dir = os.path.join("datasets", "generated")
    if not os.path.isdir(base_dir):
        return []

    try:
        dataset_names = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        return sorted(dataset_names)
    except Exception as e:
        print(f"Error scanning for generated datasets: {e}")
        return []