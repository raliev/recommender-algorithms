# app/utils.py
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
    # ... (this function remains unchanged)
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


def precision_recall_at_k(predicted_scores_df, test_df, k=10):
    precisions = []
    recalls = []
    for user_id in test_df.index:
        test_ratings = test_df.loc[user_id]
        relevant_items = set(test_ratings[test_ratings > 0].index)
        if not relevant_items:
            continue
        predicted_scores = predicted_scores_df.loc[user_id]
        train_ratings = predicted_scores_df.columns.difference(test_ratings.index)
        predicted_scores = predicted_scores.drop(train_ratings, errors='ignore')
        top_k_items = set(predicted_scores.nlargest(k).index)
        hits = len(top_k_items.intersection(relevant_items))
        precision = hits / k if k > 0 else 0
        recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
    return np.mean(precisions), np.mean(recalls)