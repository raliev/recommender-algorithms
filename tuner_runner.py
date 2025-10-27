import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
import itertools
import random
import matplotlib.pyplot as plt
import seaborn as sns
import optuna # Import Optuna

from utils import download_and_load_movielens, load_synthetic_data, split_data, calculate_regression_metrics, precision_recall_at_k
from algorithms import *

def run_tuning_process(configured_params, data_source, data_params, n_trials):
    st.header("Running Experiments...")

    # --- Setup Report Directory ---
    base_report_dir = "reports"
    report_subdir_name = f"report-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    report_dir = os.path.join(base_report_dir, report_subdir_name)
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "full_report.csv")
    best_hp_path = os.path.join(report_dir, "best-hp.csv")

    # Prepare Data ---
    with st.spinner("Preparing data..."):
        if data_source == "Load MovieLens CSV":
            full_df, _ = download_and_load_movielens()
            if full_df is not None:
                num_users = data_params.get('num_users', 100)
                num_movies = data_params.get('num_movies', 500)
                movie_counts = full_df[full_df > 0].count(axis=0)
                top_movies_ids = movie_counts.nlargest(num_movies).index
                df_filtered_movies = full_df[top_movies_ids]
                user_counts = df_filtered_movies[df_filtered_movies > 0].count(axis=1)
                top_users_ids = user_counts.nlargest(num_users).index
                data_to_use = df_filtered_movies.loc[top_users_ids]
        elif data_source == "Synthetic 20x20":
            data_to_use, _, _ = load_synthetic_data()

        if data_to_use is None:
            st.error("Failed to load data."); st.stop()
        train_df, test_df = split_data(data_to_use)
        data_to_train = train_df.to_numpy()
    st.success(f"Data prepared: {data_to_use.shape[0]} users, {data_to_use.shape[1]} movies.")

    # UI Placeholders ---
    st.subheader("Progress")
    overall_progress = st.progress(0)
    overall_text = st.empty()

    st.subheader("Best Result for Current Algorithm")
    best_result_placeholder = st.empty()

    # Main Loop ---
    all_results = []
    best_results_per_algo = []
    completed_iterations = 0
    total_iterations = len(configured_params) * n_trials

    model_map = {
        "SVD": SVDRecommender, "ALS": ALSRecommender, "ALS (Improved)": ALSImprovedRecommender,
        "ALS (PySpark)": ALSPySparkRecommender,
        "BPR": BPRRecommender, "ItemKNN": ItemKNNRecommender, "UserKNN": UserKNNRecommender,
        "NMF": NMFRecommender, "FunkSVD": FunkSVDRecommender, "SVD++": SVDppRecommender, "WRMF": WRMFRecommender,
    }

    for algo_idx, (algo_name, params) in enumerate(configured_params.items()):

        st.subheader(f"Tuning {algo_name}...")
        algo_progress_bar = st.progress(0)

        # Optuna Objective Function
        def objective(trial):
            nonlocal completed_iterations # Allow modification of the outer scope variable
            model_params = {}
            for param_name, config in params.items():
                if "options" in config: # Categorical
                    model_params[param_name] = trial.suggest_categorical(param_name, config)
                else: # Numerical
                    if config['is_int']:
                        model_params[param_name] = trial.suggest_int(param_name, config['low'], config['high'])
                    else:
                        model_params[param_name] = trial.suggest_float(param_name, config['low'], config['high'])

            start_time = time.time()
            try:
                model_class = model_map.get(algo_name)
                if not model_class:
                    st.warning(f"Algorithm {algo_name} not found. Skipping.")
                    return float('inf') # Return a high value for minimization

                model = model_class(**model_params)
                model.fit(data_to_train)
                predicted_matrix = model.predict()
                predicted_df = pd.DataFrame(predicted_matrix, index=data_to_use.index, columns=data_to_use.columns)

                if algo_name in ["BPR", "WRMF"]:
                    metrics = {'type': 'implicit', **precision_recall_at_k(predicted_df, test_df)}
                    primary_metric = 'precision'
                    value_to_optimize = -metrics[primary_metric] # Optuna minimizes, so negate precision
                else:
                    metrics = {'type': 'explicit', **calculate_regression_metrics(predicted_df, test_df)}
                    primary_metric = 'rmse'
                    value_to_optimize = metrics[primary_metric]

                result = {"algorithm": algo_name, "runtime": time.time() - start_time, **model_params, **metrics}
                all_results.append(result)

                # --- Update UI ---
                completed_iterations += 1
                progress = completed_iterations / total_iterations
                overall_progress.progress(progress)
                overall_text.text(f"Overall Progress: {completed_iterations} / {total_iterations}")

                # Find and display the current best result for this algorithm
                current_best_trial = trial.study.best_trial
                with best_result_placeholder.container():
                    st.write(f"**Best {primary_metric} for {algo_name} so far:** {current_best_trial.value if primary_metric == 'rmse' else -current_best_trial.value:.4f}")
                    st.json(current_best_trial.params)

                return value_to_optimize

            except Exception as e:
                st.error(f"Error running {algo_name} with params {model_params}: {e}")
                return float('inf') # Error case

        # Create and Run Optuna Study
        is_minimizing = not (algo_name in ["BPR", "WRMF"]) # Precision needs maximization
        study = optuna.create_study(direction="minimize" if is_minimizing else "maximize")
        study.optimize(objective, n_trials=n_trials, callbacks=[lambda study, trial: algo_progress_bar.progress( (trial.number + 1) / n_trials )] )

        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value

        # Find the full result dictionary corresponding to the best trial
        best_run_data = {}
        for r in all_results:
            match = True
            for p_name, p_val in best_params.items():
                if r.get(p_name) != p_val:
                    match = False
                    break
            if match and r['algorithm'] == algo_name:
                best_run_data = r
                break

        if best_run_data:
            best_results_per_algo.append(best_run_data)

        # Save intermediate results
        pd.DataFrame(all_results).to_csv(report_path, index=False)
        pd.DataFrame(best_results_per_algo).to_csv(best_hp_path, index=False)

        best_result_placeholder.empty()

    st.balloons()
    st.success("Tuning process completed!")
    st.info(f"Reports saved to: {os.path.abspath(report_dir)}")