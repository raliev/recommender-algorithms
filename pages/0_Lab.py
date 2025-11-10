import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from ui_components import (
    render_sidebar,
    render_results_tabs,
    render_performance_tab,
    render_visualizations_tab,
    render_previous_runs_explorer
)
from algorithm_config import ALGORITHM_CONFIG
from utils import (
    download_and_load_movielens,
    load_synthetic_data,
    load_generated_data,
    get_generated_datasets,
    split_data,
    calculate_regression_metrics,
    precision_recall_at_k
)
from algorithms import *

from utils import split_data_leave_one_out

from utils import evaluate_ranking_loo

st.set_page_config(page_title="Recommender System Lab", layout="wide")
st.title("Recommender System Laboratory")

if 'results' not in st.session_state: st.session_state['results'] = None
if 'metrics' not in st.session_state: st.session_state['metrics'] = None
if 'movie_titles' not in st.session_state: st.session_state['movie_titles'] = None
if 'user_profiles' not in st.session_state: st.session_state['user_profiles'] = None
if 'selected_visuals_run_dir' not in st.session_state:
    st.session_state.selected_visuals_run_dir = None
if 'Q_matrix' not in st.session_state: st.session_state['Q_matrix'] = None

data_source, data, algorithm, model_params, data_params, run_button, show_internals = render_sidebar()

# This will display the selectbox and update st.session_state.selected_visuals_run_dir
render_previous_runs_explorer(algorithm, 'visuals')

if run_button:
    data_to_use = None
    data_to_train = None
    test_df_8020 = None
    test_df_loo = None
    train_df_8020 = None
    train_df_loo = None

    generated_dataset_names = get_generated_datasets()
    with st.spinner("Preparing data..."):

        if data_source == "MovieLens (Filtered Subset)":
            full_df, movie_titles_df = download_and_load_movielens()
            st.session_state['movie_titles'] = movie_titles_df
            st.session_state['user_profiles'] = None
            st.session_state['Q_matrix'] = None
            if full_df is not None:
                num_users = data_params.get('num_users', 100)
                num_movies = data_params.get('num_movies', 500)
                movie_counts = full_df[full_df > 0].count(axis=0)
                top_movies_ids = movie_counts.nlargest(num_movies).index
                df_filtered_movies = full_df[top_movies_ids]
                user_counts = df_filtered_movies[df_filtered_movies > 0].count(axis=1)
                top_users_ids = user_counts.nlargest(num_users).index
                data_to_use = df_filtered_movies.loc[top_users_ids]
                st.write(f"Using a subset of the data: **{data_to_use.shape[0]} users** and **{data_to_use.shape[1]} items**.")

        elif data_source == "MovieLens (Full Dataset)":
            full_df, movie_titles_df = download_and_load_movielens()
            st.session_state['movie_titles'] = movie_titles_df
            st.session_state['user_profiles'] = None
            st.session_state['Q_matrix'] = None
            if full_df is not None:
                data_to_use = full_df
                st.warning("️️[!] You are using the full dataset. Calculations may be very slow...")
                st.write(f"Using the full dataset: **{data_to_use.shape[0]} users** and **{data_to_use.shape[1]} movies**.")

        elif data_source in generated_dataset_names:
            full_df, movie_titles_df, user_profiles_df, q_matrix_df = load_generated_data(data_source)
            if full_df is not None:
                st.session_state['movie_titles'] = movie_titles_df
                st.session_state['user_profiles'] = user_profiles_df
                st.session_state['Q_matrix'] = q_matrix_df
                data_to_use = full_df
                st.write(f"Using generated dataset '{data_source}': **{data_to_use.shape[0]} users** and **{data_to_use.shape[1]} movies**.")

        elif data_source == "Synthetic 20x20":
            full_df, movie_titles_df, user_profiles_df = load_synthetic_data()
            if full_df is not None:
                st.session_state['movie_titles'] = movie_titles_df
                st.session_state['user_profiles'] = user_profiles_df
                st.session_state['Q_matrix'] = None
                data_to_use = full_df
                st.write(f"Using the synthetic dataset: **{data_to_use.shape[0]} users** and **{data_to_use.shape[1]} movies**.")

    if data_to_use is not None:
        train_df_8020, test_df_8020 = split_data(data_to_use)
        train_df_loo, test_df_loo = split_data_leave_one_out(data_to_use)

        data_to_train = train_df_8020.to_numpy()

        progress_bar = st.progress(0, text=f"Training {algorithm} model...")

        algo_config_all = ALGORITHM_CONFIG.get(algorithm, {})
        model_class = algo_config_all["model_class"]

        if model_class:
            import inspect
            sig = inspect.signature(model_class.__init__)
            algo_specific_params = {k: v for k, v in model_params.items() if k in sig.parameters}
            if 'k' in sig.parameters and 'k' in ALGORITHM_CONFIG[algorithm].get("parameters", {}):
                algo_specific_params['k'] = model_params.get('k')

            model = model_class(**algo_specific_params)
        else:
            st.error(f"Algorithm {algorithm} not found."); st.stop()

        visualizer = None
        params_to_save = None
        if show_internals:
            VisClass = ALGORITHM_CONFIG[algorithm].get("visualizer_class")
            if VisClass:
                import inspect
                vis_sig = inspect.signature(VisClass.__init__)
                vis_args = {}
                if 'k_factors' in vis_sig.parameters:
                    vis_args['k_factors'] = model_params.get('k', 0)
                if 'k' in vis_sig.parameters:
                    vis_args['k'] = model_params.get('k', 10)
                visualizer = VisClass(**vis_args)
                params_to_save = {
                    **model_params,
                    'algorithm': algorithm,
                    'data_source': data_source,
                    'data_params': data_params
                }

        try:
            model.train_data = data_to_train
            model.fit(data_to_train,
                      progress_callback=lambda p: progress_bar.progress(p, text=f"Training {algorithm} model... {int(p*100)}%"),
                      visualizer=visualizer,
                      params_to_save=params_to_save)
        except ImportError as e:
            st.error(f"Could not run {algorithm}. Please make sure required libraries are installed: {e}")
            st.stop()
        except Exception as e:
            st.error(f"An error occurred during model training or prediction: {e}")
            import traceback
            st.error(traceback.format_exc())
            st.stop()

        progress_bar.empty()
        predicted_matrix = model.predict()
        predicted_matrix = np.nan_to_num(predicted_matrix)
        predicted_df = pd.DataFrame(predicted_matrix, index=data_to_use.index, columns=data_to_use.columns)

        st.session_state['results'] = {
            'algo_name': model.name,
            'predicted_df': predicted_df,
            'original_df': data_to_use,
            'P': getattr(model, 'P', None),
            'Q': getattr(model, 'Q', None),
            'sigma': getattr(model, 'sigma', None),
            'similarity_matrix': getattr(model, 'similarity_matrix', None),
            'reconstructed_matrix': getattr(model, 'reconstructed_matrix', None),
            'visuals_dir': visualizer.get_run_directory() if visualizer else None,
            'visuals_base_dir': visualizer.get_base_directory() if visualizer else 'visuals',
            'model_params': model_params
        }

        st.session_state['metrics'] = None

        algo_config_all = ALGORITHM_CONFIG.get(algorithm, {})
        is_implicit_model = algo_config_all.get("is_implicit", False)

        if is_implicit_model:
            k_prec_rec = 10
            metrics_8020 = {}
            metrics_loo = {}

            precision, recall = precision_recall_at_k(predicted_df, test_df_8020, train_df_8020, k=k_prec_rec)
            metrics_8020 = {'type': 'implicit_holdout', 'precision': precision, 'recall': recall, 'k': k_prec_rec}

            loo_metrics_dict = evaluate_ranking_loo(predicted_df, test_df_loo, train_df_loo, k=k_prec_rec)
            metrics_loo = {'type': 'implicit_loo', 'k': k_prec_rec, **loo_metrics_dict}

            st.session_state['metrics'] = {
                'holdout': metrics_8020,
                'loo': metrics_loo
            }
        else:
            if test_df_8020 is not None:
                metrics = calculate_regression_metrics(predicted_df, test_df_8020)
                st.session_state['metrics'] = {'type': 'explicit', **metrics}

        if visualizer and st.session_state['metrics']:
            visuals_dir = st.session_state['results'].get('visuals_dir')
            if visuals_dir and os.path.isdir(visuals_dir):
                metrics_path = os.path.join(visuals_dir, 'metrics.json')
                try:
                    # Save the new structured metrics
                    with open(metrics_path, 'w') as f:
                        json.dump(st.session_state['metrics'], f, indent=4)
                except Exception as e:
                    st.toast(f"Failed to save metrics: {e}")

        st.session_state.selected_visuals_run_dir = st.session_state['results']['visuals_dir']
        st.rerun()

if st.session_state.get('results') and st.session_state.selected_visuals_run_dir == st.session_state['results'].get('visuals_dir'):
    results_with_titles = {
        **st.session_state['results'],
        'movie_titles': st.session_state.get('movie_titles'),
        'user_profiles': st.session_state.get('user_profiles'),
        'Q_matrix': st.session_state.get('Q_matrix')
    }

    if st.session_state['metrics']:
        main_tabs = st.tabs(["Results", "Performance", "Visualizations"])
        with main_tabs[0]:
            render_results_tabs(results_with_titles)
        with main_tabs[1]:
            render_performance_tab(st.session_state['metrics'])
        with main_tabs[2]:
            # Pass the specific visuals_dir from the new run
            render_visualizations_tab(results_with_titles, st.session_state['results']['visuals_dir'])
    else:
        main_tabs = st.tabs(["Results", "Visualizations"])
        with main_tabs[0]:
            render_results_tabs(results_with_titles)
        with main_tabs[1]:
            # Pass the specific visuals_dir from the new run
            render_visualizations_tab(results_with_titles, st.session_state['results']['visuals_dir'])

elif st.session_state.get('selected_visuals_run_dir'):
    st.header("Viewing Previous Run")
    st.divider()
    selected_dir = st.session_state.selected_visuals_run_dir
    params_str = "Params not found."
    params_path = os.path.join(selected_dir, 'params.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
            params_str = ", ".join(f"{k}={v}" for k, v in params.items() if k not in ['algorithm', 'timestamp'])

    st.info(f"**Algorithm:** `{algorithm}`  \n**Run:** `{os.path.basename(selected_dir)}`  \n**Params:** `{params_str}`")

    metrics_path = os.path.join(selected_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            # Use the existing performance tab renderer
            render_performance_tab(metrics)
        except Exception as e:
            st.error(f"Error loading metrics.json: {e}")
    else:
        st.info("No performance metrics (metrics.json) were saved for this run.")

    # Manually build minimal info for render_visualizations_tab
    results_for_viz = {
        'algo_name': algorithm,
        'visuals_base_dir': 'visuals'
    }

    # Render the visualizations tab
    render_visualizations_tab(results_for_viz, selected_dir)
else:
    st.info("Select your data, algorithm, and parameters in the sidebar, then click 'Run'.\n\nOr, select a previous run from the 'Explore Previous Runs' section.")