# pages/0_Lab.py
import streamlit as st
import pandas as pd
import numpy as np
from ui_components import render_sidebar, render_results_tabs, render_performance_tab, render_visualizations_tab
from algorithm_config import ALGORITHM_CONFIG # Import the main config
from utils import download_and_load_movielens, load_synthetic_data, split_data, calculate_regression_metrics, precision_recall_at_k #
from algorithms import *

st.set_page_config(page_title="Recommender System Lab", layout="wide") #
st.title("Recommender System Laboratory") #

if 'results' not in st.session_state: st.session_state['results'] = None #
if 'metrics' not in st.session_state: st.session_state['metrics'] = None #
if 'movie_titles' not in st.session_state: st.session_state['movie_titles'] = None #
if 'user_profiles' not in st.session_state: st.session_state['user_profiles'] = None #


data_source, data, algorithm, model_params, data_params, run_button, show_internals = render_sidebar() #

if run_button: #
    # Initialize data containers
    data_to_use = None #
    data_to_train = None #
    test_df = None #

    with st.spinner("Preparing data..."): #
        if data_source == "MovieLens (Filtered Subset)": #
            full_df, movie_titles_df = download_and_load_movielens() #
            st.session_state['movie_titles'] = movie_titles_df #
            st.session_state['user_profiles'] = None # No profiles for MovieLens #
            if full_df is not None: #
                num_users = data_params.get('num_users', 100) #
                num_movies = data_params.get('num_movies', 500) #
                movie_counts = full_df[full_df > 0].count(axis=0) #
                top_movies_ids = movie_counts.nlargest(num_movies).index #
                df_filtered_movies = full_df[top_movies_ids] #
                user_counts = df_filtered_movies[df_filtered_movies > 0].count(axis=1) #
                top_users_ids = user_counts.nlargest(num_users).index #
                data_to_use = df_filtered_movies.loc[top_users_ids] #
                st.write(f"Using a subset of the data: **{data_to_use.shape[0]} users** and **{data_to_use.shape[1]} movies**.") #
                train_df, test_df = split_data(data_to_use) #
                data_to_train = train_df.to_numpy() #

        elif data_source == "MovieLens (Full Dataset)": #
            full_df, movie_titles_df = download_and_load_movielens() #
            st.session_state['movie_titles'] = movie_titles_df #
            st.session_state['user_profiles'] = None #
            if full_df is not None: #
                data_to_use = full_df #
                st.warning("️️[!] You are using the full dataset. Calculations may be very slow or cause memory issues, especially for Python-based algorithms like `ItemKNN` or `Slope One`.") #
                st.write(f"Using the full dataset: **{data_to_use.shape[0]} users** and **{data_to_use.shape[1]} movies**.") #
                train_df, test_df = split_data(data_to_use) #
                data_to_train = train_df.to_numpy() #

        elif data_source == "Synthetic 20x20": #
            full_df, movie_titles_df, user_profiles_df = load_synthetic_data() #
            if full_df is not None: #
                st.session_state['movie_titles'] = movie_titles_df #
                st.session_state['user_profiles'] = user_profiles_df #
                data_to_use = full_df #
                st.write(f"Using the synthetic dataset: **{data_to_use.shape[0]} users** and **{data_to_use.shape[1]} movies**.") #
                train_df, test_df = split_data(data_to_use) #
                data_to_train = train_df.to_numpy() #


    # --- FIX: This block was moved outside the `with st.spinner` block ---
    if data_to_use is not None:
        progress_bar = st.progress(0, text=f"Training {algorithm} model...")

        model_map = { # Keep this mapping to find the correct class constructor
            "SVD": SVDRecommender, "ALS": ALSRecommender, "ALS (Improved)": ALSImprovedRecommender,
            "BPR": BPRRecommender, "ItemKNN": ItemKNNRecommender, "Slope One": SlopeOneRecommender,
            "NMF": NMFRecommender, "ALS (PySpark)": ALSPySparkRecommender,
            "FunkSVD": FunkSVDRecommender, "PureSVD": PureSVDRecommender, "SVD++": SVDppRecommender,
            "WRMF": WRMFRecommender, "CML": CMLRecommender, "UserKNN": UserKNNRecommender,
            "NCFNeuMF": NCFRecommender, "SASRec": SASRecRecommender, "SLIM": SLIMRecommender,
            "VAE": VAERecommender, "FISM": FISMRecommender
        }
        model_class = model_map.get(algorithm)
        if model_class:
            # Pass only relevant params from model_params to the constructor
            # This avoids passing params meant for other algorithms
            import inspect
            sig = inspect.signature(model_class.__init__)
            algo_specific_params = {k: v for k, v in model_params.items() if k in sig.parameters}
            # Ensure 'k' is passed if the constructor expects it and it's in the general config
            if 'k' in sig.parameters and 'k' in ALGORITHM_CONFIG[algorithm].get("parameters", {}):
                algo_specific_params['k'] = model_params.get('k')

            model = model_class(**algo_specific_params)
        else:
            st.error(f"Algorithm {algorithm} not found."); st.stop()

        visualizer = None
        if show_internals:
            VisClass = ALGORITHM_CONFIG[algorithm].get("visualizer_class")
            if VisClass:
                # Check if the visualizer needs specific params like k_factors
                vis_sig = inspect.signature(VisClass.__init__)
                vis_args = {}

                # Check for 'k_factors' (used by WRMF, BPR, FunkSVD, etc.)
                if 'k_factors' in vis_sig.parameters:
                    vis_args['k_factors'] = model_params.get('k', 0) # 'k' from sidebar is used as k_factors

                # Check for 'k' (used by SLIMVisualizer)
                if 'k' in vis_sig.parameters:
                    # model_params['k'] might not exist if 'k' isn't in ALGORITHM_CONFIG for SLIM.
                    # We'll default to 10 for the "Top-K" plot if it's not found.
                    # Note: SLIMRecommender itself takes 'k' but defaults to 0. 
                    # The visualizer needs a non-zero k for its plot. 
                    vis_args['k'] = model_params.get('k', 10)

                visualizer = VisClass(**vis_args)

        try:
            model.train_data = data_to_train # Pass train data if needed by predict()
            model.fit(data_to_train,
                      progress_callback=lambda p: progress_bar.progress(p, text=f"Training {algorithm} model... {int(p*100)}%"),
                      visualizer=visualizer)
        except ImportError as e:
            st.error(f"Could not run {algorithm}. Please make sure required libraries are installed: {e}")
            st.stop()
        except Exception as e:
            st.error(f"An error occurred during model training or prediction: {e}")
            import traceback
            st.error(traceback.format_exc()) # Show traceback for debugging
            st.stop()

        progress_bar.empty()
        predicted_matrix = model.predict()
        predicted_matrix = np.nan_to_num(predicted_matrix)
        predicted_df = pd.DataFrame(predicted_matrix, index=data_to_use.index, columns=data_to_use.columns)

        # Store results, including visualizer directory if created
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
            'model_params': model_params # Store the parameters used for this run
        }

        # --- DYNAMIC Metrics Calculation based on Type ---
        if test_df is not None:
            algo_result_type = ALGORITHM_CONFIG[algorithm].get("result_type", "other")
            # Define which types are implicit (could be refined)
            implicit_types = ["bpr", "wrmf", "cml", "neural", "vae", "slim", "fism"] # Include result_types needing ranking metrics

            if any(t in algo_result_type.lower() for t in implicit_types): # Check if type indicates implicit handling
                k_prec_rec = 10
                precision, recall = precision_recall_at_k(predicted_df, test_df, k=k_prec_rec)
                st.session_state['metrics'] = {'type': 'implicit', 'precision': precision, 'recall': recall, 'k': k_prec_rec}
            else: # Assume explicit (regression) metrics otherwise
                metrics = calculate_regression_metrics(predicted_df, test_df)
                st.session_state['metrics'] = {'type': 'explicit', **metrics}
        else:
            st.session_state['metrics'] = None
        # ---

# --- Display Results (Remains mostly the same, uses the new functions from ui_components) ---
if st.session_state['results']:
    results_with_titles = {
        **st.session_state['results'],
        'movie_titles': st.session_state.get('movie_titles'),
        'user_profiles': st.session_state.get('user_profiles')
    }
    if st.session_state['metrics']:
        main_tabs = st.tabs(["Results", "Performance", "Visualizations"])
        with main_tabs[0]:
            render_results_tabs(results_with_titles)
        with main_tabs[1]:
            render_performance_tab(st.session_state['metrics'])
        with main_tabs[2]:
            render_visualizations_tab(results_with_titles) # Now uses results_with_titles
    else:
        main_tabs = st.tabs(["Results", "Visualizations"])
        with main_tabs[0]:
            render_results_tabs(results_with_titles)
        with main_tabs[1]:
            render_visualizations_tab(results_with_titles) # Now uses results_with_titles
else:
    st.info("Select your data, algorithm, and parameters in the sidebar, then click 'Run'.")