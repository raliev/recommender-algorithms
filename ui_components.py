import glob
import json
import os
import re

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from algorithm_config import ALGORITHM_CONFIG, WIDGET_MAP
from visualization.renderers.BaseVisualizationRenderer import BaseVisualizationRenderer

def render_sidebar():
    """
    Renders the sidebar with all user controls and returns the selections.
    """
    link_url = "https://testmysearch.com/books/recommender-algorithms.html"
    image_url = "https://testmysearch.com/img/ra-ws.jpg"

    with st.sidebar:
        col1, col2 = st.columns([1, 2], gap="small")
        with col1:
            st.markdown(
                f"""
                    <a href="{link_url}" target="_blank">
                        <img src="{image_url}" alt="Book Cover" style="width:100%; max-width: 100px; border-radius: 5px;">
                    </a>
                    """,
                unsafe_allow_html=True
            )

        with col2:
            # Using markdown with inline CSS to control top margin for better alignment
            st.markdown("<h5 style='margin-top: 0px; margin-bottom: 10px;'>Recommender Algorithms</h5>", unsafe_allow_html=True)
            st.link_button("Buy this book", link_url, use_container_width=True)

    st.sidebar.divider()

    st.sidebar.header("Controls")

    # Data Source Selection ---
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["Synthetic 20x20", "MovieLens (Filtered Subset)", "MovieLens (Full Dataset)"],
        help="Choose between the synthetic data, a fast filtered subset, or the complete (slower) MovieLens dataset."
    )

    # --- DYNAMIC Algorithm Selection ---
    algorithm_names = list(ALGORITHM_CONFIG.keys())
    algorithm = st.sidebar.selectbox(
        "Select Recommendation Algorithm",
        algorithm_names
    )

    # Initialize dictionaries for parameters
    model_params = {}
    data_params = {}
    data = None

    show_internals = st.sidebar.checkbox("Show Internals / Visualizations", value=True,
                                         help="Generate and display visualizations during/after "
                                              "training (if available for the algorithm).") #

    #  Data Source Specific Controls
    if data_source == "MovieLens (Filtered Subset)":
        st.sidebar.info("A fast subset of the most active users and popular movies.")
        st.sidebar.subheader("1. Select Data Size") #
        data_params['num_users'] = st.sidebar.slider("Number of Users to Use", 50, 610, 100)
        data_params['num_movies'] = st.sidebar.slider("Number of Movies to Use", 100, 2000, 500)
    elif data_source == "Synthetic 20x20":
        st.sidebar.info("Using your generated 20x20 synthetic dataset.")
    elif data_source == "MovieLens (Full Dataset)":
        st.sidebar.info("Using the complete, unfiltered MovieLens dataset. This may be slow.")

    # DYNAMIC Hyperparameters
    st.sidebar.subheader(f"2. {algorithm} Hyperparameters")

    # Display algorithm-specific info if available
    algo_info = ALGORITHM_CONFIG[algorithm].get("info")
    if algo_info:
        st.sidebar.info(algo_info) #

    # Get parameter definitions for the selected algorithm
    param_definitions = ALGORITHM_CONFIG[algorithm].get("parameters", {})

    # Check if algorithm has changed to reset session state values
    if 'current_sidebar_algo' not in st.session_state or st.session_state.current_sidebar_algo != algorithm:
        st.session_state.current_sidebar_algo = algorithm
        # Clear old keys if necessary (or just let them be overwritten)
        pass # Initialization will happen inside the loop

    # Dynamically create widgets
    for param_name, config in param_definitions.items():
        widget_func = WIDGET_MAP.get(config["type"])

        if config["type"] == "slider":
            label = config["label"]

            # --- Create unique session state keys ---
            base_key = f"param_{algorithm}_{param_name}"
            # This is the SHARED state key
            value_key = f"{base_key}_value"
            # These are for the range editors
            min_key = f"{base_key}_min"
            max_key = f"{base_key}_max"
            step_key = f"{base_key}_step"
            num_input_widget_key = f"{base_key}_num_input_widget"
            slider_widget_key = f"{base_key}_slider_widget"

            # Get default values from config
            default_min = config.get("min")
            default_max = config.get("max")
            default_step = config.get("step")
            default_value = config.get("default")
            val_format = config.get("format")

            # Use a small value for step if not defined
            if default_step is None:
                default_step = 1 if isinstance(default_value, int) else 0.0001
            # Initialize session state if keys don't exist
            if min_key not in st.session_state or st.session_state.current_sidebar_algo != algorithm:
                st.session_state[min_key] = default_min
            if max_key not in st.session_state or st.session_state.current_sidebar_algo != algorithm:
                st.session_state[max_key] = default_max
            if step_key not in st.session_state or st.session_state.current_sidebar_algo != algorithm:
                st.session_state[step_key] = default_step
            if value_key not in st.session_state or st.session_state.current_sidebar_algo != algorithm:
                st.session_state[value_key] = default_value

            # Read current values from session state
            current_min = st.session_state[min_key]
            current_max = st.session_state[max_key]
            current_step = st.session_state[step_key]

            # Ensure current value is within the new bounds
            if st.session_state[value_key] > current_max:
                st.session_state[value_key] = current_max
            if st.session_state[value_key] < current_min:
                st.session_state[value_key] = current_min

            # Read the shared value to pass to both widgets
            current_shared_value = st.session_state[value_key]

            # --- Render Primary Value Input ---
            st.sidebar.number_input(
                label=label,
                min_value=current_min,
                max_value=current_max,
                step=current_step,
                value=current_shared_value,       # Read from shared state
                key=num_input_widget_key,
                on_change=update_state,
                args=(num_input_widget_key, value_key), # (source, target)
                format=val_format
            )

            # --- Render Slider (linked to the same state key) ---
            st.sidebar.slider(
                label=f"{label} Slider", # Needs a unique label for streamlit
                label_visibility="collapsed", # Hide the label
                min_value=current_min,
                max_value=current_max,
                step=current_step,
                value=current_shared_value,       # Read from shared state
                key=slider_widget_key,
                on_change=update_state,
                args=(slider_widget_key, value_key), # (source, target)
                format=val_format
            )

            # Render Secondary Min/Max/Step Editors ---
            with st.sidebar.expander("Edit Range", expanded=False):
                cols = st.columns(3)
                # These widgets write to their respective state keys
                cols[0].number_input(
                    "Min",
                    key=min_key,
                    step=default_step,
                    format=val_format
                )
                cols[1].number_input(
                    "Step",
                    key=step_key,
                    step=default_step,
                    format=val_format,
                    min_value=0.0000001 if isinstance(current_step, float) else 1 # Step must be positive
                )
                cols[2].number_input(
                    "Max",
                    key=max_key,
                    step=default_step,
                    format=val_format
                )

            st.sidebar.divider() # Add a separator

            # Store the final value
            model_params[param_name] = st.session_state[value_key]

        elif widget_func: # Existing logic for selectbox/select_slider
            label = config["label"] #
            widget_args = {}
            if config["type"] == "select_slider":
                widget_args["options"] = config.get("options")
                widget_args["value"] = config.get("default")
            elif config["type"] == "selectbox":
                widget_args["options"] = config.get("options")
                try:
                    default_index = config.get("options", []).index(config.get("default"))
                    widget_args["index"] = default_index
                except ValueError:
                    widget_args["index"] = 0 #

            widget_args = {k: v for k, v in widget_args.items() if v is not None} #

            model_params[param_name] = widget_func(label, **widget_args)
        else:
            st.sidebar.error(f"Unknown widget type '{config['type']}' for parameter '{param_name}'")
    # ---

    run_button = st.sidebar.button("Run Algorithm", use_container_width=True) #

    return data_source, data, algorithm, model_params, data_params, run_button, show_internals


def get_movie_title(movie_id, movie_titles_df):
    """Helper function to get movie title from movieId."""
    if movie_titles_df is None:
        return f"Movie ID: {movie_id}"
    try:
        return movie_titles_df.loc[movie_id, 'title']
    except KeyError:
        return f"Movie ID: {movie_id} (Unknown)"

@st.cache_data # Cache the markdown content
def load_visualization_info(algo_name):
    # ... (implementation remains the same) ...
    # Construct the file path based on the algorithm name
    info_path = os.path.join('visualizations_info', f'{algo_name.lower().replace(" / ", "_").replace(" ", "_")}.md')
    info_path = info_path.replace("+", "p");
    explanations = {} # Initialize an empty dictionary

    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            content = f.read()
        sections = re.split(r'\n-{3,}\n', content)
        for section in sections:
            section = section.strip()
            if not section: continue
            parts = re.split(r'\n###\s+', section)
            for i, part_content in enumerate(parts):
                if i == 0 and not section.startswith("###"): continue
                lines = part_content.split('\n', 1)
                heading_text = lines[0].strip().replace("### ", "")
                if heading_text: explanations[heading_text] = "### " + part_content
    except FileNotFoundError:
        print(f"Warning: Visualization info file not found at {info_path}")
        explanations['Error'] = f"Visualization explanation file not found for {algo_name}."
    except Exception as e:
        print(f"Error reading or parsing {info_path}: {e}")
        explanations['Error'] = f"Error loading visualization explanations for {algo_name}."

    key_map = {'Objective Function Plot (Approximate)': 'Objective', 'Factor Change Norm Plot': 'Factor Change',
               'Heatmaps (P and Q Snapshots)': 'Heatmaps', 'Histograms (P and Q Snapshots)': 'Histograms',
               '2D Latent Space Plot (if k=2)': '2D Space'}
    for md_key, old_key in key_map.items():
        if md_key in explanations: explanations[old_key] = explanations[md_key]
    return explanations

def render_results_tabs(results):
    st.header("Results")
    tab1, tab2, tab3 = st.tabs(["Model Internals", "Predictions", "Recommendations"])

    movie_titles_df = results.get('movie_titles')

    def rename_columns_to_titles(df):
        if movie_titles_df is not None and not df.columns.is_numeric():
            # If columns are already renamed, just return
            return df
        if movie_titles_df is not None:
            title_map = {mid: get_movie_title(mid, movie_titles_df) for mid in df.columns}
            return df.rename(columns=title_map)
        return df

    with tab1:
        st.subheader(f"Inside the {results['algo_name']} Model")
        algo_config = ALGORITHM_CONFIG.get(results['algo_name'], {})
        result_type = algo_config.get("result_type", "other") # Get result type from config

        if result_type == "matrix_factorization": # Use result_type
            st.write("These models decompose the original matrix into User-Factors (P) and Item-Factors (Q).")
            if results.get('P') is not None and results.get('Q') is not None:
                q_df = pd.DataFrame(results['Q'], index=results['original_df'].columns)
                q_df_renamed = rename_columns_to_titles(q_df.T).T
                col1, col2 = st.columns(2)
                with col1: st.write("**P (Users x Factors)**"); st.dataframe(pd.DataFrame(results['P'], index=results['original_df'].index).style.format("{:.2f}"))
                with col2: st.write("**Q (Items x Factors)**"); st.dataframe(q_df_renamed.style.format("{:.2f}"))
            else:
                st.info(f"{results['algo_name']} does not expose user/item factor matrices in a simple format.")

        elif result_type == "knn_similarity":
            st.write("These models compute or learn a **Similarity Matrix** to find similar users or items.")
            sim_matrix = results.get('similarity_matrix')
            if sim_matrix is not None:
                df = pd.DataFrame(sim_matrix)
                if results['algo_name'] in ["ItemKNN", "SLIM", "FISM"]:
                    df.index = results['original_df'].columns
                    df.columns = results['original_df'].columns
                    df = rename_columns_to_titles(rename_columns_to_titles(df.T).T)
                else: # UserKNN
                    df.index = results['original_df'].index
                    df.columns = results['original_df'].index

                st.write(f"**Learned Similarity Matrix (subset)**")
                max_dim = 25
                if df.shape[0] > max_dim:
                    st.info(f"Displaying a {max_dim}x{max_dim} subset of the full similarity matrix.")
                    df_subset = df.iloc[:max_dim, :max_dim]
                else:
                    df_subset = df
                fig = px.imshow(df_subset, text_auto=".2f", aspect="auto", title=f"{results['algo_name']} Similarity")
                st.plotly_chart(fig)
            else:
                st.info("Similarity matrix not available.")

        elif result_type == "vae":
            st.write("Autoencoder models learn to **reconstruct** a user's interaction history from a compressed latent representation.")
            recon_matrix = results.get('reconstructed_matrix')
            if recon_matrix is not None:
                user_list = results['original_df'].index
                selected_user = st.selectbox("Select a User to Visualize:", options=user_list)
                original_vec = results['original_df'].loc[selected_user]
                user_list_as_list = list(user_list)
                user_index = user_list_as_list.index(selected_user)
                recon_vec = pd.Series(recon_matrix[user_index], index=original_vec.index)
                rated_items = original_vec[original_vec > 0].index
                top_unrated_recon = recon_vec.drop(rated_items).nlargest(10).index
                items_to_show = rated_items.union(top_unrated_recon)
                vis_df = pd.DataFrame({'Original Interaction': (original_vec[items_to_show] > 0).astype(int),'Reconstructed Score': recon_vec[items_to_show]})
                vis_df.index = vis_df.index.map(lambda mid: get_movie_title(mid, movie_titles_df))
                fig = px.bar(vis_df, barmode='group', title=f"Original vs. Reconstructed Interactions for User {selected_user}")
                st.plotly_chart(fig)
            else: st.info("Reconstructed matrix not available.")

        else:
            st.info(f"No specific internal visualization is available for {results['algo_name']}.")

    with tab2:
        st.subheader("Original Data vs. Predicted Scores")
        if results['algo_name'] == 'BPR': st.warning("Reminder: BPR outputs scores for ranking, not predicted ratings.")
        original_df, predicted_df = results['original_df'], results['predicted_df']
        max_users, max_items = 20, 20
        original_df_renamed = rename_columns_to_titles(original_df)
        predicted_df_renamed = rename_columns_to_titles(predicted_df)

        if original_df.shape[0] > max_users or original_df.shape[1] > max_items:
            st.info(f"Displaying a subset ({max_users} users, {max_items} items) of the full matrix for performance.")
            display_original_df = original_df_renamed.iloc[:max_users, :max_items]
            display_predicted_df = predicted_df_renamed.iloc[:max_users, :max_items]
        else:
            display_original_df, display_predicted_df = original_df_renamed, predicted_df_renamed

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original Data (subset)**")
            st.dataframe(display_original_df)
        with col2:
            st.write("**Predictions (subset, new items highlighted)**")
            def style_predictions(row):
                original_row = display_original_df.loc[row.name]
                highlight = 'background-color: #d1ecf1; color: #0c5460; font-weight: bold;'
                return [highlight if original_row[col] == 0 else '' for col in original_row.index]
            st.dataframe(display_predicted_df.style.format("{:.2f}").apply(style_predictions, axis=1))

    with tab3:
        st.subheader("Get Top N Recommendations")
        user_list = results['predicted_df'].index
        selected_user = st.selectbox("Select a User:", options=user_list)
        num_recs = st.number_input("Number of Recommendations (N):", 1, 20, 5)

        if selected_user:
            user_profiles_df = results.get('user_profiles')
            if user_profiles_df is not None:
                st.subheader(f"Ground Truth Interest Profile for User {selected_user}")
                try:
                    user_index = selected_user - 1 # Assuming user IDs start from 1
                    user_profile_data = user_profiles_df.iloc[user_index]
                    profile_df = user_profile_data.to_frame(name='Interest Score').sort_values(by='Interest Score', ascending=False)
                    st.dataframe(profile_df.style.background_gradient(cmap='viridis', axis=None).format("{:.2f}"), use_container_width=True)
                except (KeyError, IndexError):
                    st.warning(f"Could not find a ground truth profile for User {selected_user}.")

            original_data, user_scores = results['original_df'], results['predicted_df'].loc[selected_user]
            seen_items = original_data.loc[selected_user][original_data.loc[selected_user] > 0].index
            top_n_ids = user_scores.drop(seen_items, errors='ignore').nlargest(num_recs)
            top_n_titles = top_n_ids.copy()
            top_n_titles.index = top_n_titles.index.map(lambda mid: get_movie_title(mid, movie_titles_df))
            st.subheader(f"Top {num_recs} Recommendations for User {selected_user}")
            st.dataframe(top_n_titles.to_frame(name="Predicted Score").style.format("{:.2f}"))

def render_performance_tab(metrics):
    st.header("Model Performance on Test Set")
    if metrics['type'] == 'explicit':
        st.info("These metrics evaluate the accuracy of the predicted ratings against the actual ratings in the test set.")
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Root Mean Squared Error (RMSE)", value=f"{metrics.get('rmse', 0):.4f}", help="Measures the average error in predicted ratings. Lower is better.")
        col2.metric(label="Mean Absolute Error (MAE)", value=f"{metrics.get('mae', 0):.4f}", help="Similar to RMSE, but less sensitive to large errors. Lower is better.")
        col3.metric(label="R-squared (R²)", value=f"{metrics.get('r2', 0):.4f}", help="Indicates the proportion of variance in the actual ratings that is predictable from the model. Closer to 1 is better.")
        col4, col5 = st.columns(2)
        col4.metric(label="Mean Absolute Percentage Error (MAPE)", value=f"{metrics.get('mape', 0):.2f}%", help="Expresses the mean absolute error as a percentage of actual values. Lower is better.")
        col5.metric(label="Explained Variance Score", value=f"{metrics.get('explained_variance', 0):.4f}", help="Measures how well the model accounts for the variation in the original data. Closer to 1 is better.")
    elif metrics['type'] == 'implicit':
        st.info("These metrics evaluate the quality of the item rankings produced by the model.")
        col1, col2 = st.columns(2)
        k_val = metrics.get('k', 10)
        col1.metric(label=f"Precision@{k_val}", value=f"{metrics.get('precision', 0):.2%}")
        col2.metric(label=f"Recall@{k_val}", value=f"{metrics.get('recall', 0):.2%}")
        st.info(f"**Precision**: Of the top {k_val} items recommended, what percentage were actually relevant items from the test set?\n\n**Recall**: Of all the relevant items in the test set, what percentage did the model successfully recommend in the top {k_val}?")
        fig = go.Figure(data=[go.Bar(name='Precision', x=['Performance'], y=[metrics.get('precision', 0)]), go.Bar(name='Recall', x=['Performance'], y=[metrics.get('recall', 0)])])
        fig.update_layout(title_text=f'Precision and Recall @ {k_val}', yaxis_title="Score", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)


def render_visualizations_tab(results):
    st.header("Training Visualizations")
    algo_name = results.get('algo_name')
    selected_run_dir = results.get('visuals_dir')

    if not algo_name or not selected_run_dir or not os.path.isdir(selected_run_dir):
        st.info(f"Visualizations directory not found for this run.")
        return
    algo_config = ALGORITHM_CONFIG.get(algo_name, {})
    RendererClass = algo_config.get("visualization_renderer_class")
    explanations = load_visualization_info(algo_name)
    st.write(f"Displaying visualizations from: `{selected_run_dir}`")

    # --- Instantiate and Call Renderer ---
    if RendererClass and issubclass(RendererClass, BaseVisualizationRenderer):
        try:
            renderer = RendererClass(selected_run_dir, explanations)
            renderer.render() # Call the algorithm-specific render method
        except Exception as e:
            st.error(f"Error rendering visualizations for {algo_name}: {e}")
            import traceback
            st.exception(traceback.format_exc()) # Show traceback in app for debugging
    else:
        st.info(f"No specific visualization renderer defined or found for {algo_name}.")

    #  Explore Previous Runs (Moved to helper function) ---
    _render_explore_previous_runs(algo_name, results.get('visuals_base_dir'), selected_run_dir)


def _render_explore_previous_runs(algo_name, visuals_base_dir, current_run_dir):
    """Helper function to render the 'Explore Previous Runs' section."""
    st.divider()
    st.subheader("Explore Previous Runs")

    if not visuals_base_dir or not algo_name:
        st.info("Cannot explore previous runs without base directory information.")
        return

    algo_visuals_dir = os.path.join(visuals_base_dir, algo_name) # e.g., visuals/WRMF

    if not os.path.isdir(algo_visuals_dir):
        st.info(f"No previous visualization runs found for algorithm '{algo_name}' in `{algo_visuals_dir}`.")
        return

    # Find all subdirectories (potential runs) in the algorithm's visual directory
    try:
        run_dirs = sorted(
            [d for d in glob.glob(os.path.join(algo_visuals_dir, '*')) if os.path.isdir(d)],
            reverse=True # Show most recent first
        )
    except Exception as e:
        st.error(f"Error accessing visualization directories: {e}")
        return

    if not run_dirs:
        st.info(f"No specific run subdirectories found in '{algo_visuals_dir}'.")
        return

    run_options = {}
    run_display_list = []
    current_run_basename = os.path.basename(current_run_dir) if current_run_dir else None

    # Create display names with timestamps and parameters
    for run_dir in run_dirs:
        timestamp = os.path.basename(run_dir)
        params_path = os.path.join(run_dir, 'params.json')
        params_str = "Params: N/A"
        try:
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    params = json.load(f)
                # Filter out potentially redundant keys for display
                params_str = ", ".join(f"{k}={v}" for k, v in params.items()
                                       if k not in ['algorithm', 'timestamp', 'visuals_dir', 'visuals_base_dir']) # Exclude common/internal params
                if not params_str: params_str = "Default Params" # Handle case where only excluded params exist
        except Exception as e:
            params_str = f"Error loading params: {e}"

        is_current = (timestamp == current_run_basename)
        display_text = f"Run: {timestamp} ({params_str})" + (" (Current)" if is_current else "")
        run_options[display_text] = run_dir
        run_display_list.append(display_text)

    # Find the index of the current run for the selectbox default
    current_index = 0
    if current_run_basename:
        try:
            current_index = next(i for i, s in enumerate(run_display_list) if "(Current)" in s)
        except StopIteration:
            pass # Default to 0 if current run dir isn't listed (e.g., deleted)


    selected_display_run_explore = st.selectbox(
        "Select Previous Run to Visualize:",
        run_display_list,
        index=current_index,
        help="Select a previous run. The visualizations above will update to show the selected run's results."
    )

    # Logic to potentially re-render based on selection ---
    # This requires storing the selected run_dir in session state and triggering a rerun.
    # We'll set up the session state update here. The main app loop (0_Lab.py)
    # would need to read this state if it exists.

    selected_run_path = run_options.get(selected_display_run_explore)

    if 'selected_visuals_run_dir' not in st.session_state:
        st.session_state.selected_visuals_run_dir = current_run_dir

    # Detect if the user *changed* the selection from what's currently displayed
    # (either the initial 'current' run or a previously selected one)
    if selected_run_path and selected_run_path != st.session_state.selected_visuals_run_dir:
        st.info(f"Switching view to run: `{os.path.basename(selected_run_path)}`")
        st.session_state.selected_visuals_run_dir = selected_run_path
        # Force Streamlit to rerun the script to reflect the change
        # This will cause render_visualizations_tab to be called again,
        # but this time `selected_run_dir` should ideally come from session state.
        st.rerun()
    elif not selected_run_path and selected_display_run_explore:
        st.warning("Selected run option doesn't map to a valid directory.")

def update_state(source_widget_key, target_state_key):
    """Callback to sync a widget's state with a session_state key."""
    st.session_state[target_state_key] = st.session_state[source_widget_key]