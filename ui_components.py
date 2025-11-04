import glob
import json
import os
import re
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from algorithm_config import ALGORITHM_CONFIG, WIDGET_MAP
from visualization.renderers.BaseVisualizationRenderer import BaseVisualizationRenderer
from utils import get_generated_datasets

def update_selected_run_dir(key_name, options_dict):
    """Callback to update session state with the DIR PATH, not the display name."""
    display_name = st.session_state[key_name]
    selected_dir = options_dict.get(display_name)
    st.session_state.selected_visuals_run_dir = selected_dir

    # If user selects "- Select a run -", clear results
    if selected_dir is None:
        st.session_state.results = None
        st.session_state.metrics = None


def render_sidebar():
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
            st.markdown("<h5 style='margin-top: 0px; margin-bottom: 10px;'>Recommender Algorithms</h5>", unsafe_allow_html=True)
            st.link_button("Buy this book", link_url, use_container_width=True)

    st.sidebar.divider()
    st.sidebar.header("Controls")
    base_data_options = ["Synthetic 20x20", "MovieLens (Filtered Subset)", "MovieLens (Full Dataset)"]
    generated_data_options = get_generated_datasets() # Get list of names
    all_data_options = base_data_options + generated_data_options

    data_source = st.sidebar.radio(
        "Select Data Source",
        all_data_options,
        help="Choose between synthetic data, MovieLens, or a custom-generated dataset."
    )
    algorithm_names = list(ALGORITHM_CONFIG.keys())
    algorithm = st.sidebar.selectbox(
        "Select Recommendation Algorithm",
        algorithm_names,
        on_change=lambda: st.session_state.update(selected_visuals_run_dir=None, results=None, metrics=None)
    )
    model_params = {}
    data_params = {}
    data = None
    show_internals = st.sidebar.checkbox("Show Internals / Visualizations", value=True,
                                         help="Generate and display visualizations during/after "
                                              "training (if available for the algorithm).") #
    if data_source == "MovieLens (Filtered Subset)":
        st.sidebar.info("A fast subset of the most active users and popular movies.")
        st.sidebar.subheader("1. Select Data Size") #
        data_params['num_users'] = st.sidebar.slider("Number of Users to Use", 50, 610, 100)
        data_params['num_movies'] = st.sidebar.slider("Number of Items to Use", 100, 2000, 500)
    elif data_source == "Synthetic 20x20":
        st.sidebar.info("Using your generated 20x20 synthetic dataset.")
    elif data_source == "MovieLens (Full Dataset)":
        st.sidebar.info("Using the complete, unfiltered MovieLens dataset. This may be slow.")
    elif data_source in generated_data_options:
        st.sidebar.info(f"Using your custom generated dataset: '{data_source}'")
    st.sidebar.subheader(f"2. {algorithm} Hyperparameters")
    algo_info = ALGORITHM_CONFIG[algorithm].get("info")
    if algo_info:
        st.sidebar.info(algo_info) #
    param_definitions = ALGORITHM_CONFIG[algorithm].get("parameters", {})
    if 'current_sidebar_algo' not in st.session_state or st.session_state.current_sidebar_algo != algorithm:
        st.session_state.current_sidebar_algo = algorithm
        pass
    for param_name, config in param_definitions.items():
        widget_func = WIDGET_MAP.get(config["type"])
        if config["type"] == "slider":
            label = config["label"]
            base_key = f"param_{algorithm}_{param_name}"
            value_key = f"{base_key}_value"
            min_key = f"{base_key}_min"
            max_key = f"{base_key}_max"
            step_key = f"{base_key}_step"
            num_input_widget_key = f"{base_key}_num_input_widget"
            slider_widget_key = f"{base_key}_slider_widget"
            default_min = config.get("min")
            default_max = config.get("max")
            default_step = config.get("step")
            default_value = config.get("default")
            val_format = config.get("format")
            if default_step is None:
                default_step = 1 if isinstance(default_value, int) else 0.0001
            if min_key not in st.session_state or st.session_state.current_sidebar_algo != algorithm:
                st.session_state[min_key] = default_min
            if max_key not in st.session_state or st.session_state.current_sidebar_algo != algorithm:
                st.session_state[max_key] = default_max
            if step_key not in st.session_state or st.session_state.current_sidebar_algo != algorithm:
                st.session_state[step_key] = default_step
            if value_key not in st.session_state or st.session_state.current_sidebar_algo != algorithm:
                st.session_state[value_key] = default_value
            current_min = st.session_state[min_key]
            current_max = st.session_state[max_key]
            current_step = st.session_state[step_key]
            if st.session_state[value_key] > current_max:
                st.session_state[value_key] = current_max
            if st.session_state[value_key] < current_min:
                st.session_state[value_key] = current_min
            current_shared_value = st.session_state[value_key]
            st.sidebar.number_input(
                label=label,
                min_value=current_min,
                max_value=current_max,
                step=current_step,
                value=current_shared_value,
                key=num_input_widget_key,
                on_change=update_state,
                args=(num_input_widget_key, value_key),
                format=val_format
            )
            st.sidebar.slider(
                label=f"{label} Slider",
                label_visibility="collapsed",
                min_value=current_min,
                max_value=current_max,
                step=current_step,
                value=current_shared_value,
                key=slider_widget_key,
                on_change=update_state,
                args=(slider_widget_key, value_key),
                format=val_format
            )
            with st.sidebar.expander("Edit Range", expanded=False):
                cols = st.columns(3)
                cols[0].number_input( "Min", key=min_key, step=default_step, format=val_format )
                cols[1].number_input( "Step", key=step_key, step=default_step, format=val_format, min_value=0.0000001 if isinstance(current_step, float) else 1 )
                cols[2].number_input( "Max", key=max_key, step=default_step, format=val_format )
            st.sidebar.divider()
            model_params[param_name] = st.session_state[value_key]
        elif widget_func:
            label = config["label"]
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
                    widget_args["index"] = 0
            widget_args = {k: v for k, v in widget_args.items() if v is not None}
            model_params[param_name] = widget_func(label, **widget_args)
        else:
            st.sidebar.error(f"Unknown widget type '{config['type']}' for parameter '{param_name}'")
    run_button = st.sidebar.button("Run Algorithm", use_container_width=True)
    return data_source, data, algorithm, model_params, data_params, run_button, show_internals

def render_previous_runs_explorer(algo_name, visuals_base_dir):
    """
    Renders the selectbox for exploring previous runs for the selected algorithm.
    This function updates st.session_state.selected_visuals_run_dir via its callback.
    """
    st.divider()
    st.header("Explore Previous Runs")

    if not visuals_base_dir or not algo_name:
        st.info("Algorithm not selected.")
        return

    algo_visuals_dir = os.path.join(visuals_base_dir, algo_name)

    if not os.path.isdir(algo_visuals_dir):
        st.info(f"No previous visualization runs found for '{algo_name}'.")
        return

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

    # {display_text: run_dir}
    run_options = {"- Select a run to view -": None}
    run_display_list = ["- Select a run to view -"]

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
                                       if k not in ['algorithm', 'timestamp'])
                if not params_str: params_str = "Default Params"
        except Exception:
            params_str = "Error loading params"

        display_text = f"Run: {timestamp} ({params_str})"
        run_options[display_text] = run_dir
        run_display_list.append(display_text)

    # Find the index of the currently selected run, if any
    current_index = 0
    if st.session_state.get('selected_visuals_run_dir'):
        try:
            # Find the display_text that matches the selected_visuals_run_dir
            current_display_text = next(disp for disp, path in run_options.items() if path == st.session_state.selected_visuals_run_dir)
            current_index = run_display_list.index(current_display_text)
        except (StopIteration, ValueError):
            st.session_state.selected_visuals_run_dir = None # Reset if not found
            current_index = 0

    selectbox_key = f"{algo_name}_run_selector"
    st.selectbox(
        "Select Previous Run to Visualize:",
        run_display_list,
        index=current_index,
        help="Select a previous run. The visualizations will appear below.",
        key=selectbox_key,
        on_change=update_selected_run_dir,
        args=(selectbox_key, run_options)
    )

def get_movie_title(movie_id, movie_titles_df):
    # ... (existing function is unchanged) ...
    if movie_titles_df is None:
        return f"Movie ID: {movie_id}"
    try:
        # Check if movie_id is in the index
        if movie_id in movie_titles_df.index:
            return movie_titles_df.loc[movie_id, 'title']
        else:
            # Fallback for generated datasets where index might be the title itself
            if movie_id in movie_titles_df['title'].values:
                return movie_id
            return f"Movie ID: {movie_id} (Unknown)"
    except KeyError:
        return f"Movie ID: {movie_id} (Unknown)"
    except Exception:
        # Handle cases where movie_id might be a string for generated data
        if isinstance(movie_id, str) and movie_id in movie_titles_df['title'].values:
            return movie_id
        return f"Movie ID: {movie_id} (Error)"


@st.cache_data
def load_visualization_info(algo_name):
    info_path = os.path.join('visualizations_info', f'{algo_name.lower().replace(" / ", "_").replace(" ", "_")}.md')
    info_path = info_path.replace("+", "p");
    explanations = {}
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

    # Helper to rename columns from movie_id to title
    def rename_columns_to_titles(df):
        if movie_titles_df is None:
            return df

        # Check if columns are numeric (like MovieLens) or strings (like generated data)
        if pd.api.types.is_numeric_dtype(df.columns):
            # MovieLens logic
            title_map = {mid: get_movie_title(mid, movie_titles_df) for mid in df.columns}
            return df.rename(columns=title_map)
        else:
            # Generated data logic (columns are already titles)
            return df

    with tab1:
        st.subheader(f"Inside the {results['algo_name']} Model")
        algo_config = ALGORITHM_CONFIG.get(results['algo_name'], {})
        result_type = algo_config.get("result_type", "other")

        if result_type == "matrix_factorization":
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
                else:
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

                # Handle both numeric and string user lists
                if pd.api.types.is_numeric_dtype(user_list):
                    user_list_as_list = list(user_list)
                    user_index = user_list_as_list.index(selected_user)
                else:
                    user_index = user_list.get_loc(selected_user)

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
        if 'BPR' in results['algo_name']: st.warning("Reminder: BPR outputs scores for ranking, not predicted ratings.")

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

        is_generated_dataset = results.get('user_profiles') is not None and results.get('Q_matrix') is not None

        algo_name = results['algo_name']
        algo_config_all = ALGORITHM_CONFIG.get(algo_name, {})
        is_implicit_model = algo_config_all.get("is_implicit", False)

        if is_generated_dataset and not is_implicit_model:
            st.divider()
            st.subheader("Ground Truth Analysis (for reference only)")
            st.info("The following matrices were used to **generate** this dataset. They were **not** passed to the recommender.")

            p_matrix = results.get('user_profiles')
            q_matrix = results.get('Q_matrix')

            # Get the same user/item indices/names as the displayed subsets
            user_subset_index = display_original_df.index
            item_subset_index = display_original_df.columns

            # Filter P and Q
            P_subset = p_matrix.loc[user_subset_index] # Selects rows (users)
            Q_subset = q_matrix.loc[item_subset_index] # Selects rows (items)
            R_pred_subset = display_predicted_df

            col3, col4 = st.columns(2)
            with col3:
                st.write("**P (User-Preference) Matrix (subset)**")
                st.dataframe(P_subset.style.format("{:.2f}").background_gradient(cmap='viridis', axis=None, vmin=-1, vmax=1), use_container_width=True)
            with col4:
                st.write("**Q (Item-Feature) Matrix (subset)**")
                st.dataframe(Q_subset.style.format("{:.0f}"), use_container_width=True)

            # --- NEW: Calculate and Display Expectation Matrix ---
            st.subheader("Prediction Expectation Analysis")
            st.info("This table shows the **difference** between the model's prediction and the 'ideal' rating (generated from the ground-truth P and Q matrices). "
                    "\n- **Near 0 (Yellow):** The prediction is *expected*. "
                    "\n- **Positive (Green):** The prediction is *higher than expected*. "
                    "\n- **Negative (Red):** The prediction is *lower than expected*.")

            try:
                R_ideal_raw = P_subset @ Q_subset.T

                if R_ideal_raw.size > 0:
                    scaler = MinMaxScaler(feature_range=(0, 5))
                    R_ideal_scaled = scaler.fit_transform(R_ideal_raw.to_numpy().reshape(-1, 1)).reshape(R_ideal_raw.shape)
                    R_ideal_scaled_df = pd.DataFrame(R_ideal_scaled, index=R_ideal_raw.index, columns=R_ideal_raw.columns)

                    # Calculate the difference
                    Diff_df = R_pred_subset - R_ideal_scaled_df

                    st.write("**Prediction Expectation (Prediction - Ideal Rating)**")

                    # --- START FIX for Request 2 (Truncation) ---
                    def truncate_name(name, length=15):
                        if len(str(name)) > length:
                            return str(name)[:length] + '...'
                        return str(name)

                    Diff_df_display = Diff_df.copy()
                    # Truncate the *column* (item) names
                    Diff_df_display.columns = [truncate_name(col, 15) for col in Diff_df.columns]

                    # Render the truncated dataframe
                    st.dataframe(Diff_df_display.style.background_gradient(cmap='RdYlGn_r', axis=None, vmin=-3, vmax=3).format("{:.2f}"))
                    # --- END FIX for Request 2 (Truncation) ---


                    # --- NEW: Deep Dive Section ---
                    st.subheader("Prediction Deep Dive")

                    # --- START FIX for Request 1 (Clarity) ---
                    st.info("ℹ️ **How to use:** Select a user and item from the dropdowns below to see the ground-truth calculation for that specific cell. (Clicking the heatmap cell directly is not supported).")
                    # --- END FIX for Request 1 (Clarity) ---

                    deep_col1, deep_col2 = st.columns(2)
                    with deep_col1:
                        selected_user_deep = st.selectbox("Select User:", options=display_original_df.index, key="deep_dive_user")
                    with deep_col2:
                        selected_item_deep = st.selectbox("Select Item:", options=display_original_df.columns, key="deep_dive_item")

                    if selected_user_deep and selected_item_deep:
                        p_vec = P_subset.loc[selected_user_deep]
                        q_vec = Q_subset.loc[selected_item_deep]

                        ideal_raw = R_ideal_raw.loc[selected_user_deep, selected_item_deep]
                        ideal_scaled = R_ideal_scaled_df.loc[selected_user_deep, selected_item_deep]
                        prediction = R_pred_subset.loc[selected_user_deep, selected_item_deep]
                        diff = Diff_df.loc[selected_user_deep, selected_item_deep]

                        calc_col1, calc_col2, calc_col3 = st.columns(3)

                        with calc_col1:
                            st.write(f"**User Profile (P)**")
                            st.dataframe(p_vec.to_frame(name="Preference").style.background_gradient(cmap='viridis', vmin=-1, vmax=1).format("{:.2f}"))
                        with calc_col2:
                            st.write(f"**Item Profile (Q)**")
                            st.dataframe(q_vec.to_frame(name="Feature"))
                        with calc_col3:
                            st.write("**Calculation**")
                            st.metric(label="Ground Truth Score (P @ Q.T)", value=f"{ideal_raw:.3f}")
                            st.metric(label="Ideal Scaled Rating (0-5)", value=f"{ideal_scaled:.3f}")
                            st.metric(label="Model's Actual Prediction", value=f"{prediction:.3f}")
                            st.metric(label="Difference (Prediction - Ideal)", value=f"{diff:.3f}")

            except Exception as e:
                st.error(f"Error during expectation analysis: {e}")
                import traceback
                st.exception(traceback.format_exc())
        elif is_generated_dataset and is_implicit_model:
            st.divider()
            st.subheader("Ground Truth Analysis (Not Applicable)")
            st.info(f"The **Prediction Expectation Analysis** is not shown because **{algo_name}** is an implicit feedback (ranking) model. "
                    "It predicts ranking scores, not explicit 0-5 ratings, so it cannot be directly compared to the 'Ideal Rating' ground truth.")
    with tab3:
        st.subheader("Get Top N Recommendations")
        user_list = results['predicted_df'].index
        selected_user = st.selectbox("Select a User:", options=user_list)
        num_recs = st.number_input("Number of Recommendations (N):", 1, 20, 5)

        if selected_user:
            user_profiles_df = results.get('user_profiles')

            # --- MODIFIED BLOCK: Show P matrix profile if it exists ---
            if user_profiles_df is not None:
                st.subheader(f"Ground Truth Interest Profile for User {selected_user}")
                try:
                    # --- FIX: Use .loc[] for string-based index ---
                    user_profile_data = user_profiles_df.loc[selected_user]
                    profile_df = user_profile_data.to_frame(name='Interest Score').sort_values(by='Interest Score', ascending=False)
                    st.dataframe(profile_df.style.background_gradient(cmap='viridis', axis=None, vmin=-1, vmax=1).format("{:.2f}"), use_container_width=True)
                except (KeyError, IndexError, TypeError):
                    st.warning(f"Could not find a ground truth profile for User {selected_user}.")
            # --- END MODIFIED BLOCK ---

            original_data, user_scores = results['original_df'], results['predicted_df'].loc[selected_user]
            seen_items = original_data.loc[selected_user][original_data.loc[selected_user] > 0].index
            top_n_ids = user_scores.drop(seen_items, errors='ignore').nlargest(num_recs)

            top_n_titles = top_n_ids.copy()
            top_n_titles.index = top_n_titles.index.map(lambda mid: get_movie_title(mid, movie_titles_df))

            top_n_df = top_n_titles.to_frame(name="Predicted Score")

            # --- MODIFIED BLOCK: Add characteristics column ---
            is_generated_dataset = results.get('user_profiles') is not None and results.get('Q_matrix') is not None
            if is_generated_dataset:
                q_matrix = results.get('Q_matrix')

                # top_n_ids.index contains the item names/IDs
                item_names_to_lookup = top_n_ids.index

                # Filter Q_matrix to only the recommended items
                q_subset = q_matrix.loc[item_names_to_lookup]

                # Function to get features
                def get_feature_string(item_row):
                    features = item_row[item_row == 1].index.tolist()
                    return ", ".join(features)

                # Apply function row-wise
                top_n_df["Known Item Characteristics"] = q_subset.apply(get_feature_string, axis=1)
            # --- END MODIFIED BLOCK ---

            st.subheader(f"Top {num_recs} Recommendations for User {selected_user}")
            st.dataframe(top_n_df.style.format({"Predicted Score": "{:.2f}"}), use_container_width=True)


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


def render_visualizations_tab(results, selected_run_dir):
    """
    Renders the Visualizations tab content.
    'results' dict provides context (algo_name),
    'selected_run_dir' specifies which directory to load.
    """
    st.header("Training Visualizations")
    algo_name = results.get('algo_name')
    # Use the passed-in selected_run_dir

    if not algo_name or not selected_run_dir or not os.path.isdir(selected_run_dir):
        st.info(f"Visualizations directory not found for this run: {selected_run_dir}")
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
            st.exception(traceback.format_exc())
    else:
        st.info(f"No specific visualization renderer defined or found for {algo_name}.")

def update_state(source_widget_key, target_state_key):
    """Callback to sync a widget's state with a session_state key."""
    st.session_state[target_state_key] = st.session_state[source_widget_key]


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


def render_visualizations_tab(results, selected_run_dir):
    """
    Renders the Visualizations tab content.
    'results' dict provides context (algo_name),
    'selected_run_dir' specifies which directory to load.
    """
    st.header("Training Visualizations")
    algo_name = results.get('algo_name')
    # Use the passed-in selected_run_dir

    if not algo_name or not selected_run_dir or not os.path.isdir(selected_run_dir):
        st.info(f"Visualizations directory not found for this run: {selected_run_dir}")
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
            st.exception(traceback.format_exc())
    else:
        st.info(f"No specific visualization renderer defined or found for {algo_name}.")

def update_state(source_widget_key, target_state_key):
    """Callback to sync a widget's state with a session_state key."""
    st.session_state[target_state_key] = st.session_state[source_widget_key]