import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import glob
from sklearn.preprocessing import MinMaxScaler
PREDEFINED_USERS = [f"User_{i+1:03d}" for i in range(5000)]
BASE_GT_PATH = "datasets/ground_truth"

SAMPLING_THRESHOLD = 5000

@st.cache_data
def load_ground_truth_datasets():
    """Scans the ground_truth directory for available dataset subfolders."""
    if not os.path.isdir(BASE_GT_PATH):
        return []
    try:
        dataset_names = [d for d in os.listdir(BASE_GT_PATH) if os.path.isdir(os.path.join(BASE_GT_PATH, d))]
        return sorted(dataset_names)
    except Exception as e:
        st.error(f"Error scanning for ground truth datasets: {e}")
        return []

@st.cache_data
def load_ground_truth_data(dataset_name):
    """Loads the features, items, and map for a selected ground truth dataset."""
    dataset_path = os.path.join(BASE_GT_PATH, dataset_name)

    try:
        # Load features
        features_path = os.path.join(dataset_path, "latent_characteristics.csv")
        features_df = pd.read_csv(features_path)
        all_features = features_df['Feature Dimension'].str.strip().tolist()

        # Load items
        items_path = os.path.join(dataset_path, "items.csv")
        items_df = pd.read_csv(items_path)
        all_items = items_df['Item Name'].str.strip().tolist()

        # Load map
        map_path = os.path.join(dataset_path, "item_feature_map.json")
        with open(map_path, 'r') as f:
            item_feature_map = json.load(f)

        return all_features, all_items, item_feature_map

    except FileNotFoundError as e:
        st.error(f"Error: Missing file in '{dataset_path}'. Could not find: {e.filename}")
        return [], [], {}
    except Exception as e:
        st.error(f"Error loading ground truth data for '{dataset_name}': {e}")
        return [], [], {}


def get_full_item_feature_matrix(item_map, all_items, all_features):
    """
    Creates the full item-feature matrix (Q_full) based on loaded data.
    Values are 1 if the feature is present, 0 otherwise.
    """
    num_items_total = len(all_items)
    num_features_total = len(all_features)

    # Create the full Q matrix
    Q_full = pd.DataFrame(0, index=all_items, columns=all_features, dtype=int)

    for item_name, feature_indices in item_map.items():
        # Ensure item is in our "all_items" list (in case map is stale)
        if item_name not in all_items:
            continue

        for feature_index in feature_indices:
            # Convert 1-based index from prompt to 0-based index for list
            if 0 <= feature_index - 1 < num_features_total:
                feature_name = all_features[feature_index - 1]
                Q_full.loc[item_name, feature_name] = 1
    return Q_full

def initialize_session_state():
    if 'P_matrix' not in st.session_state:
        st.session_state.P_matrix = pd.DataFrame()
    if 'Q_matrix' not in st.session_state:
        st.session_state.Q_matrix = pd.DataFrame()
    if 'R_ideal' not in st.session_state:
        st.session_state.R_ideal = pd.DataFrame()
    if 'R_final' not in st.session_state:
        st.session_state.R_final = pd.DataFrame()
    if 'selected_ground_truth' not in st.session_state:
        st.session_state.selected_ground_truth = "movies_and_genres"

    if 'sampled_item_names' not in st.session_state:
        st.session_state.sampled_item_names = []
    if 'all_available_features' not in st.session_state:
        st.session_state.all_available_features = []


def clear_generated_ratings():
    """
    Callback to clear ideal and final ratings when dimensions change.
    """
    st.session_state.R_ideal = pd.DataFrame()
    st.session_state.R_final = pd.DataFrame()


st.set_page_config(page_title="Dataset Wizard", layout="wide")
st.title("Dataset Generation Wizard")

initialize_session_state()

st.header("Step 1: Select Ground Truth Dataset")
st.info("Choose the set of items and latent features to build your dataset from.")

ground_truth_options = load_ground_truth_datasets()
if not ground_truth_options:
    st.error(f"No ground truth datasets found in '{BASE_GT_PATH}'. Please create at least one.")
    st.stop()

# Set default if current selection is invalid
if st.session_state.selected_ground_truth not in ground_truth_options:
    st.session_state.selected_ground_truth = ground_truth_options[0]

selected_gt_name = st.selectbox(
    "Available Ground Truth Datasets:",
    options=ground_truth_options,
    index=ground_truth_options.index(st.session_state.selected_ground_truth),
    key="wiz_gt_select"
)

# Update session state on change
if selected_gt_name != st.session_state.selected_ground_truth:
    st.session_state.selected_ground_truth = selected_gt_name
    # Clear all matrices if the ground truth changes
    st.session_state.P_matrix = pd.DataFrame()
    st.session_state.Q_matrix = pd.DataFrame()
    st.session_state.R_ideal = pd.DataFrame()
    st.session_state.R_final = pd.DataFrame()
    st.session_state.sampled_item_names = []
    st.session_state.all_available_features = []
    st.rerun()

# Load the selected ground truth data
with st.spinner(f"Loading '{selected_gt_name}' dataset..."):
    ALL_FEATURES, ALL_ITEMS, ITEM_FEATURE_MAP = load_ground_truth_data(selected_gt_name)
    if not ALL_FEATURES or not ALL_ITEMS:
        st.error(f"Failed to load data for '{selected_gt_name}'. Check file integrity.")
        st.stop()

    Q_full = get_full_item_feature_matrix(ITEM_FEATURE_MAP, ALL_ITEMS, ALL_FEATURES)

st.session_state.all_available_features = ALL_FEATURES
show_sampling = len(ALL_ITEMS) > SAMPLING_THRESHOLD
has_popular_feature = "Popular" in ALL_FEATURES
Q_filtered = Q_full.copy()
final_sampled_item_names = []

if show_sampling:
    with st.expander("Step 1.5: Sampling & Filtering (Large Dataset Detected)"):
        st.warning(f"This dataset has **{len(ALL_ITEMS)} items**, which is larger than the threshold of {SAMPLING_THRESHOLD}. Sampling is recommended.")

        cols_sampling = st.columns(3)
        with cols_sampling[0]:
            sample_size = st.number_input("Sample Size", min_value=1, max_value=len(ALL_ITEMS), value=5000, step=100, key="wiz_sample_size", on_change=clear_generated_ratings)
        with cols_sampling[1]:
            sampling_seed = st.number_input("Sampling Seed", min_value=0, value=42, step=1, key="wiz_sampling_seed", on_change=clear_generated_ratings)
        with cols_sampling[2]:
            only_popular = st.checkbox("Sample popular items only", key="wiz_only_popular", disabled=not has_popular_feature, on_change=clear_generated_ratings)
            if not has_popular_feature:
                st.caption("('Popular' feature not found in this dataset)")

        rng = np.random.RandomState(int(sampling_seed))

        if only_popular and has_popular_feature:
            Q_filtered = Q_filtered[Q_filtered["Popular"] == 1]
            st.info(f"Filtered to **{len(Q_filtered)}** popular items.")

        current_available_items = Q_filtered.index.tolist()

        if len(current_available_items) > sample_size:
            final_sampled_item_names = rng.choice(current_available_items, sample_size, replace=False).tolist()
            st.info(f"Sampling **{sample_size}** items from the filtered list.")
        else:
            final_sampled_item_names = current_available_items
            st.info(f"Using all **{len(final_sampled_item_names)}** items (filtered list is smaller than sample size).")
else:
    # If sampling is not shown, just use all items
    final_sampled_item_names = ALL_ITEMS

if st.session_state.sampled_item_names != final_sampled_item_names:
    st.session_state.sampled_item_names = final_sampled_item_names
    clear_generated_ratings()


st.header("Step 2: Configure Dataset Dimensions")

cols_config = st.columns(3)
with cols_config[0]:
    num_users = st.number_input("Number of Users", min_value=1, max_value=len(PREDEFINED_USERS), value=min(50, len(PREDEFINED_USERS)), step=10, key="wiz_num_users", on_change=clear_generated_ratings)
with cols_config[1]:
    num_items = st.number_input("Number of Items",
                                min_value=1,
                                max_value=len(st.session_state.sampled_item_names),
                                value=min(20, len(st.session_state.sampled_item_names)),
                                step=1,
                                key="wiz_num_items",
                                on_change=clear_generated_ratings)
with cols_config[2]:
    num_features = st.number_input("Number of Latent Features",
                                   min_value=1,
                                   max_value=len(st.session_state.all_available_features),
                                   value=min(10, len(st.session_state.all_available_features)),
                                   step=1,
                                   key="wiz_num_features",
                                   on_change=clear_generated_ratings)

st.header("Step 3: Define User Preferences and Item Features")

user_names = PREDEFINED_USERS[:num_users]
item_names = st.session_state.sampled_item_names[:num_items]
feature_names = st.session_state.all_available_features[:num_features]

st.session_state.Q_matrix = Q_full.loc[item_names, feature_names]

current_p_df = st.session_state.get('P_matrix', pd.DataFrame())
if current_p_df.shape != (num_users, num_features) or list(current_p_df.index) != user_names or list(current_p_df.columns) != feature_names:
    new_P = pd.DataFrame(
        np.zeros((num_users, num_features)),
        index=user_names,
        columns=feature_names
    )
    common_rows = new_P.index.intersection(current_p_df.index)
    common_cols = new_P.columns.intersection(current_p_df.columns)
    if not common_rows.empty and not common_cols.empty:
        new_P.loc[common_rows, common_cols] = current_p_df.loc[common_rows, common_cols]
    st.session_state.P_matrix = new_P.fillna(0)


tab_p, tab_q = st.tabs(["User-Preference Matrix (P)", "Item-Feature Matrix (Q)"])

with tab_p:
    st.subheader("User-Preference Matrix (P)")
    st.markdown("Define how much each user likes each latent feature. Values from -1 (dislikes) to 1 (loves).")

    st.markdown("##### 🤖 Auto-fill Controls")
    c1, c2, c3 = st.columns(3)
    with c1:
        contrast_level = st.slider(
            "Preference Contrast",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            key="wiz_p_contrast",
            help="Controls the strength of preferences. 0.0 = all preferences are neutral (0.0), 1.0 = maximum contrast (-1.0 or 1.0)."
        )
    with c2:
        max_peaks = max(1, num_features)
        num_peaks = st.slider(
            "Target # of Max Preferences",
            min_value=0,
            max_value=max_peaks,
            value=max(1, max_peaks // 4),
            step=1,
            key="wiz_p_peaks",
            help="The target number of 'loved' (max value) features for each user."
        )
    with c3:
        max_variability = max(0, int(num_features / 2))
        peak_variability = st.slider(
            "Preference Variability (Chaos)",
            min_value=0,
            max_value=max_variability,
            value=0,
            step=1,
            key="wiz_p_variability",
            help="How much the number of max preferences can vary per user. 0 = no variation (all users get 'Target #' peaks)."
        )

    if st.button("Auto-fill User Preferences (Randomly)", key="fill_p"):

        contrast = st.session_state.wiz_p_contrast
        target_peaks = st.session_state.wiz_p_peaks
        variability = st.session_state.wiz_p_variability

        P_matrix_np = np.zeros((num_users, num_features))

        low_val = -1.0 * contrast
        high_val = 1.0 * contrast

        for u in range(num_users):
            actual_num_peaks = target_peaks
            if variability > 0:
                chaos = np.random.randint(-variability, variability + 1)
                actual_num_peaks += chaos
            actual_num_peaks = np.clip(actual_num_peaks, 0, num_features)

            user_vector = np.full(num_features, low_val)

            if actual_num_peaks > 0:
                peak_indices = np.random.choice(num_features, actual_num_peaks, replace=False)
                user_vector[peak_indices] = high_val

            noise = np.random.normal(0, 0.05, num_features)
            user_vector = np.clip(user_vector + noise, -1.0, 1.0)

            P_matrix_np[u, :] = user_vector

        st.session_state.P_matrix = pd.DataFrame(
            P_matrix_np,
            index=user_names,
            columns=feature_names
        )
        st.toast("User preferences generated with new settings!")

    edited_p = st.data_editor(
        st.session_state.P_matrix,
        use_container_width=True,
        key="p_matrix_editor",
        num_rows="dynamic"
    )

    if not edited_p.equals(st.session_state.P_matrix):
        st.session_state.P_matrix = edited_p
        st.session_state.R_ideal = pd.DataFrame()
        st.session_state.R_final = pd.DataFrame()

with tab_q:
    st.subheader("Item-Feature Matrix (Q)")
    st.markdown("This matrix is **fixed** based on the selected ground truth data. (1 = has feature, 0 = does not).")
    st.dataframe(st.session_state.Q_matrix, use_container_width=True)

st.header("Step 4: Generate, Adjust, and Save Dataset")

st.subheader("Generation Settings")
cols_gen = st.columns(2)
with cols_gen[0]:
    dataset_type = st.radio(
        "Select the type of data to generate:",
        ["Explicit (1-5 Ratings)", "Implicit (0/1 Interactions)"],
        key="wiz_dataset_type",
        help="**Explicit:** Generates ratings from 0-5. '0' means 'missing'. \n\n**Implicit:** Generates interactions (0 or 1). '0' means 'no interaction'."
    )
with cols_gen[1]:
    interaction_threshold = st.slider(
        "Implicit Interaction Threshold",
        min_value=0.0, max_value=1.0, value=0.75, step=0.05,
        key="wiz_interaction_threshold",
        help="If 'Implicit' is chosen, scores above this probability (after scaling P@Q.T to 0-1) will become a '1' (interacted).",
        disabled=(dataset_type != "Implicit (0/1 Interactions)")
    )

if st.button("Generate Ideal Ratings", type="primary", use_container_width=True):
    P = st.session_state.P_matrix.to_numpy()
    Q = st.session_state.Q_matrix.to_numpy()

    R_raw = P @ Q.T

    if R_raw.size > 0:
        if dataset_type == "Explicit (1-5 Ratings)":
            scaler = MinMaxScaler(feature_range=(0, 5))
            R_scaled = scaler.fit_transform(R_raw.reshape(-1, 1)).reshape(R_raw.shape)
            st.session_state.R_ideal = pd.DataFrame(
                R_scaled,
                index=st.session_state.P_matrix.index,
                columns=st.session_state.Q_matrix.index
            )
        else: # "Implicit (0/1 Interactions)"
            scaler = MinMaxScaler(feature_range=(0, 1))
            R_prob = scaler.fit_transform(R_raw.reshape(-1, 1)).reshape(R_raw.shape)
            R_binary = (R_prob >= interaction_threshold).astype(int)
            st.session_state.R_ideal = pd.DataFrame(
                R_binary,
                index=st.session_state.P_matrix.index,
                columns=st.session_state.Q_matrix.index
            )
        st.session_state.R_final = st.session_state.R_ideal.copy()
        st.toast("Ideal ratings generated!")
    else:
        st.error("Cannot generate ratings. Matrices are empty.")

if not st.session_state.R_ideal.empty:
    st.subheader("Adjust Final Ratings")
    dataset_type = st.session_state.get("wiz_dataset_type", "Explicit (1-5 Ratings)")
    if dataset_type == "Explicit (1-5 Ratings)":
        cols_adjust = st.columns(2)
        with cols_adjust[0]:
            noise_level = st.slider("Rating Noise (Std. Deviation)", min_value=0.0, max_value=2.0, value=0.5, step=0.1, key="wiz_noise")
            st.markdown("Adds random Gaussian noise to the ideal ratings.")
        with cols_adjust[1]:
            sparsity_pct = st.slider("Sparsity (Remove % of Ratings)", min_value=0, max_value=100, value=70, step=5, key="wiz_sparsity")
            st.markdown("Randomly sets a percentage of all ratings to 0 (missing).")

        R_noisy = st.session_state.R_ideal + np.random.normal(0, noise_level, st.session_state.R_ideal.shape)
        R_clipped = R_noisy.clip(0, 5)

        sparsity_mask = np.random.rand(*R_clipped.shape) < (sparsity_pct / 100.0)
        R_final = R_clipped.mask(sparsity_mask, 0)

    else: # "Implicit (0/1 Interactions)"
        cols_adjust = st.columns(2)
        with cols_adjust[0]:
            noise_level = st.slider(
                "Interaction Noise (% to flip)", 0, 100, 10,
                key="wiz_noise_implicit",
                help="Percentage of all interactions (both 0s and 1s) to randomly flip."
            )
        with cols_adjust[1]:
            sparsity_pct = st.slider(
                "Sparsity (Remove % of *Positive* Ratings)", 0, 100, 70,
                key="wiz_sparsity_implicit",
                help="Randomly removes this percentage of '1's, setting them to '0'."
            )

        R_ideal_np = st.session_state.R_ideal.to_numpy()

        noise_mask = np.random.rand(*R_ideal_np.shape) < (noise_level / 100.0)
        R_noisy = np.where(noise_mask, 1 - R_ideal_np, R_ideal_np)

        positive_indices = np.where(R_noisy == 1)
        num_positives_to_remove = int(len(positive_indices[0]) * (sparsity_pct / 100.0))

        R_final_np = R_noisy.copy()
        if num_positives_to_remove > 0:
            indices_to_remove = np.random.choice(len(positive_indices[0]), num_positives_to_remove, replace=False)

            sparse_mask_rows = positive_indices[0][indices_to_remove]
            sparse_mask_cols = positive_indices[1][indices_to_remove]

            R_final_np[sparse_mask_rows, sparse_mask_cols] = 0

        R_final = pd.DataFrame(R_final_np, index=st.session_state.R_ideal.index, columns=st.session_state.R_ideal.columns)

    st.session_state.R_final = R_final

    st.subheader("Final Generated Ratings Matrix (R_final)")

    R_final_df = st.session_state.R_final
    MAX_DISPLAY_ROWS = 100
    MAX_DISPLAY_COLS = 100

    R_display_df = R_final_df

    if R_final_df.shape[0] > MAX_DISPLAY_ROWS or R_final_df.shape[1] > MAX_DISPLAY_COLS:
        R_display_df = R_final_df.iloc[:MAX_DISPLAY_ROWS, :MAX_DISPLAY_COLS]
        st.info(f"Displaying a subset ({R_display_df.shape[0]} users, {R_display_df.shape[1]} items) of the full {R_final_df.shape[0]}x{R_final_df.shape[1]} matrix. The complete matrix will be used for saving.")

    if dataset_type == "Implicit (0/1 Interactions)":
        st.dataframe(R_display_df.style.format("{:.0f}"), use_container_width=True)
    else:
        st.dataframe(R_display_df.style.format("{:.2f}").background_gradient(cmap='viridis', axis=None, vmin=0, vmax=5), use_container_width=True)

    st.subheader("Step 5: Save Dataset")

    dataset_name = st.text_input("Dataset Name (e.g., 'My_Test_Data')", key="wiz_dataset_name")

    if st.button("Save Dataset", key="wiz_save"):
        if not dataset_name:
            st.error("Please provide a dataset name.")
        else:
            try:
                safe_name = "".join(c for c in dataset_name if c.isalnum() or c in ('_', '-')).rstrip()
                if not safe_name:
                    st.error("Invalid dataset name. Use letters, numbers, _, -.")
                    st.stop()

                save_dir = os.path.join("datasets", "generated", safe_name)
                os.makedirs(save_dir, exist_ok=True)

                ratings_path = os.path.join(save_dir, "ratings.csv")
                st.session_state.R_final.to_csv(ratings_path)

                # Save the ground truth P matrix
                p_path = os.path.join(save_dir, "ground_truth_user_profiles.csv")
                st.session_state.P_matrix.to_csv(p_path)

                # Save the ground truth Q matrix
                q_path = os.path.join(save_dir, "ground_truth_item_features.csv")
                st.session_state.Q_matrix.to_csv(q_path)

                # Save a movies.csv for compatibility
                movies_df = pd.DataFrame({
                    "movieId": st.session_state.Q_matrix.index,
                    "title": st.session_state.Q_matrix.index
                })
                movies_path = os.path.join(save_dir, "movies.csv")
                movies_df.set_index("movieId", inplace=True)
                movies_df.to_csv(movies_path)

                st.success(f"Dataset '{safe_name}' saved successfully in '{save_dir}'!")
                st.info("You can now select this dataset from the 'Lab' and 'Hyperparameter Tuning' pages (you may need to refresh).")

            except Exception as e:
                st.error(f"Failed to save dataset: {e}")

else:
    st.info("Click 'Generate Ideal Ratings' to create the dataset.")