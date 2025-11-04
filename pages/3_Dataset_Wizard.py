import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import glob
from sklearn.preprocessing import MinMaxScaler
PREDEFINED_USERS = [f"User_{i+1:03d}" for i in range(500)] # Pool of 500 users
BASE_GT_PATH = "datasets/ground_truth"

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
        st.session_state.selected_ground_truth = "recipes_and_tastes"

# --- Page UI ---

st.set_page_config(page_title="Dataset Wizard", layout="wide")
st.title("🧙‍♂️ Dataset Generation Wizard")

# Initialize state
initialize_session_state()

# --- Step 1: Select Ground Truth Dataset ---
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
    st.rerun()

# Load the selected ground truth data
ALL_FEATURES, ALL_ITEMS, ITEM_FEATURE_MAP = load_ground_truth_data(selected_gt_name)
if not ALL_FEATURES or not ALL_ITEMS:
    st.error(f"Failed to load data for '{selected_gt_name}'. Check file integrity.")
    st.stop()

Q_full = get_full_item_feature_matrix(ITEM_FEATURE_MAP, ALL_ITEMS, ALL_FEATURES)

# --- Step 2: Configure Dataset Dimensions ---
st.header("Step 2: Configure Dataset Dimensions")

cols_config = st.columns(3)
with cols_config[0]:
    num_users = st.number_input("Number of Users", min_value=1, max_value=len(PREDEFINED_USERS), value=min(50, len(PREDEFINED_USERS)), step=10, key="wiz_num_users")
with cols_config[1]:
    num_items = st.number_input("Number of Items", min_value=1, max_value=len(ALL_ITEMS), value=min(20, len(ALL_ITEMS)), step=1, key="wiz_num_items")
with cols_config[2]:
    num_features = st.number_input("Number of Latent Features", min_value=1, max_value=len(ALL_FEATURES), value=min(10, len(ALL_FEATURES)), step=1, key="wiz_num_features")

# --- Step 3: Define Preferences & Features ---
st.header("Step 3: Define User Preferences and Item Features")

# Get the slices of data based on user inputs
user_names = PREDEFINED_USERS[:num_users]
item_names = ALL_ITEMS[:num_items]
feature_names = ALL_FEATURES[:num_features]

# Update Q_matrix in session state
st.session_state.Q_matrix = Q_full.loc[item_names, feature_names]

# Update or initialize P_matrix in session state
# Check if the existing P_matrix matches the desired shape
current_p_df = st.session_state.get('P_matrix', pd.DataFrame())
if current_p_df.shape != (num_users, num_features) or list(current_p_df.index) != user_names or list(current_p_df.columns) != feature_names:
    # Create a new one, preserving old data if possible
    new_P = pd.DataFrame(
        np.zeros((num_users, num_features)),
        index=user_names,
        columns=feature_names
    )
    # Copy over matching cells
    common_rows = new_P.index.intersection(current_p_df.index)
    common_cols = new_P.columns.intersection(current_p_df.columns)
    if not common_rows.empty and not common_cols.empty:
        new_P.loc[common_rows, common_cols] = current_p_df.loc[common_rows, common_cols]
    st.session_state.P_matrix = new_P.fillna(0)


tab_p, tab_q = st.tabs(["User-Preference Matrix (P)", "Item-Feature Matrix (Q)"])

with tab_p:
    st.subheader("User-Preference Matrix (P)")
    st.markdown("Define how much each user likes each latent feature. Values from -1 (dislikes) to 1 (loves).")

    if st.button("Auto-fill User Preferences (Randomly)", key="fill_p"):
        # Fill with random values between -1 and 1
        st.session_state.P_matrix = pd.DataFrame(
            np.random.uniform(-1, 1, size=(num_users, num_features)),
            index=user_names,
            columns=feature_names
        )
        st.toast("User preferences randomized!")

    # Display editable data editor for P
    edited_p = st.data_editor(
        st.session_state.P_matrix,
        use_container_width=True,
        key="p_matrix_editor",
        num_rows="dynamic" # Allow adding/removing users if needed, though PREDEFINED_USERS is the cap
    )

    # Check if the matrix was changed by the editor
    if not edited_p.equals(st.session_state.P_matrix):
        st.session_state.P_matrix = edited_p
        # Clear downstream generated data
        st.session_state.R_ideal = pd.DataFrame()
        st.session_state.R_final = pd.DataFrame()

with tab_q:
    st.subheader("Item-Feature Matrix (Q)")
    st.markdown("This matrix is **fixed** based on the selected ground truth data. (1 = has feature, 0 = does not).")
    st.dataframe(st.session_state.Q_matrix, use_container_width=True)

# --- Step 4: Generate Ratings ---
st.header("Step 4: Generate, Adjust, and Save Dataset")

if st.button("Generate Ideal Ratings", type="primary", use_container_width=True):
    P = st.session_state.P_matrix.to_numpy()
    Q = st.session_state.Q_matrix.to_numpy()

    # R_raw = P @ Q.T
    R_raw = P @ Q.T

    # Scale to 0-5
    if R_raw.size > 0:
        scaler = MinMaxScaler(feature_range=(0, 5))
        # We need to scale the *entire* matrix, not column-wise
        R_scaled = scaler.fit_transform(R_raw.reshape(-1, 1)).reshape(R_raw.shape)

        st.session_state.R_ideal = pd.DataFrame(
            R_scaled,
            index=st.session_state.P_matrix.index,
            columns=st.session_state.Q_matrix.index
        )
        st.session_state.R_final = st.session_state.R_ideal.copy() # Initialize R_final
        st.toast("Ideal ratings generated!")
    else:
        st.error("Cannot generate ratings. Matrices are empty.")

if not st.session_state.R_ideal.empty:
    st.subheader("Adjust Final Ratings")

    cols_adjust = st.columns(2)
    with cols_adjust[0]:
        noise_level = st.slider("Rating Noise (Std. Deviation)", min_value=0.0, max_value=2.0, value=0.5, step=0.1, key="wiz_noise")
        st.markdown("Adds random Gaussian noise to the ideal ratings to make them less perfect.")

    with cols_adjust[1]:
        sparsity_pct = st.slider("Sparsity (Remove % of Ratings)", min_value=0, max_value=100, value=70, step=5, key="wiz_sparsity")
        st.markdown("Randomly removes a percentage of ratings to create a sparse matrix (sets them to 0).")

    # Apply adjustments
    R_noisy = st.session_state.R_ideal + np.random.normal(0, noise_level, st.session_state.R_ideal.shape)
    R_clipped = R_noisy.clip(0, 5)

    # Apply sparsity
    # Create a mask where True means "remove"
    sparsity_mask = np.random.rand(*R_clipped.shape) < (sparsity_pct / 100.0)
    R_final = R_clipped.mask(sparsity_mask, 0) # Set masked values to 0

    st.session_state.R_final = R_final

    st.subheader("Final Generated Ratings Matrix (R_final)")
    st.dataframe(st.session_state.R_final.style.format("{:.2f}").background_gradient(cmap='viridis', axis=None, vmin=0, vmax=5), use_container_width=True)

    # --- Step 5: Save Dataset ---
    st.subheader("Step 5: Save Dataset")

    dataset_name = st.text_input("Dataset Name (e.g., 'My_Test_Data')", key="wiz_dataset_name")

    if st.button("Save Dataset", key="wiz_save"):
        if not dataset_name:
            st.error("Please provide a dataset name.")
        else:
            try:
                # Sanitize name
                safe_name = "".join(c for c in dataset_name if c.isalnum() or c in ('_', '-')).rstrip()
                if not safe_name:
                    st.error("Invalid dataset name. Use letters, numbers, _, -.")
                    st.stop()

                save_dir = os.path.join("datasets", "generated", safe_name)
                os.makedirs(save_dir, exist_ok=True)

                # Save the final ratings matrix
                ratings_path = os.path.join(save_dir, "ratings.csv")
                # We save the pivoted matrix directly.
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