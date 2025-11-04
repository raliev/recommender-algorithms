import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import json
from datetime import datetime, time, timedelta

st.set_page_config(page_title="All Lab Runs", layout="wide")
st.title("All Recent Lab Runs")
st.info("This page lists all runs from the **Lab** page. Runs from the **Hyperparameter Tuning** page are in the **Report Viewer**.")

VISUALS_DIR = "visuals"
METRIC_COLS = ['rmse', 'mae', 'r2', 'precision', 'recall', 'k', 'runtime']

@st.cache_data(ttl=60)
def load_all_runs(visuals_dir):
    """
    Scans all algorithm subdirectories in visuals_dir to load
    run data, parameters, and metrics.
    """
    all_runs = []
    if not os.path.isdir(visuals_dir):
        return pd.DataFrame()

    try:
        algo_folders = glob.glob(os.path.join(visuals_dir, "*"))
    except Exception as e:
        st.error(f"Error reading visuals directory '{visuals_dir}': {e}")
        return pd.DataFrame()

    for algo_path in algo_folders:
        if not os.path.isdir(algo_path):
            continue

        algo_name = os.path.basename(algo_path)
        run_folders = glob.glob(os.path.join(algo_path, "*"))

        for run_path in run_folders:
            if not os.path.isdir(run_path):
                continue

            run_timestamp_str = os.path.basename(run_path)
            try:
                run_timestamp = datetime.strptime(run_timestamp_str, '%Y%m%d_%H%M%S')
            except ValueError:
                continue  # Skip folders that aren't timestamps

            run_data = {'algorithm': algo_name, 'timestamp': run_timestamp}
            params_path = os.path.join(run_path, 'params.json')
            metrics_path = os.path.join(run_path, 'metrics.json')

            # Load parameters
            if os.path.exists(params_path):
                try:
                    with open(params_path, 'r') as f:
                        params = json.load(f)
                    run_data['data_source'] = params.get('data_source', 'N/A')

                    data_params = params.get('data_params', {})
                    if isinstance(data_params, dict):
                        run_data.update(data_params)

                    algo_params = {k: v for k, v in params.items() if k not in ['algorithm', 'data_source', 'data_params', 'timestamp']}
                    run_data.update(algo_params)
                except Exception:
                    run_data['data_source'] = 'Error'
            else:
                run_data['data_source'] = 'N/A'
            # Load metrics
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    run_data.update(metrics)
                except Exception:
                    pass

            all_runs.append(run_data)

    if not all_runs:
        return pd.DataFrame()

    df = pd.DataFrame(all_runs)
    return df

# --- Main Page ---
df = load_all_runs(VISUALS_DIR)

if df.empty:
    st.warning(f"No lab runs found in the '{VISUALS_DIR}' directory. Please run some experiments on the 'Lab' page first.")
    st.stop()

# --- Filters ---
st.header("Filters")
col1, col2 = st.columns(2)

with col1:
    all_algos = sorted(df['algorithm'].unique())
    selected_algos = st.multiselect("Algorithm", all_algos, default=all_algos)

    all_datasets = sorted(df['data_source'].unique())
    selected_datasets = st.multiselect("Dataset", all_datasets, default=all_datasets)

with col2:
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()

    from_date = st.date_input("From Date", min_date, min_value=min_date, max_value=max_date)
    to_date = st.date_input("To Date", max_date, min_value=min_date, max_value=max_date)

# Convert dates to datetimes for filtering
try:
    from_datetime = datetime.combine(from_date, time.min)
    to_datetime = datetime.combine(to_date, time.max)
except Exception:
    st.error("Invalid date range.")
    st.stop()

# Apply filters
filtered_df = df[
    (df['algorithm'].isin(selected_algos)) &
    (df['data_source'].isin(selected_datasets)) &
    (df['timestamp'] >= from_datetime) &
    (df['timestamp'] <= to_datetime)
    ].copy()

# --- Display Table ---
st.header("Filtered Runs")

if filtered_df.empty:
    st.info("No runs match the current filter criteria.")
else:
    # Define column order
    base_cols = ['timestamp', 'algorithm', 'data_source']

    # Find which metric and param columns are present in the filtered set
    present_metrics = [col for col in METRIC_COLS if col in filtered_df.columns]

    # All other columns are parameters
    param_cols = [col for col in filtered_df.columns if col not in base_cols + present_metrics + ['type']]

    # Combine and ensure no duplicates and all columns exist
    display_cols = base_cols + present_metrics + param_cols
    display_cols = [col for col in display_cols if col in filtered_df.columns]

    # Format for display
    display_df = filtered_df[display_cols].sort_values('timestamp', ascending=False)

    # Format floating point numbers for better readability
    float_cols = display_df.select_dtypes(include='float').columns
    style_format = {col: "{:.4f}" for col in float_cols if col not in ['precision', 'recall']}
    if 'precision' in style_format:
        style_format['precision'] = "{:.2%}"
    if 'recall' in style_format:
        style_format['recall'] = "{:.2%}"

    st.dataframe(display_df.style.format(style_format, na_rep="N/A"))

    if st.button("Copy to Clipboard (TSV)"):
        try:
            display_df.to_clipboard(index=False, sep='\t')
            st.success("Table copied to clipboard in TSV format (tab-separated), ready to paste into Excel or Google Sheets.")
        except Exception as e:
            st.error(f"Could not copy to clipboard. You may need to install 'pandas.io.clipboard.PyperclipClipboard'. Error: {e}")