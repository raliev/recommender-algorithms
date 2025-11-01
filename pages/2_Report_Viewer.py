import streamlit as st
import pandas as pd
import os
import glob
import json
from datetime import datetime

st.set_page_config(page_title="Report Viewer", layout="wide")
st.title("Hyperparameter Tuning Report Viewer")

REPORT_DIR = "reports"
METRIC_COLS = ['rmse', 'mae', 'r2', 'mape', 'explained_variance', 'precision', 'recall', 'k', 'runtime', 'type']

def load_report_data(report_dir):
    """Loads CSVs and params from a selected report directory."""
    best_hp_path = os.path.join(report_dir, "best-hp.csv")
    full_report_path = os.path.join(report_dir, "full_report.csv")

    data = {
        "best_hp": None,
        "full_report": None,
        "run_time": os.path.basename(report_dir).replace("report-", "")
    }

    try:
        if os.path.exists(best_hp_path):
            data["best_hp"] = pd.read_csv(best_hp_path)
        if os.path.exists(full_report_path):
            data["full_report"] = pd.read_csv(full_report_path)
    except Exception as e:
        st.error(f"Error loading report files: {e}")

    return data

def find_reports(report_dir):
    """Finds all report directories."""
    if not os.path.isdir(report_dir):
        return []

    report_dirs = glob.glob(os.path.join(report_dir, "report-*"))
    # Sort by directory name (timestamp) descending
    return sorted(report_dirs, reverse=True)

def display_report_df(df, title):
    st.subheader(title)
    if df is not None:
        # Identify which metric columns are actually in this dataframe
        met_cols_present = [col for col in METRIC_COLS if col in df.columns]
        # Identify hyperparameter columns (anything not metric and not 'algorithm')
        hp_cols_present = [col for col in df.columns if col not in METRIC_COLS and col != 'algorithm']

        # Reorder for clarity
        display_cols = ['algorithm'] + hp_cols_present + met_cols_present

        st.dataframe(df[display_cols].style.format(precision=4))
    else:
        st.info(f"No data found for '{title}'.")

report_list = find_reports(REPORT_DIR)

if not report_list:
    st.warning(f"No reports found in the '{REPORT_DIR}' directory. Run the 'Hyperparameter Tuning' page to generate reports.")
    st.stop()

# Create display names from directory basenames
report_options = {os.path.basename(d): d for d in report_list}
selected_report_name = st.selectbox("Select a Report to View:", list(report_options.keys()))

if selected_report_name:
    selected_dir = report_options[selected_report_name]
    report_data = load_report_data(selected_dir)

    st.header(f"Report: {report_data['run_time']}")

    display_report_df(report_data["best_hp"], "Best Hyperparameters & Performance")

    with st.expander("View Full Report (All Trials)"):
        display_report_df(report_data["full_report"], "Full Trial Report")