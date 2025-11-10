import streamlit as st
from utils import cleanup_old_reports
import os  # Import os to check for file existence

st.set_page_config(
    page_title="Recommender System Lab",
    layout="wide"
)

@st.cache_data
def load_readme():
    """Loads the README.md file from the root directory."""
    readme_path = "README.md"
    if not os.path.exists(readme_path):
        return "Error: README.md file not found. Could not load introduction."
    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading README.md: {e}"

if 'cleanup_done' not in st.session_state:
    try:
        deleted_count = cleanup_old_reports(max_age_hours=24)
        if deleted_count > 0:
            st.toast(f"🧹 Cleaned up {deleted_count} old reports/visuals.")
    except Exception as e:
        st.toast(f"Cleanup failed: {e}")
    st.session_state.cleanup_done = True

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

    st.title("Recommender System Laboratory & Tuner")
    st.info("""Select a page from the sidebar to get started:
        - **Lab**: Interactively train and visualize a single algorithm.
        - **Hyperparameter Tuning**: Run automated (Optuna) tuning for one or more algorithms.
        - **Dataset Wizard**: Create and save your own synthetic datasets.
        - **Report Viewer**: View the results from past tuning runs.
        """)

readme_content = load_readme()
st.markdown(readme_content)
