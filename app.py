import streamlit as st
st.set_page_config(
    page_title="Recommender System Lab",
    layout="wide"
)

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
st.info("Select a page from the sidebar to get started.")