# File: visualization/generic_renderers.py
import streamlit as st
import os

def _render_image_with_interpretation(file_path, caption, explanation_text, column=None):
    """Generic helper to render one image."""
    container = column if column else st
    if os.path.exists(file_path):
        container.image(file_path, caption=caption)
        with container.expander("Interpretation"):
            st.markdown(explanation_text, unsafe_allow_html=True)
    else:
        container.caption(f"{caption} (File not found: {os.path.basename(file_path)})")

def render_line_plot(run_dir, manifest_entry, explanations, column=None):
    """Renders a single line_plot visualization."""
    file_path = os.path.join(run_dir, manifest_entry["file"])
    caption = manifest_entry["name"]
    interp_key = manifest_entry["interpretation_key"]
    interp_text = explanations.get(interp_key, f"Explanation for '{interp_key}' not found.")

    _render_image_with_interpretation(file_path, caption, interp_text, column)

def render_scree_plot(run_dir, manifest_entry, explanations, column=None):
    """Renders a single scree_plot visualization."""
    file_path = os.path.join(run_dir, manifest_entry["file"])
    caption = manifest_entry["name"]
    interp_key = manifest_entry["interpretation_key"]
    interp_text = explanations.get(interp_key, f"Explanation for '{interp_key}' not found.")

    _render_image_with_interpretation(file_path, caption, interp_text, column)

def render_similarity_heatmap(run_dir, manifest_entry, explanations, column=None):
    """Renders a single similarity_heatmap visualization."""
    # This is identical to render_line_plot, just a different name for clarity
    file_path = os.path.join(run_dir, manifest_entry["file"])
    caption = manifest_entry["name"]
    interp_key = manifest_entry["interpretation_key"]
    interp_text = explanations.get(interp_key, f"Explanation for '{interp_key}' not found.")

    _render_image_with_interpretation(file_path, caption, interp_text, column)

def render_histogram(run_dir, manifest_entry, explanations, column=None):
    """Renders a single histogram visualization."""
    file_path = os.path.join(run_dir, manifest_entry["file"])
    caption = manifest_entry["name"]
    interp_key = manifest_entry["interpretation_key"]
    interp_text = explanations.get(interp_key, f"Explanation for '{interp_key}' not found.")

    _render_image_with_interpretation(file_path, caption, interp_text, column)


def render_latent_distribution_plot(run_dir, manifest_entry, explanations, column=None):
    """Renders a single latent_distribution visualization."""
    file_path = os.path.join(run_dir, manifest_entry["file"])
    caption = manifest_entry["name"]
    interp_key = manifest_entry["interpretation_key"]
    interp_text = explanations.get(interp_key, f"Explanation for '{interp_key}' not found.")

    _render_image_with_interpretation(file_path, caption, interp_text, column)

def render_recommendation_breakdown(run_dir, manifest_entry,
                                    explanations, column=None):
    """Renders a single recommendation_breakdown visualization."""
    file_path = os.path.join(run_dir, manifest_entry["file"])
    caption = manifest_entry["name"]
    interp_key = manifest_entry.get("interpretation_key", "Error")
    interp_text = explanations.get(interp_key,
                                   f"Explanation for '{interp_key}' not found.")

    _render_image_with_interpretation(file_path, caption, interp_text, column)

def render_factor_snapshot(run_dir, manifest_entry, explanations, column=None):
    """Renders a factor_snapshot visualization (heatmaps, hist, latent space)."""
    container = column if column else st
    files = manifest_entry["files"]
    iter_num = manifest_entry["iteration"]
    interp_key = manifest_entry["interpretation_key"]
    interp_text = explanations.get(interp_key, f"Explanation for '{interp_key}' not found.")

    container.markdown(f"**Snapshot: Iteration {iter_num}**")
    with container.expander("Interpretation"):
        st.markdown(interp_text, unsafe_allow_html=True)

    # Render plots that exist in the manifest
    if "heatmap_P" in files:
        _render_image_with_interpretation(
            os.path.join(run_dir, files["heatmap_P"]),
            f"User Factors P (Iter {iter_num})", "", container
        )
    if "heatmap_Q" in files:
        _render_image_with_interpretation(
            os.path.join(run_dir, files["heatmap_Q"]),
            f"Item Factors Q.T (Iter {iter_num})", "", container
        )
    if "heatmap_Y" in files:
        _render_image_with_interpretation(
            os.path.join(run_dir, files["heatmap_Y"]),
            f"Implicit Item Factors Y.T (Iter {iter_num})", "", container # Use appropriate caption
        )
    if "histogram" in files:
        _render_image_with_interpretation(
            os.path.join(run_dir, files["histogram"]),
            f"Histograms (Iter {iter_num})", "", container
        )
    if "latent_2d" in files:
        _render_image_with_interpretation(
            os.path.join(run_dir, files["latent_2d"]),
            f"2D Latent Space (Iter {iter_num})", "", container
        )