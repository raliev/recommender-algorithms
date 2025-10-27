# visualization/renderers/KNNVisualizationRenderer.py
import streamlit as st
import os
import json
from .BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers

class KNNVisualizationRenderer(BaseVisualizationRenderer):
    """Renders visualizations specific to KNN-based algorithms (ItemKNN, UserKNN)."""

    def render(self):
        st.subheader("Model Internals")

        manifest_path = os.path.join(self.run_dir, 'visuals.json')
        if not os.path.exists(manifest_path):
            st.warning(f"Visuals manifest 'visuals.json' not found.")
            return
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception as e:
            st.error(f"Error loading 'visuals.json': {e}")
            return

        # Find plots by their interpretation key (which we defined in the visualizer)
        raw_heatmap = next((e for e in manifest if e["interpretation_key"] == "Raw Similarity Heatmap"), None)
        final_heatmap = next((e for e in manifest if e["interpretation_key"] == "Final (Adjusted) Similarity Heatmap"), None)
        co_rated_heatmap = next((e for e in manifest if e["interpretation_key"] == "Co-rated Counts Heatmap"), None)
        final_hist = next((e for e in manifest if e["interpretation_key"] == "Histogram of Final Similarity Values"), None)

        col_s1, col_s2 = st.columns(2)
        if raw_heatmap:
            generic_renderers.render_similarity_heatmap(self.run_dir, raw_heatmap, self.explanations, col_s1)
        if final_heatmap:
            generic_renderers.render_similarity_heatmap(self.run_dir, final_heatmap, self.explanations, col_s2)

        st.divider()
        col_s3, col_s4 = st.columns(2)

        if co_rated_heatmap:
            generic_renderers.render_similarity_heatmap(self.run_dir, co_rated_heatmap, self.explanations, col_s3)
        else:
            col_s3.caption("Co-rated counts heatmap not generated/found.")

        if final_hist:
            generic_renderers.render_histogram(self.run_dir, final_hist, self.explanations, col_s4)

        st.divider()
        st.subheader("Recommendation Breakdown")

        breakdown_plot = next((e for e in manifest if e["type"] == "recommendation_breakdown"), None)

        if breakdown_plot:
            generic_renderers.render_recommendation_breakdown(
                self.run_dir, breakdown_plot, self.explanations, st
            )
        else:
            st.info("No recommendation breakdown plot was generated for this run.")
        #