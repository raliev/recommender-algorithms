# File: visualization/renderers/PureSVDVisualizationRenderer.py
import streamlit as st
import os
import json
from .BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers

class PureSVDVisualizationRenderer(BaseVisualizationRenderer):
    """Renders visualizations specific to PureSVD from visuals.json."""

    def __init__(self, run_dir, explanations):
        """Initialize the renderer."""
        super().__init__(run_dir, explanations)
        self.algorithm_name = "PureSVD"
        self.run_timestamp = os.path.basename(run_dir)
        self.explanations.update({
            "Scree Plot": "The scree plot shows the variance explained (derived from the singular values) for each component, sorted in descending order. The 'elbow' (point of inflection) is often a good heuristic for selecting 'k'.",
            "Snapshots": "These plots visualize the distribution and relationships within the final latent factor matrices (P and Q) after decomposition."
        })

    def render(self):
        """Render the visualizations for PureSVD."""
        st.header(f"{self.algorithm_name} Visualizations for {self.run_timestamp}")

        manifest_path = os.path.join(self.run_dir, 'visuals.json')
        if not os.path.exists(manifest_path):
            st.warning(f"Visuals manifest 'visuals.json' not found in {self.run_dir}.")
            return

        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception as e:
            st.error(f"Error loading 'visuals.json': {e}")
            return

        if not manifest:
            st.info("No visualizations were generated for this run according to visuals.json.")
            return

        # --- Find plots in manifest ---
        scree_plot = next((e for e in manifest if e.get("type") == "scree_plot"), None)
        snapshot = next((e for e in manifest if e.get("type") == "factor_snapshot"), None)

        # --- Render Scree Plot ---
        st.subheader("Singular Value Analysis")
        if scree_plot:
            generic_renderers.render_scree_plot(self.run_dir, scree_plot, self.explanations, st)
        else:
            st.info("No singular value (scree) plot found.")

        st.divider()

        # --- Render Snapshots ---
        st.subheader("Final Factor Matrices")
        if snapshot:
            generic_renderers.render_factor_snapshot(self.run_dir, snapshot, self.explanations, st)
        else:
            st.info("No latent factor snapshots found.")