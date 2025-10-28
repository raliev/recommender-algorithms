import streamlit as st
import os
import json

from visualization.renderers.BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers

class VAEVisualizationRenderer(BaseVisualizationRenderer):
    """
    Renders visualizations specific to VAE, reading from visuals.json
    and utilizing generic rendering components.
    """

    def __init__(self, run_dir, explanations):
        """Initialize the renderer."""
        super().__init__(run_dir, explanations)
        self.algorithm_name = "VAE"
        self.run_timestamp = os.path.basename(run_dir)
        self.explanations.update({
            "Objective": "Shows the VAE loss (Reconstruction Loss + KL Divergence) over epochs. A decreasing trend indicates convergence.",
            "Latent Distribution": "This plot shows the distribution of the learned latent variable 'μ' (mean) vectors, compared to the 'prior' (a standard N(0,1) normal distribution). A good VAE learns a distribution that is close to the prior, which encourages a well-structured latent space.",
            "Reconstruction Heatmap": "Compares a batch of original user-item interactions (binarized) against the model's reconstructed output scores for that same batch. This shows how well the model learned to recreate its input."
        })

    def render(self):
        st.header(f"VAE Visualizations for {self.run_timestamp}")
        st.write(f"Displaying visualizations for algorithm: {self.algorithm_name}")

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

        # --- Render Convergence Plot ---
        st.subheader("Convergence Plot")
        objective_plot = next((e for e in manifest if e["type"] == "line_plot" and e["interpretation_key"] == "Objective"), None)

        if objective_plot:
            generic_renderers.render_line_plot(self.run_dir, objective_plot, self.explanations, st)
        else:
            st.info("No loss convergence plot found for VAE.")

        st.divider()

        # --- Render Latent Distribution Plot ---
        st.subheader("Latent Space Distribution")
        dist_plot = next((e for e in manifest if e["type"] == "latent_distribution"), None)

        if dist_plot:
            generic_renderers.render_latent_distribution_plot(self.run_dir, dist_plot, self.explanations, st)
        else:
            st.info("No latent distribution plot found.")

        st.divider()

        # --- Render Reconstruction Example ---
        st.subheader("Reconstruction Example (Last Batch)")
        original_plot = next((e for e in manifest if e["type"] == "similarity_heatmap" and "Original" in e["name"]), None)
        recon_plot = next((e for e in manifest if e["type"] == "similarity_heatmap" and "Reconstructed" in e["name"]), None)

        col1, col2 = st.columns(2)
        if original_plot:
            generic_renderers.render_similarity_heatmap(self.run_dir, original_plot, self.explanations, col1)
        else:
            col1.info("Original batch heatmap not found.")

        if recon_plot:
            generic_renderers.render_similarity_heatmap(self.run_dir, recon_plot, self.explanations, col2)
        else:
            col2.info("Reconstructed batch heatmap not found.")