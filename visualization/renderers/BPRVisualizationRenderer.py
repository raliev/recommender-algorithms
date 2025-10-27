# visualization/renderers/BPRVisualizationRenderer.py
import streamlit as st
import os
import json
# Import the parent renderer that handles factor-based visualizations
from .WRMFVisualizationRenderer import WRMFVisualizationRenderer
from visualization import generic_renderers # Still useful if you override and use specific renderers

class BPRVisualizationRenderer(WRMFVisualizationRenderer): # Inherit from WRMF for similar layout
    """
    Renders visualizations specific to BPR.
    It inherits from WRMFVisualizationRenderer because BPR also uses latent factors
    and often shows similar convergence and factor snapshot plots.

    The WRMFVisualizationRenderer's 'render' method will automatically adapt
    if BPR's 'visuals.json' doesn't contain certain plot types (e.g., objective function).
    """
    def __init__(self, run_dir, explanations):
        """
        Initialize the renderer.
        Args:
            run_dir (str): The path to the specific run directory.
            explanations (dict): A dictionary of explanations loaded from markdown.
        """
        super().__init__(run_dir, explanations) # Pass explanations to base
        self.algorithm_name = "BPR" # Set algorithm name
        self.run_timestamp = os.path.basename(run_dir) # Get timestamp from dir
        # You can add BPR-specific explanations here if needed
        self.explanations.update({
            "Objective": "BPR typically minimizes a pairwise ranking loss. This plot might not be available or might represent a different loss metric than WRMF's RMSE-like objective.", 
            "Factor Change": "Shows the Frobenius norm of the change in user (P) and item (Q) latent factor matrices between iterations. A decreasing trend indicates convergence.", 
            "Snapshots": "These plots visualize the distribution and relationships within the latent factor matrices (P and Q) at key iterations. Heatmaps show factor magnitudes, histograms show value distributions, and 2D latent space plots show user/item embeddings if k=2." 
            # Add or override more BPR-specific interpretations
            })

    # You could override the render method here if BPR needs a *fundamentally different*
    # layout or if you wanted to add BPR-specific sections.
    # However, for simply showing convergence and factor snapshots, the inherited
    # WRMFVisualizationRenderer.render() method should work well, as it's designed
    # to look for specific types in the manifest and render them if present.
    # For example, BPRVisualizer does not typically generate an 'objective' plot,
    # so WRMFVisualizationRenderer's logic will simply not find it and won't try to render it.

    # Example of how you *would* override if necessary:
    # def render(self):
    #     st.header(f"BPR Visualizations for {self.run_timestamp}")
    #     st.write(f"Displaying visualizations for algorithm: {self.algorithm_name}")

    #     manifest_path = os.path.join(self.run_dir, 'visuals.json')
    #     if not os.path.exists(manifest_path):
    #         st.warning(f"Visuals manifest 'visuals.json' not found.")
    #         return
    #     try:
    #         with open(manifest_path, 'r') as f:
    #             manifest = json.load(f)
    #     except Exception as e:
    #         st.error(f"Error loading 'visuals.json': {e}")
    #         return

    #     # --- BPR-Specific Layout ---
    #     st.subheader("Factor Convergence")
    #     factor_plot = next((e for e in manifest if e["type"] == "line_plot" and e["interpretation_key"] == "Factor Change"), None)
    #     if factor_plot:
    #         generic_renderers.render_line_plot(self.run_dir, factor_plot, self.explanations)
    #     else:
    #         st.info("No factor convergence plot found for BPR.")

    #     st.divider()

    #     st.subheader("Latent Factor Snapshots")
    #     snapshots = [e for e in manifest if e["type"] == "factor_snapshot"]
    #     snapshots.sort(key=lambda x: x["iteration"])

    #     if snapshots:
    #         # Render all snapshots, or just first/last as in WRMF
    #         for snapshot_entry in snapshots:
    #             st.markdown(f"**Iteration {snapshot_entry['iteration']}**")
    #             generic_renderers.render_factor_snapshot(self.run_dir, snapshot_entry, self.explanations)
    #     else:
    #         st.info("No latent factor snapshots found for BPR.")