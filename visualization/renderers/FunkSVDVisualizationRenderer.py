# visualization/renderers/FunkSVDVisualizationRenderer.py
import streamlit as st
import os
import json

from visualization.renderers.BaseVisualizationRenderer import BaseVisualizationRenderer

from visualization import generic_renderers


class FunkSVDVisualizationRenderer(BaseVisualizationRenderer):
    """
    Renders visualizations specific to FunkSVD, now reading from visuals.json
    and utilizing generic rendering components.
    """

    def __init__(self, run_dir, explanations):
        """
        Initialize the renderer.
        Args:
            run_dir (str): The path to the specific run directory.
            explanations (dict): A dictionary of explanations loaded from markdown.
        """
        super().__init__(run_dir, explanations) # Pass explanations to base
        self.algorithm_name = "FunkSVD" # Set algorithm name
        self.run_timestamp = os.path.basename(run_dir) # Get timestamp from dir
        self.explanations.update({
            "Factor Change": "Shows the Frobenius norm of the change in user (P) and item (Q) latent factor matrices between iterations. A decreasing trend indicates convergence.", 
        "Snapshots": "These plots visualize the distribution and relationships within the latent factor matrices (P and Q) at key iterations. Heatmaps show factor magnitudes, histograms show value distributions, and 2D latent space plots show user/item embeddings if k=2." 
        # Add or override more FunkSVD-specific interpretations here
        })

    def render(self):
        st.header(f"Visualizations for {self.run_timestamp}")
        st.write(f"Displaying visualizations for algorithm: {self.algorithm_name}")

        manifest_path = os.path.join(self.run_dir, 'visuals.json')
        if not os.path.exists(manifest_path):
            st.warning(f"Visuals manifest 'visuals.json' not found in {self.run_dir}. Please ensure the visualizer generated the manifest.")
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

        # --- REFFACTORED LOGIC ---

        st.subheader("Convergence Plots")

        # 1. Find and render factor change convergence plot
        # FunkSVD typically only has factor change, not a general 'objective'
        factor_change_plot = next((e for e in manifest if e["type"] == "line_plot" and e["interpretation_key"] == "Factor Change"), None)

        if factor_change_plot:
            generic_renderers.render_line_plot(self.run_dir, factor_change_plot, self.explanations, st)
        else:
            st.info("No factor change convergence plot found.")

        st.divider()

        # 2. Find and render snapshots
        snapshots = [e for e in manifest if e["type"] == "factor_snapshot"]
        snapshots.sort(key=lambda x: x["iteration"]) # Sort by iteration number

        if snapshots:
            first_snapshot = snapshots[0]
            last_snapshot = snapshots[-1] if len(snapshots) > 1 else snapshots[0]

            if first_snapshot["iteration"] == last_snapshot["iteration"]:
                st.subheader(f"Snapshot: Iteration {first_snapshot['iteration']}")
                generic_renderers.render_factor_snapshot(self.run_dir, first_snapshot, self.explanations, st)
            else:
                st.subheader(f"Snapshot Comparison: Iteration {first_snapshot['iteration']} vs Iteration {last_snapshot['iteration']}")
                col_snap1, col_snap2 = st.columns(2)

                # Render first snapshot in col1
                generic_renderers.render_factor_snapshot(self.run_dir, first_snapshot, self.explanations, col_snap1)
                # Render last snapshot in col2
                generic_renderers.render_factor_snapshot(self.run_dir, last_snapshot, self.explanations, col_snap2)
        else:
            st.info("No latent factor snapshots found.")
