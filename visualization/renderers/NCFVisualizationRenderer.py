# visualization/renderers/NCFVisualizationRenderer.py
import streamlit as st
import os
import json

from visualization.renderers.BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers

class NCFVisualizationRenderer(BaseVisualizationRenderer):
    """
    Renders visualizations specific to NCF, reading from visuals.json
    and utilizing generic rendering components.
    """

    def __init__(self, run_dir, explanations):
        """
        Initialize the renderer.
        """
        super().__init__(run_dir, explanations) # Pass explanations to base
        self.algorithm_name = "NCFNeuMF" # Set algorithm name
        self.run_timestamp = os.path.basename(run_dir) # Get timestamp from dir
        self.explanations.update({
            "Objective": "Shows the training loss (e.g., Binary Cross-Entropy) over epochs. A decreasing trend indicates convergence.",
            "Snapshots": "These plots visualize the distribution and relationships within the embedding matrices (User and Item) at key epochs."
        })

    def render(self):
        st.header(f"NCF Visualizations for {self.run_timestamp}")
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

        # --- Render Convergence Plots ---
        st.subheader("Convergence Plots")
        # NCF visualizer saves loss as 'Objective'
        objective_plot = next((e for e in manifest if e["type"] == "line_plot" and e["interpretation_key"] == "Objective"), None)

        if objective_plot:
            generic_renderers.render_line_plot(self.run_dir, objective_plot, self.explanations, st)
        else:
            st.info("No loss convergence plot found for NCF.")

        st.divider()

        # --- Render Snapshots (Copy from FunkSVD/CML) ---
        snapshots = [e for e in manifest if e["type"] == "factor_snapshot"]
        snapshots.sort(key=lambda x: x["iteration"])

        if snapshots:
            first_snapshot = snapshots[0]
            last_snapshot = snapshots[-1] if len(snapshots) > 1 else snapshots[0]

            if first_snapshot["iteration"] == last_snapshot["iteration"]:
                st.subheader(f"Embedding Snapshot: Epoch {first_snapshot['iteration']}")
                generic_renderers.render_factor_snapshot(self.run_dir, first_snapshot, self.explanations, st)
            else:
                st.subheader(f"Snapshot Comparison: Epoch {first_snapshot['iteration']} vs Epoch {last_snapshot['iteration']}")
                col_snap1, col_snap2 = st.columns(2)
                generic_renderers.render_factor_snapshot(self.run_dir, first_snapshot, self.explanations, col_snap1)
                generic_renderers.render_factor_snapshot(self.run_dir, last_snapshot, self.explanations, col_snap2)
        else:
            st.info("No embedding factor snapshots found for NCF.")