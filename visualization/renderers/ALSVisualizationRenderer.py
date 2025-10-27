# visualization/renderers/ALSVisualizationRenderer.py
import streamlit as st
import os
import json
from .FunkSVDVisualizationRenderer import FunkSVDVisualizationRenderer # Inherit layout
from visualization import generic_renderers

class ALSVisualizationRenderer(FunkSVDVisualizationRenderer): # Inherit from FunkSVD
    """
    Renders visualizations specific to standard ALS.
    Inherits layout from FunkSVDVisualizationRenderer and adds the objective plot. 
    """
    def __init__(self, run_dir, explanations):
        """Initialize the renderer."""
        super().__init__(run_dir, explanations) # Call parent __init__ 
        self.algorithm_name = "ALS" # Set correct name 
        # Update explanations for ALS
        self.explanations.update({
            "Objective": "Shows the Root Mean Squared Error (RMSE) calculated on the *observed ratings* in the training set over iterations. A decreasing trend indicates convergence.", # 
        "Factor Change": "Shows the Frobenius norm of the change in user (P) and item (Q) latent factor matrices between iterations. A decreasing trend indicates convergence.",
        "Snapshots": "These plots visualize the distribution and relationships within the latent factor matrices (P and Q) at key iterations."
        })

        def render(self):
            """Overrides the parent render to add the objective plot."""
            st.header(f"{self.algorithm_name} Visualizations for {self.run_timestamp}")
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

            # 1. Find and render Objective plot (RMSE for ALS)
            objective_plot = next((e for e in manifest if e["type"] == "line_plot" and e["interpretation_key"] == "Objective"), None)
            # 2. Find and render Factor Change plot
            factor_change_plot = next((e for e in manifest if e["type"] == "line_plot" and e["interpretation_key"] == "Factor Change"), None) # 

            col_conv1, col_conv2 = st.columns(2)
            if objective_plot:
                generic_renderers.render_line_plot(self.run_dir, objective_plot, self.explanations, col_conv1)
            else:
                col_conv1.info("No objective convergence plot found.")

            if factor_change_plot:
                generic_renderers.render_line_plot(self.run_dir, factor_change_plot, self.explanations, col_conv2) # 
            else:
                col_conv2.info("No factor change convergence plot found.") # 

            st.divider()

            # --- Render Snapshots (Using parent's logic via super() is complex, so replicate) ---
            snapshots = [e for e in manifest if e["type"] == "factor_snapshot"] # 
            snapshots.sort(key=lambda x: x["iteration"]) # 

            if snapshots:
                first_snapshot = snapshots[0] # 
                last_snapshot = snapshots[-1] if len(snapshots) > 1 else snapshots[0] # 

                if first_snapshot["iteration"] == last_snapshot["iteration"]:
                    st.subheader(f"Snapshot: Iteration {first_snapshot['iteration']}") # 
                    generic_renderers.render_factor_snapshot(self.run_dir, first_snapshot, self.explanations, st) # 
                else:
                    st.subheader(f"Snapshot Comparison: Iteration {first_snapshot['iteration']} vs Iteration {last_snapshot['iteration']}") # 
                    col_snap1, col_snap2 = st.columns(2)
                    generic_renderers.render_factor_snapshot(self.run_dir, first_snapshot, self.explanations, col_snap1) # 
                    generic_renderers.render_factor_snapshot(self.run_dir, last_snapshot, self.explanations, col_snap2) # 
            else:
                st.info("No latent factor snapshots found for ALS.") # 