# visualization/renderers/wmfbpr_renderer.py
import streamlit as st
import os
import json
from .BPRVisualizationRenderer import BPRVisualizationRenderer
from visualization import generic_renderers

class WMFBPRVisualizationRenderer(BPRVisualizationRenderer): # Inherit from BPR
    """
    Renders visualizations specific to WMFBPR.
    Overrides the parent render method to include the
    global item weights histogram.
    """
    def __init__(self, run_dir, explanations):
        """Initialize the renderer."""
        super().__init__(run_dir, explanations) # Pass explanations to base
        self.algorithm_name = "WMFBPR" # Set algorithm name

        # Add/Update BPR-specific explanations here
        self.explanations.update({
            "Factor Change": "Shows the Frobenius norm of the change in user (P) and item (Q) "
                             "latent factor matrices between iterations. A decreasing trend "
                             "indicates convergence.",
            "Snapshots": "These plots visualize the distribution and relationships within "
                         "the latent factor matrices (P and Q) at key iterations.",
            "AUC": "Shows the **Area Under Curve (AUC)** on a validation set. "
                   "A rising curve indicates the model is learning to rank correctly.",
            "TSNE": "This plot shows a 2D t-SNE projection of the user (P) and item (Q) "
                    "embedding vectors, visualizing the 'interest map' the model has learned.",
            "Item Weights": "This histogram shows the distribution of the global item "
                            "importance weights (w_i) calculated using PageRank. "
                            "These weights are added to the item vectors during "
                            "score calculation, boosting the rank of 'important' items."
        })

    def render(self):
        """Overrides the parent render to add the item weights histogram."""
        st.header(f"WMFBPR Visualizations for {self.run_timestamp}")
        st.write(f"Displaying visualizations for algorithm: {self.algorithm_name}")

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

        if not manifest:
            st.info("No visualizations were generated for this run.")
            return

        # --- Render Convergence (from parent) ---
        st.subheader("Convergence Plots")
        factor_plot = next((e for e in manifest if e["type"] == "line_plot" and e["interpretation_key"] == "Factor Change"), None)
        auc_plot = next((e for e in manifest if e["type"] == "line_plot" and e["interpretation_key"] == "AUC"), None)

        col1, col2 = st.columns(2)
        if factor_plot:
            generic_renderers.render_line_plot(self.run_dir, factor_plot, self.explanations, col1)
        else:
            col1.info("No factor convergence plot found.")
        if auc_plot:
            generic_renderers.render_line_plot(self.run_dir, auc_plot, self.explanations, col2)
        else:
            col2.info("No AUC convergence plot found.")

        st.divider()

        # --- Render Snapshots (from parent) ---
        st.subheader("Latent Factor Snapshots")
        # (This logic is copied from BPRVisualizationRenderer's inherited render method)
        snapshots = [e for e in manifest if e["type"] == "factor_snapshot"]
        snapshots.sort(key=lambda x: x["iteration"])
        if snapshots:
            first_snapshot, last_snapshot = snapshots[0], snapshots[-1]
            if first_snapshot["iteration"] == last_snapshot["iteration"]:
                st.subheader(f"Snapshot: Iteration {first_snapshot['iteration']}")
                generic_renderers.render_factor_snapshot(
                    self.run_dir, first_snapshot, self.explanations, st)
            else:
                st.subheader(f"Snapshot Comparison: Iteration {first_snapshot['iteration']} vs Iteration {last_snapshot['iteration']}")
                col_snap1, col_snap2 = st.columns(2)
                generic_renderers.render_factor_snapshot(
                    self.run_dir, first_snapshot, self.explanations, col_snap1)
                generic_renderers.render_factor_snapshot(
                    self.run_dir, last_snapshot, self.explanations, col_snap2)
        else:
            st.info("No latent factor snapshots found.")

        st.divider()

        # --- NEW SECTION: Item Weights ---
        st.subheader("Global Item Weights (PageRank)")
        weights_plot = next((e for e in manifest if e["type"] == "histogram" and e["interpretation_key"] == "Item Weights"), None)

        if weights_plot:
            generic_renderers.render_histogram(self.run_dir, weights_plot, self.explanations, st)
        else:
            st.info("No item weights histogram plot found.")

        st.divider()

        # --- Render t-SNE (from parent) ---
        st.subheader("Embedding t-SNE Plot")
        # (This logic is copied from BPRVisualizationRenderer's render method)
        tsne_plots = [e for e in manifest if e["type"] == "tsne_plot"]
        tsne_plots.sort(key=lambda x: x["iteration"])
        if tsne_plots:
            first_tsne, last_tsne = tsne_plots[0], tsne_plots[-1]
            if first_tsne["iteration"] == last_tsne["iteration"]:
                st.subheader(f"t-SNE Plot: Iteration {first_tsne['iteration']}")
                generic_renderers.render_tsne_plot(
                    self.run_dir, first_tsne, self.explanations, st)
            else:
                st.subheader(f"t-SNE Comparison: Iteration {first_tsne['iteration']} vs Iteration {last_tsne['iteration']}")
                col_tsne1, col_tsne2 = st.columns(2)
                generic_renderers.render_tsne_plot(
                    self.run_dir, first_tsne, self.explanations, col_tsne1)
                generic_renderers.render_tsne_plot(
                    self.run_dir, last_tsne, self.explanations, col_tsne2)
        else:
            st.info("No t-SNE plots found.")

        st.divider()

        # --- Render Breakdown (from parent) ---
        st.subheader("Recommendation Breakdown")
        # (This logic is copied from BPRVisualizationRenderer's inherited render method)
        breakdown_plot = next((e for e in manifest if e["type"] == "recommendation_breakdown"), None)
        if breakdown_plot:
            generic_renderers.render_recommendation_breakdown(
                self.run_dir, breakdown_plot, self.explanations, st
            )
        else:
            st.info("No recommendation breakdown plot was generated for this run.")