import streamlit as st
import os
import json
from .WRMFVisualizationRenderer import WRMFVisualizationRenderer
from visualization import generic_renderers

class BPRAdaptiveVisualizationRenderer(WRMFVisualizationRenderer): # Inherit from WRMF for similar layout
    """
    Renders visualizations specific to BPR (Adaptive).
    Overrides the render method to show the new negative score plot.
    """
    def __init__(self, run_dir, explanations):
        """
        Initialize the renderer.
        """
        super().__init__(run_dir, explanations) # Pass explanations to base
        self.algorithm_name = "BPR (Adaptive)" # Set algorithm name
        self.run_timestamp = os.path.basename(run_dir) # Get timestamp from dir

        self.explanations.update({
            "Objective": "BPR typically minimizes a pairwise ranking loss. This plot might not be available.",
            "Factor Change": "Shows the Frobenius norm of the change in user (P) and item (Q) latent factor matrices between iterations. A decreasing trend indicates convergence.",
            "Snapshots": "These plots visualize the distribution and relationships within the latent factor matrices (P and Q) at key iterations.",
            # NEW EXPLANATION
            "Avg. Negative Score": "Shows the average predicted score (x_uj) of the 'hardest' negative item sampled during each epoch. A downward trend shows the model is learning to rank even these hard negatives lower, making it 'harder' to find bad predictions.",
            # COPIED EXPLANATIONS
            "AUC": "Shows the **Area Under Curve (AUC)** on a validation set. AUC measures the model's ability to correctly rank a random positive item higher than a random negative item. A value of 1.0 is perfect, 0.5 is random. A rising curve indicates the model is learning to rank correctly.",
            "TSNE": "This plot shows a 2D t-SNE projection of the user (P) and item (Q) embedding vectors. It visualizes the 'interest map' the model has learned. Clusters of items/users indicate similarity in the latent space."
        })

    def render(self):
        """Overrides the parent render to add the new convergence plot."""
        st.header(f"BPR (Adaptive) Visualizations for {self.run_timestamp}")
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

        # --- BPR-Adaptive Layout ---
        st.subheader("Convergence Plots")

        # Find the plots
        factor_plot = next((e for e in manifest if e["type"] == "line_plot" and e["interpretation_key"] == "Factor Change"), None)
        neg_score_plot = next((e for e in manifest if e["type"] == "line_plot" and e["interpretation_key"] == "Avg. Negative Score"), None)
        auc_plot = next((e for e in manifest if e["type"] == "line_plot" and e["interpretation_key"] == "AUC"), None) # 1. FIND AUC PLOT

        # 2. CHANGE TO 3 COLUMNS
        col1, col2, col3 = st.columns(3)

        if factor_plot:
            generic_renderers.render_line_plot(self.run_dir, factor_plot, self.explanations, col1)
        else:
            col1.info("No factor convergence plot found.")

        if neg_score_plot:
            generic_renderers.render_line_plot(self.run_dir, neg_score_plot, self.explanations, col2)
        else:
            col2.info("No average negative score plot found.")

        # 3. RENDER AUC PLOT
        if auc_plot:
            generic_renderers.render_line_plot(self.run_dir, auc_plot, self.explanations, col3)
        else:
            col3.info("No AUC convergence plot found. (Note: This must be calculated and passed by the algorithm's 'fit' method).")


        st.divider()

        # --- Render Snapshots (Re-implementing from parent WRMFVisualizationRenderer) ---
        st.subheader("Latent Factor Snapshots")
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
                    self.run_dir, first_snapshot,
                    self.explanations, col_snap1)
                generic_renderers.render_factor_snapshot(
                    self.run_dir, last_snapshot,
                    self.explanations, col_snap2)
        else:
            st.info("No latent factor snapshots found.")

        st.divider()

        # 4. ADD T-SNE SECTION
        st.subheader("Embedding t-SNE Plot")
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
                    self.run_dir, first_tsne,
                    self.explanations, col_tsne1)
                generic_renderers.render_tsne_plot(
                    self.run_dir, last_tsne,
                    self.explanations, col_tsne2)
        else:
            st.info("No t-SNE plots found.")

        st.divider()

        #  Render Recommendation Breakdown (Re-implementing from parent WRMFVisualizationRenderer) ---
        st.subheader("Recommendation Breakdown")
        breakdown_plot = next((e for e in manifest if e["type"] == "recommendation_breakdown"), None)
        if breakdown_plot:
            generic_renderers.render_recommendation_breakdown(
                self.run_dir, breakdown_plot, self.explanations, st
            )
        else:
            st.info("No recommendation breakdown plot was generated for this run.")