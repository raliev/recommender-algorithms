import streamlit as st
import os
import json
from .WRMFVisualizationRenderer import WRMFVisualizationRenderer
from visualization import generic_renderers

class BPRAdaptiveVisualizationRenderer(WRMFVisualizationRenderer):
    """
    Renders visualizations specific to BPR (Adaptive).
    Overrides the render method to show the new negative score plot.
    """
    def __init__(self, run_dir, explanations):
        """
        Initialize the renderer.
        """
        super().__init__(run_dir, explanations)
        self.algorithm_name = "BPR (Adaptive)"
        self.run_timestamp = os.path.basename(run_dir)

        self.explanations.update({
            "Objective": "BPR typically minimizes a pairwise ranking loss. This plot might not be available.",
            "Factor Change": "Shows the Frobenius norm of the change in user (P) and item (Q) latent factor matrices between iterations. A decreasing trend indicates convergence.",
            "Snapshots": "These plots visualize the distribution and relationships within the latent factor matrices (P and Q) at key iterations.",
            "Avg. Negative Score": "Shows the average predicted score (x_uj) of the 'hardest' negative item sampled during each epoch. A downward trend shows the model is learning to rank even these hard negatives lower, making it 'harder' to find bad predictions.",
            "AUC": "Shows the **Area Under Curve (AUC)** on a validation set. AUC measures the model's ability to correctly rank a random positive item higher than a random negative item. A value of 1.0 is perfect, 0.5 is random. A rising curve indicates the model is learning to rank correctly.",
            "TSNE": "This plot shows a 2D t-SNE projection of the user (P) and item (Q) embedding vectors. It visualizes the 'interest map' the model has learned. Clusters of items/users indicate similarity in the latent space."
        })

    def _render_plots(self, manifest):
        """Overrides the parent render to add the new convergence plot."""
        self.show_factor_auc_plots_3column(manifest)
        self.show_latent_factor_snapshots(manifest)
        self.show_tsne_plot(manifest)
        self.show_breakdown_plot(manifest)