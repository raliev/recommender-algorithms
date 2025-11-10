import streamlit as st
import os
import json
from .WRMFVisualizationRenderer import WRMFVisualizationRenderer
from visualization import generic_renderers

class BPRVisualizationRenderer(WRMFVisualizationRenderer):
    """
    Renders visualizations specific to BPR.
    Overrides the parent _render_plots method to include AUC and t-SNE plots.
    """
    def __init__(self, run_dir, explanations):
        """
        Initialize the renderer.
        """
        super().__init__(run_dir, explanations)
        self.algorithm_name = "BPR"
        self.run_timestamp = os.path.basename(run_dir)

        self.explanations.update({
            "Factor Change": "Shows the Frobenius norm of the change in user (P) and item (Q) latent factor matrices between iterations. A decreasing trend indicates convergence.",
            "Snapshots": "These plots visualize the distribution and relationships within the latent factor matrices (P and Q) at key iterations.",
            "AUC": "Shows the **Area Under Curve (AUC)** on a validation set. AUC measures the model's ability to correctly rank a random positive item higher than a random negative item. A value of 1.0 is perfect, 0.5 is random. A rising curve indicates the model is learning to rank correctly.",
            "TSNE": "This plot shows a 2D t-SNE projection of the user (P) and item (Q) embedding vectors. It visualizes the 'interest map' the model has learned. Clusters of items/users indicate similarity in the latent space."
        })

    def _render_plots(self, manifest):
        self.show_factor_auc_2column(manifest)
        self.show_latent_factor_snapshots(manifest)
        self.show_tsne_plot(manifest)
        self.show_breakdown_plot(manifest)