import streamlit as st
import os
import json
from .BPRVisualizationRenderer import BPRVisualizationRenderer
from visualization import generic_renderers

class WMFBPRVisualizationRenderer(BPRVisualizationRenderer):
    """
    Renders visualizations specific to WMFBPR.
    Overrides the parent render method to include the
    global item weights histogram.
    """
    def __init__(self, run_dir, explanations):
        """Initialize the renderer."""
        super().__init__(run_dir, explanations)
        self.algorithm_name = "WMFBPR"

        self.explanations.update({
            "Factor Change": "Shows the Frobenius norm of the change in user (P) and item (Q) latent factor matrices between iterations. A decreasing trend indicates convergence.",
            "Snapshots": "These plots visualize the distribution and relationships within the latent factor matrices (P and Q) at key iterations.",
            "AUC": "Shows the **Area Under Curve (AUC)** on a validation set. A rising curve indicates the model is learning to rank correctly.",
            "TSNE": "This plot shows a 2D t-SNE projection of the user (P) and item (Q) embedding vectors, visualizing the 'interest map' the model has learned.",
            "Item Weights": "This histogram shows the distribution of the global item importance weights (w_i) calculated using PageRank. These weights are added to the item vectors during score calculation, boosting the rank of 'important' items."
        })

    def _render_plots(self, manifest):
        self.show_factor_auc_2column(manifest)
        self.show_latent_factor_snapshots(manifest)
        self.show_item_weights_histogram_plot(manifest)
        self.show_tsne_plot(manifest)
        self.show_breakdown_plot(manifest)