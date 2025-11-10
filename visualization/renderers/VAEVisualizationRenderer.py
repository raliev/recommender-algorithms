import streamlit as st
import os
import json

from visualization.renderers.BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers

class VAEVisualizationRenderer(BaseVisualizationRenderer):
    def __init__(self, run_dir, explanations):
        super().__init__(run_dir, explanations)
        self.algorithm_name = "VAE"
        self.run_timestamp = os.path.basename(run_dir)
        self.explanations.update({
            "Objective": "Shows the VAE loss (Reconstruction Loss + KL Divergence) over epochs. A decreasing trend indicates convergence.",
            "Latent Distribution": "This plot shows the distribution of the learned latent variable 'μ' (mean) vectors, compared to the 'prior' (a standard N(0,1) normal distribution). A good VAE learns a distribution that is close to the prior, which encourages a well-structured latent space.",
            "Reconstruction Heatmap": "Compares a batch of original user-item interactions (binarized) against the model's reconstructed output scores for that same batch. This shows how well the model learned to recreate its input."
        })

    def _render_plots(self, manifest):
        self.show_convergence_plot(manifest)
        self.show_latent_space_distribution(manifest)
        self.show_original_reconstructed_plot(manifest)