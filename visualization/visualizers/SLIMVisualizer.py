import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .AlgorithmVisualizer import AlgorithmVisualizer

class SLIMVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for SLIM.
    Plots sparsity, coefficient distribution, and recommendation breakdown.
    """
    def __init__(self, k, **kwargs):
        super().__init__("SLIM")
        self.k = k # k is used for the breakdown plot

    def visualize_fit_results(self, W, R, params=None):
        """
        Called once by SLIM.fit() to save all visualization artifacts.
        """
        self.start_run(params)
        self.visuals_manifest = []

        self._plot_sparsity_pattern(W)
        self._plot_coefficient_distribution(W)
        self._plot_recommendation_breakdown(W, R)

        self.params_saved['iterations_run'] = 1
        self._save_params()
        self._save_history()
        self._save_visuals_manifest()

    def _plot_sparsity_pattern(self, W, max_sample_size=200):
        """
        Visualizes the sparsity of the W matrix using SparsityPatternPlotter.
        """
        manifest_entry = self.sparsity_plotter.plot(
            matrix=W,
            title="Sparsity Pattern of W",
            filename="W_sparsity_pattern.png",
            interpretation_key="W Sparsity",
            max_sample_size=max_sample_size
        )
        self.visuals_manifest.append(manifest_entry)

    def _plot_coefficient_distribution(self, W):
        """
        Plots a histogram of the non-zero coefficients in W
        using SimilarityMatrixPlotter.
        """
        non_zero_coeffs = W[W > 0]
        if non_zero_coeffs.size == 0:
            print("Warning: W matrix is all zeros. No distribution to plot.")
            return

        manifest_entry = self.similarity_plotter.plot_histogram(
            matrix=W,
            title=f'Distribution of Non-Zero W Coefficients (Total: {non_zero_coeffs.size})',
            filename='W_coefficient_distribution.png',
            interpretation_key='W Distribution'
        )
        self.visuals_manifest.append(manifest_entry)

    def _plot_recommendation_breakdown(self, W, R, num_items_to_show=50):
        """
        Creates a multi-panel plot using RecommendationBreakdownPlotter.
        """
        # 1. Use base helper to find the user
        user_idx, history_vec = self._find_sample_user(R)
        if user_idx is None:
            print("Warning: Could not find sample user. Skipping recommendation plot.")
            return

        # 2. Calculate the algorithm-specific score vector
        result_vec = history_vec.dot(W).ravel()

        num_items = R.shape[1]
        item_names = [f"Item {i}" for i in range(num_items)]

        # 3. Use base helper to plot
        manifest_entry = self.breakdown_plotter.plot(
            user_history_vector=history_vec,
            result_vector=result_vec,
            item_names=item_names,
            user_id=str(user_idx),
            k=self.k,
            filename="single_user_recommendation.png",
            interpretation_key="SLIM Recommendation",
            max_items_to_show=num_items_to_show
        )
        if manifest_entry:
            self.visuals_manifest.append(manifest_entry)