# visualization/visualizers/SLIMVisualizer.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .ItemKNNVisualizer import ItemKNNVisualizer
from visualization.components.RecommendationBreakdownPlotter import RecommendationBreakdownPlotter
from visualization.components.SimilarityMatrixPlotter import SimilarityMatrixPlotter
from visualization.components.SparsityPatternPlotter import SparsityPatternPlotter


class SLIMVisualizer(ItemKNNVisualizer):
    """
    Specific visualizer for SLIM.
    Inherits from ItemKNNVisualizer but implements SLIM-specific plots
    (sparsity, coefficient distribution, and recommendation breakdown).
    """
    def __init__(self, k, **kwargs):
        """
        Initialize the visualizer.
        'k' is required for the Top-K recommendation plot.
        """
        # Call the parent's __init__
        super().__init__(k=k, **kwargs)
        # Store k for our own plotting needs
        self.k = k
        # Override the algorithm name and directory
        self.algorithm_name = "SLIM"
        self.visuals_dir = os.path.join(self.visuals_base_dir, self.algorithm_name, self.run_timestamp)
        self.sparsity_plotter = SparsityPatternPlotter(self.visuals_dir)
        self.hist_plotter = SimilarityMatrixPlotter(self.visuals_dir) # Reusing for histogram
        self.breakdown_plotter = RecommendationBreakdownPlotter(self.visuals_dir)

    def visualize_fit_results(self, W, R, params=None):
        """
        Called once by SLIM.fit() to save all visualization artifacts.
        Args:
            W (np.ndarray): The final learned sparse W (item-item) matrix.
            R (np.ndarray): The training User-Item matrix (as a dense array).
            params (dict): Dictionary of hyperparameters to save.
        """
        # 1. Start the run, create directory, save params
        self.start_run(params)
        self.visuals_manifest = []

        # 2. Plot and save visualizations (NOW USES COMPONENTS)
        self._plot_sparsity_pattern(W)
        self._plot_coefficient_distribution(W)

        # Pick a sample user to visualize
        user_interaction_counts = (R > 0).sum(axis=1)
        sample_user_id = np.where((user_interaction_counts >= 5) & (user_interaction_counts <= 15))[0]
        if len(sample_user_id) > 0:
            sample_user_id = sample_user_id[0]
        else:
            sample_user_id = np.argmax(user_interaction_counts)

        self._plot_single_user_recommendation(W, R, sample_user_id)

        # 3. Finalize
        self.params_saved['iterations_run'] = 1 # Mark as 1 "step"
        self._save_params()
        self._save_history() # Saves an empty history.json
        self._save_visuals_manifest()

    # --- Refactored Plotting Methods ---

    def _plot_sparsity_pattern(self, W, max_sample_size=200):
        """
        Visualizes the sparsity of the W matrix using SparsityPatternPlotter.
        """
        manifest_entry = self.sparsity_plotter.plot(
            matrix=W,
            title="Sparsity Pattern of W",
            filename="W_sparsity_pattern.png",
            interpretation_key="W Sparsity", # Key for renderer
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

        # The hist_plotter component expects the full matrix
        manifest_entry = self.hist_plotter.plot_histogram(
            matrix=W, # Pass the full matrix, plotter finds non-zeros
            title=f'Distribution of Non-Zero W Coefficients (Total: {non_zero_coeffs.size})',
            filename='W_coefficient_distribution.png',
            interpretation_key='W Distribution' # Key for renderer
        )
        # Note: The component uses 'Frequency' on y-axis, not log-scale.
        # This is a simplification for component reuse.
        self.visuals_manifest.append(manifest_entry)

    def _plot_single_user_recommendation(self, W, R, user_id, num_items_to_show=50):
        """
        Creates a multi-panel plot using RecommendationBreakdownPlotter.
        """
        try:
            R_u_dense = R[user_id, :]
            liked_item_indices = np.where(R_u_dense > 0)[0]
            R_tilde_u = R_u_dense.dot(W).ravel()

            if liked_item_indices.size == 0:
                print(f"Warning: Sample user {user_id} has no interactions. Skipping recommendation plot.")
                return

            num_items = R.shape[1]
            item_names = [f"Item {i}" for i in range(num_items)]

            manifest_entry = self.breakdown_plotter.plot(
                user_history_vector=R_u_dense,
                result_vector=R_tilde_u,
                item_names=item_names,
                user_id=str(user_id),
                k=self.k, # Use the k from __init__
                filename="single_user_recommendation.png",
                interpretation_key="SLIM Recommendation",
                max_items_to_show=num_items_to_show
            )
            self.visuals_manifest.append(manifest_entry)

        except Exception as e:
            print(f"Error plotting single user recommendation: {e}")