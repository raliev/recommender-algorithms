# visualization/visualizers/EASEVisualizer.py
import os
from .ItemKNNVisualizer import ItemKNNVisualizer
from visualization.components.SimilarityMatrixPlotter import SimilarityMatrixPlotter
from visualization.components.RecommendationBreakdownPlotter import RecommendationBreakdownPlotter

class EASEVisualizer(ItemKNNVisualizer):
    """
    Specific visualizer for EASE.
    Inherits from ItemKNNVisualizer to reuse the recommendation breakdown logic.
    Plots the learned B matrix (heatmap + distribution) and a breakdown.
    """
    def __init__(self, k, **kwargs):
        # Call the parent's __init__ (ItemKNNVisualizer)
        super().__init__(**kwargs)

        self.k = k # Store k for the breakdown plot

        # --- CRITICAL: Override algorithm name and visuals_dir ---
        self.algorithm_name = "EASE"
        self.visuals_dir = os.path.join(self.visuals_base_dir, self.algorithm_name, self.run_timestamp)
        os.makedirs(self.visuals_dir, exist_ok=True)

        # --- Re-instantiate plotters with the correct directory ---
        self.similarity_plotter = SimilarityMatrixPlotter(self.visuals_dir)
        self.breakdown_plotter = RecommendationBreakdownPlotter(self.visuals_dir)

    def visualize_fit_results(self, B, R, params=None):
        """
        Called once by EASE.fit() to save all visualization artifacts.


        Args:
            B (np.ndarray): The final learned B (item-item) matrix.
            R (np.ndarray): The training User-Item matrix.
            params (dict): Dictionary of hyperparameters to save.
        """
        # 1. Start the run, create directory, save params
        self.start_run(params)
        self.visuals_manifest = []

        # 2. Plot Heatmap of the B matrix
        manifest_entry_heatmap = self.similarity_plotter.plot_heatmap(
            matrix=B,
            title="EASE Similarity Matrix (B) (Sampled)",
            filename="B_matrix_heatmap.png",
            interpretation_key="Final Similarity Heatmap"
        )
        self.visuals_manifest.append(manifest_entry_heatmap)

        # 3. Plot Histogram of B matrix coefficients
        manifest_entry_hist = self.similarity_plotter.plot_histogram(
            matrix=B,
            title="Distribution of B Matrix Coefficients (Non-Zero)",
            filename="B_matrix_histogram.png",
            interpretation_key="Histogram of Final Similarity Values"
        )
        self.visuals_manifest.append(manifest_entry_hist)

        # 4. Plot Recommendation Breakdown
        # We reuse the breakdown logic from ItemKNN/SLIM
        # The prediction is R @ B
        self._plot_recommendation_breakdown(R, B)

        # 5. Finalize
        self.params_saved['iterations_run'] = 1 # Mark as 1 "step"
        self._save_params()
        self._save_history() # Saves an empty history.json
        self._save_visuals_manifest()