import os
from .ItemKNNVisualizer import ItemKNNVisualizer

class EASEVisualizer(ItemKNNVisualizer):
    """
    Specific visualizer for EASE.
    Inherits from ItemKNNVisualizer to reuse the recommendation breakdown logic.
    Plots the learned B matrix (heatmap + distribution) and a breakdown.
    """
    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)

        self.k = k

        self.algorithm_name = "EASE"
        self.visuals_dir = os.path.join(self.visuals_base_dir, self.algorithm_name, self.run_timestamp)
        os.makedirs(self.visuals_dir, exist_ok=True)

    def visualize_fit_results(self, B, R, params=None):
        """
        Called once by EASE.fit() to save all visualization artifacts.
        """
        self.start_run(params)
        self.visuals_manifest = []

        manifest_entry_heatmap = self.similarity_plotter.plot_heatmap(
            matrix=B,
            title="EASE Similarity Matrix (B) (Sampled)",
            filename="B_matrix_heatmap.png",
            interpretation_key="Final Similarity Heatmap"
        )
        self.visuals_manifest.append(manifest_entry_heatmap)

        manifest_entry_hist = self.similarity_plotter.plot_histogram(
            matrix=B,
            title="Distribution of B Matrix Coefficients (Non-Zero)",
            filename="B_matrix_histogram.png",
            interpretation_key="Histogram of Final Similarity Values"
        )
        self.visuals_manifest.append(manifest_entry_hist)

        # This method is inherited from the refactored ItemKNNVisualizer
        self._plot_recommendation_breakdown(R, B)

        self.params_saved['iterations_run'] = 1
        self._save_params()
        self._save_history()
        self._save_visuals_manifest()