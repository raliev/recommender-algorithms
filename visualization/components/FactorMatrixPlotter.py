# visualization/components/FactorMatrixPlotter.py
import os

from .BasePlotter import BasePlotter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class FactorMatrixPlotter(BasePlotter):
    """Plots factor matrix heatmaps and histograms."""
    def __init__(self, visuals_dir, k_factors):
        super().__init__(visuals_dir)
        self.k_factors = k_factors

    def plot_snapshot(self, P, Q, iter_num, interpretation_key, Y=None):
        """Plots heatmap, histogram, and (optionally) latent space."""

        plot_files = {} # To store generated filenames

        # Plot Heatmaps
        heatmap_p_file = self._plot_heatmap(P, f'User Factors P (Iter {iter_num})', f'P_iter_{iter_num}.png')
        heatmap_q_file = self._plot_heatmap(Q.T, f'Item Factors Q.T (Iter {iter_num})', f'Q_T_iter_{iter_num}.png')
        plot_files["heatmap_P"] = os.path.basename(heatmap_p_file)
        plot_files["heatmap_Q"] = os.path.basename(heatmap_q_file)

        if Y is not None:
            heatmap_y_file = self._plot_heatmap(Y.T, f'Implicit Item Factors Y.T (Iter {iter_num})', f'Y_T_iter_{iter_num}.png')
            plot_files["heatmap_Y"] = os.path.basename(heatmap_y_file)

        hist_file = self._plot_histograms(P, Q, f'Latent Factor Histograms (Iter {iter_num})', f'hist_iter_{iter_num}.png', Y=Y) 
        plot_files["histogram"] = os.path.basename(hist_file)

        # Plot 2D Latent Space (if k=2) - No change needed here
        if self.k_factors == 2:
            latent_file = self._plot_latent_space(P, Q, f'2D Latent Space (Iter {iter_num})', f'latent_space_iter_{iter_num}.png')
            plot_files["latent_2d"] = os.path.basename(latent_file)

        # Return manifest entry
        return { # 
            "name": f"Snapshot: Iteration {iter_num}",
            "type": "factor_snapshot",
            "iteration": iter_num,
            "files": plot_files,
            "interpretation_key": interpretation_key
        }

    def _plot_heatmap(self, matrix, title, filename, sample_size=50): # 
        fig, ax = plt.subplots(figsize=(10, 8))
        data_to_plot = matrix
        if sample_size and matrix.shape[0] > sample_size and matrix.shape[1] > sample_size:
            row_indices = np.random.choice(matrix.shape[0], sample_size, replace=False)
            col_indices = np.random.choice(matrix.shape[1], sample_size, replace=False)
            data_to_plot = matrix[np.ix_(row_indices, col_indices)]
            title += f" (Sample {sample_size}x{sample_size})"

        sns.heatmap(data_to_plot, cmap="viridis", ax=ax) # 
        ax.set_title(title)
        ax.set_xlabel("Latent Factors" if data_to_plot.shape[1] == self.k_factors else "Items/Users")
        ax.set_ylabel("Users/Items" if data_to_plot.shape[1] == self.k_factors else "Latent Factors")
        plt.tight_layout()
        return self._save_plot(fig, filename)

    def _plot_histograms(self, P, Q, title, filename, Y=None):
        # --- Adjust subplot layout based on Y ---
        num_plots = 3 if Y is not None else 2
        fig_width = 18 if Y is not None else 12
        fig, axes = plt.subplots(1, num_plots, figsize=(fig_width, 5))
        # --- End Adjust ---

        ax1 = axes[0]
        ax1.hist(P.flatten(), bins=50, alpha=0.7)
        ax1.set_title('User Factors (P) Value Distribution') # 
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')

        ax2 = axes[1]
        ax2.hist(Q.flatten(), bins=50, alpha=0.7)
        ax2.set_title('Item Factors (Q) Value Distribution')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')

        if Y is not None:
            ax3 = axes[2]
            ax3.hist(Y.flatten(), bins=50, alpha=0.7)
            ax3.set_title('Implicit Item Factors (Y) Value Distribution')
            ax3.set_xlabel('Value')
            ax3.set_ylabel('Frequency')

        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return self._save_plot(fig, filename)

    def _plot_latent_space(self, P, Q, title, filename): # 
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(P[:, 0], P[:, 1], alpha=0.5, label='Users')
        ax.scatter(Q[:, 0], Q[:, 1], alpha=0.5, label='Items')
        # Note: We don't plot Y here as it's also item factors, would clutter.
        ax.set_title(title)
        ax.set_xlabel('Latent Factor 1')
        ax.set_ylabel('Latent Factor 2')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        return self._save_plot(fig, filename)