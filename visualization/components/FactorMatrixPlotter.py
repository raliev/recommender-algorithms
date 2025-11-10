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

        # Plot Heatmaps only if matrices are not None
        if P is not None:
            heatmap_p_dict = self._plot_heatmap(P, f'User Factors P (Iter {iter_num})', f'P_iter_{iter_num}.png')
            if heatmap_p_dict: # Check if plotting was successful
                plot_files["heatmap_P"] = heatmap_p_dict["file"]
        if Q is not None:
            heatmap_q_dict = self._plot_heatmap(Q.T, f'Item Factors Q.T (Iter {iter_num})', f'Q_T_iter_{iter_num}.png')
            if heatmap_q_dict:
                plot_files["heatmap_Q"] = heatmap_q_dict["file"]
        if Y is not None:
            heatmap_y_dict = self._plot_heatmap(Y.T, f'Implicit Item Factors Y.T (Iter {iter_num})', f'Y_T_iter_{iter_num}.png')
            if heatmap_y_dict:
                plot_files["heatmap_Y"] = heatmap_y_dict["file"]

        # Plot Histograms only if matrices are not None
        hist_file = self._plot_histograms(P, Q, f'Latent Factor Histograms (Iter {iter_num})', f'hist_iter_{iter_num}.png', Y=Y)
        if hist_file: # Check if plotting was successful
            plot_files["histogram"] = os.path.basename(hist_file)

        # Plot Latent Space only if k=2 AND both P and Q are not None
        if self.k_factors == 2 and P is not None and Q is not None:
            latent_file = self._plot_latent_space(P, Q, f'2D Latent Space (Iter {iter_num})', f'latent_space_iter_{iter_num}.png')
            if latent_file: # Check if plotting was successful
                plot_files["latent_2d"] = os.path.basename(latent_file)

        # Return manifest entry only if some plots were generated
        if plot_files:
            return { #
                "name": f"Snapshot: Iteration {iter_num}",
                "type": "factor_snapshot",
                "iteration": iter_num,
                "files": plot_files,
                "interpretation_key": interpretation_key
            }
        else:
            return None # Return None if no plots were generated


    def _plot_heatmap(self, matrix, title, filename, sample_size=50):
        if matrix is None:
            print(f"Warning: Skipping heatmap '{title}' because input matrix is None.")
            return None

        fig, ax = plt.subplots(figsize=(10, 8))
        data_to_plot = matrix
        plot_title = title
        if sample_size and matrix.ndim == 2 and matrix.shape[0] > sample_size and matrix.shape[1] > sample_size:
            row_indices = np.random.choice(matrix.shape[0], sample_size, replace=False)
            col_indices = np.random.choice(matrix.shape[1], sample_size, replace=False)
            data_to_plot = matrix[np.ix_(row_indices, col_indices)]
            plot_title += f" (Sample {sample_size}x{sample_size})"
        elif matrix.ndim != 2:
            print(f"Warning: Skipping heatmap sampling for '{title}' as matrix is not 2D (shape: {matrix.shape}).")
            if matrix.ndim == 1:
                try:
                    if matrix.shape[0] > 1: data_to_plot = matrix[:, np.newaxis]
                    else: data_to_plot = matrix[np.newaxis, :]
                    print(f"Reshaped 1D array to {data_to_plot.shape} for heatmap.")
                except:
                    ax.text(0.5, 0.5, 'Invalid matrix shape for heatmap', ha='center', va='center')
                    file_path = self._save_plot(fig, filename)
                    return None

        if data_to_plot is None or not isinstance(data_to_plot, np.ndarray) or data_to_plot.ndim != 2:
            print(f"Warning: Final data for heatmap '{title}' is invalid. Skipping plot.")
            plt.close(fig)
            return None

        sns.heatmap(data_to_plot, cmap="viridis", ax=ax)
        ax.set_title(plot_title)
        ax.set_xlabel("Latent Factors" if data_to_plot.shape[1] == self.k_factors else "Items/Users")
        ax.set_ylabel("Users/Items" if data_to_plot.shape[1] == self.k_factors else "Latent Factors")
        plt.tight_layout()

        file_path = self._save_plot(fig, filename)

        return {
            "name": plot_title,
            "type": "similarity_heatmap",
            "file": os.path.basename(file_path),
            "interpretation_key": None
        }


    def _plot_histograms(self, P, Q, title, filename, Y=None):
        available_matrices = {}
        if P is not None: available_matrices['P (User Factors)'] = P
        if Q is not None: available_matrices['Q (Item Factors)'] = Q
        if Y is not None: available_matrices['Y (Implicit Item Factors)'] = Y

        if not available_matrices:
            print(f"Warning: Skipping histograms '{title}' as no valid matrices provided.")
            return None

        num_plots = len(available_matrices)
        fig_width = 6 * num_plots
        fig, axes = plt.subplots(1, num_plots, figsize=(fig_width, 5), squeeze=False)

        ax_idx = 0
        for name, matrix in available_matrices.items():
            ax = axes[0, ax_idx]
            try:
                values = matrix.flatten()
                if values.size > 0:
                    ax.hist(values, bins=50, alpha=0.7)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            except AttributeError:
                ax.text(0.5, 0.5, 'Invalid data', ha='center', va='center')

            ax.set_title(f'{name} Value Distribution')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax_idx += 1


        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return self._save_plot(fig, filename)

    def _plot_latent_space(self, P, Q, title, filename):
        if P is None or Q is None or self.k_factors != 2:
            return None

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
        return self._save_plot(fig, filename) # Return the file path