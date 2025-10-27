import os

from .BasePlotter import BasePlotter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class SimilarityMatrixPlotter(BasePlotter):
    """Plots heatmaps and histograms for similarity matrices (KNN, SLIM, etc.)."""

    def plot_heatmap(self, matrix, title, filename, interpretation_key, sample_size=100):
        fig, ax = plt.subplots(figsize=(10, 8))
        data_to_plot = matrix
        if sample_size and matrix.shape[0] > sample_size and matrix.shape[1] > sample_size:
            indices = np.random.choice(matrix.shape[0], sample_size, replace=False)
            data_to_plot = matrix[np.ix_(indices, indices)]
            title += f" (Sample {sample_size}x{sample_size})"

        sns.heatmap(data_to_plot, cmap="viridis", vmin=np.min(data_to_plot), vmax=np.max(data_to_plot), ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Items")
        ax.set_ylabel("Items")
        plt.tight_layout()

        file_path = self._save_plot(fig, filename)

        # Return manifest entry
        return {
            "name": title,
            "type": "similarity_heatmap",
            "file": os.path.basename(file_path),
            "interpretation_key": interpretation_key
        }

    def plot_histogram(self, matrix, title, filename, interpretation_key):
        fig, ax = plt.subplots(figsize=(8, 5))
        values = matrix[np.nonzero(matrix)]
        values = values[~np.isnan(values)]

        if len(values) > 0:
            sns.histplot(values, bins=50, kde=True, ax=ax)
        else:
            ax.text(0.5, 0.5, 'No non-zero values found.', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        ax.set_title(title)
        ax.set_xlabel('Similarity Value')
        ax.set_ylabel('Frequency')
        plt.tight_layout()

        file_path = self._save_plot(fig, filename)

        # Return manifest entry
        return {
            "name": title,
            "type": "histogram",
            "file": os.path.basename(file_path),
            "interpretation_key": interpretation_key
        }