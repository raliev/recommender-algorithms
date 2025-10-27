import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .BasePlotter import BasePlotter
from scipy.stats import norm

class LatentDistributionPlotter(BasePlotter):
    """
    Plots a histogram of the flattened values from a latent space (e.g.,
    a VAE's 'z' vectors) to check if it approximates a prior
    (e.g., a standard normal distribution).
    """

    def plot(self, latent_vectors: np.ndarray, title: str,
             filename: str, interpretation_key: str):
        """
        Generates and saves the distribution plot.

        Args:
            latent_vectors (np.ndarray): 2D array of latent vectors
                                         (e.g., n_samples x k_dims).
            title (str): The main title for the plot.
            filename (str): The filename to save the plot as.
            interpretation_key (str): The key for the manifest.

        Returns:
            dict: The manifest entry for this visualization.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        values = latent_vectors.flatten()

        if len(values) == 0:
            ax.text(0.5, 0.5, 'No latent vector data provided.',
                    ha='center', va='center')
            ax.set_title(title)
            file_path = self._save_plot(fig, filename)
            return {
                "name": title,
                "type": "latent_distribution",
                "file": os.path.basename(file_path),
                "interpretation_key": interpretation_key
            }

        # Plot the histogram of the data
        sns.histplot(values, bins=50, kde=False, ax=ax, stat='density',
                     label='Latent Value Distribution')

        # Fit a standard normal distribution (our prior) and plot it
        x = np.linspace(values.min(), values.max(), 200)
        p_normal = norm.pdf(x, 0, 1) # Standard Normal (mu=0, std=1)

        ax.plot(x, p_normal, 'r--', linewidth=2,
                label='Standard Normal Prior (N(0,1))')

        ax.set_title(title)
        ax.set_xlabel('Latent Dimension Value')
        ax.set_ylabel('Density')
        ax.legend()
        plt.tight_layout()

        file_path = self._save_plot(fig, filename)

        return {
            "name": title,
            "type": "latent_distribution",
            "file": os.path.basename(file_path),
            "interpretation_key": interpretation_key
        }