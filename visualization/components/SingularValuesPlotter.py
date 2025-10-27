import os
import numpy as np
import matplotlib.pyplot as plt
from .BasePlotter import BasePlotter

class SingularValuesPlotter(BasePlotter):
    """
    Plots the "scree plot" of singular values and the cumulative
    variance explained.
    """
    def plot(self, singular_values: np.ndarray, k: int, title: str,
             filename: str, interpretation_key: str):
        """
        Generates and saves the scree plot.

        Args:
            singular_values (np.ndarray): 1D array of singular values,
                                          sorted descending.
            k (int): The number of factors (k) chosen, to mark on the plot.
            title (str): The main title for the plot.
            filename (str): The filename to save the plot as.
            interpretation_key (str): The key for the manifest.

        Returns:
            dict: The manifest entry for this visualization.
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Calculate variance explained
        variance_explained = (singular_values ** 2) / \
                             (singular_values ** 2).sum()
        cumulative_variance = np.cumsum(variance_explained)

        # Plot 1: Individual variance explained (Scree Plot)
        ax1.bar(range(1, len(variance_explained) + 1), variance_explained,
                alpha=0.7, align='center',
                label='Individual Variance Explained')
        ax1.set_xlabel('Principal Component (Singular Value Index)')
        ax1.set_ylabel('Variance Explained', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Plot 2: Cumulative variance
        ax2 = ax1.twinx()
        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                 'r-o', label='Cumulative Variance Explained')
        ax2.set_ylabel('Cumulative Variance', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Mark the chosen 'k'
        ax1.axvline(x=k, color='grey', linestyle='--',
                    label=f'Chosen k={k}')

        # Formatting
        fig.suptitle(title, fontsize=16)
        fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        file_path = self._save_plot(fig, filename)

        return {
            "name": title,
            "type": "scree_plot",
            "file": os.path.basename(file_path),
            "interpretation_key": interpretation_key
        }