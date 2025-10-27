# visualization/components/SparsityPatternPlotter.py
import os
import numpy as np
import matplotlib.pyplot as plt
from .BasePlotter import BasePlotter

class SparsityPatternPlotter(BasePlotter):
    """
    Visualizes the sparsity of a matrix using plt.imshow().
    This is adapted from the logic in SLIMVisualizer .
    """

    def plot(self, matrix: np.ndarray, title: str, filename: str,
             interpretation_key: str, max_sample_size: int = 200):
        """
        Generates and saves the sparsity plot.

        Args:
            matrix (np.ndarray): The 2D matrix to plot.
            title (str): The main title for the plot.
            filename (str): The filename to save the plot as.
            interpretation_key (str): The key for the manifest.
            max_sample_size (int, optional): Sample the matrix if it's
                                             larger than this.

        Returns:
            dict: The manifest entry for this visualization.
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        matrix_to_plot = matrix
        plot_title = f'{title} ({matrix.shape[0]}x{matrix.shape[1]})'
        sparsity = 1.0 - (np.count_nonzero(matrix) / matrix.size)

        # Sample if too large
        if matrix.shape[0] > max_sample_size:
            idx = np.random.choice(matrix.shape[0], max_sample_size,
                                   replace=False)
            matrix_to_plot = matrix[np.ix_(idx, idx)]
            plot_title = (f'{title} (Sampled {max_sample_size}x'
                          f'{max_sample_size})')

        # Create a binary matrix: 1 for non-zero, 0 for zero
        matrix_binary = (matrix_to_plot != 0).astype(int)

        # Use plt.imshow. 'binary' cmap shows 0 as white and 1 as black
        ax.imshow(matrix_binary, cmap='binary',
                  interpolation='none', aspect='auto') 

        # Add grid lines if the matrix is small
        if matrix_to_plot.shape[0] <= 50:
            ax.set_xticks(np.arange(matrix_to_plot.shape[1]))
            ax.set_yticks(np.arange(matrix_to_plot.shape[0]))

            if matrix_to_plot.shape[0] <= 25:
                ax.set_xticklabels(np.arange(matrix_to_plot.shape[1]),
                                   rotation=90)
                ax.set_yticklabels(np.arange(matrix_to_plot.shape[0]))
                ax.tick_params(axis='both', which='major', labelsize=8)
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            ax.set_xticks(np.arange(matrix_to_plot.shape[1]+1)-.5,
                          minor=True)
            ax.set_yticks(np.arange(matrix_to_plot.shape[0]+1)-.5,
                          minor=True)
            ax.grid(which='minor', color='k',
                    linestyle='-', linewidth=0.5) 
        else:
            ax.axis('off')

        ax.set_title(f"{plot_title}\nSparsity: {sparsity*100:.4f}%")
        ax.set_xlabel('Matrix Column Index')
        ax.set_ylabel('Matrix Row Index')

        file_path = self._save_plot(fig, filename)

        return {
            "name": plot_title,
            "type": "sparsity_plot",
            "file": os.path.basename(file_path),
            "interpretation_key": interpretation_key
        }