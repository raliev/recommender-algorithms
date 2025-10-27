import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .BasePlotter import BasePlotter

class AttentionVisualizer(BasePlotter):
    """
    Plots a heatmap of attention weights from a Transformer-based model
    like SASRec.
    """

    def plot(self, attention_weights: np.ndarray,
             x_labels: list, y_labels: list,
             title: str, filename: str, interpretation_key: str):
        """
        Generates and saves the attention heatmap.

        Args:
            attention_weights (np.ndarray): 2D or 3D array of weights.
                                            If 3D (heads, seq, seq),
                                            plots the first head.
            x_labels (list): List of strings for the x-axis (Keys).
            y_labels (list): List of strings for the y-axis (Queries).
            title (str): The main title for the plot.
            filename (str): The filename to save the plot as.
            interpretation_key (str): The key for the manifest.

        Returns:
            dict: The manifest entry for this visualization.
        """

        weights_to_plot = attention_weights

        # If multi-head, just plot the first head for this example
        if attention_weights.ndim == 3:
            weights_to_plot = attention_weights[0]
            title = f"{title} (Head 1)"

        # Ensure we don't have more labels than data
        x_labels_trunc = x_labels[:weights_to_plot.shape[1]]
        y_labels_trunc = y_labels[:weights_to_plot.shape[0]]

        fig_width = max(10, len(x_labels_trunc) * 0.8)
        fig_height = max(8, len(y_labels_trunc) * 0.8)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        sns.heatmap(
            weights_to_plot,
            xticklabels=x_labels_trunc,
            yticklabels=y_labels_trunc,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            ax=ax
        )

        ax.set_title(title)
        ax.set_xlabel("Key (Attending To)")
        ax.set_ylabel("Query (Predicting From)")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        file_path = self._save_plot(fig, filename)

        return {
            "name": title,
            "type": "attention_heatmap",
            "file": os.path.basename(file_path),
            "interpretation_key": interpretation_key
        }