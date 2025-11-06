import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from .BasePlotter import BasePlotter

class VectorHistogramPlotter(BasePlotter):
    """
    Plots a histogram for a 1D data vector.
    """

    def plot(self, data_vector: np.ndarray, title: str, filename: str,
             interpretation_key: str, bins: int = 50, x_label: str = 'Value'):
        """
        Generates and saves a histogram for the given 1D vector.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        if data_vector is None or data_vector.size == 0:
            ax.text(0.5, 0.5, 'No data to display.', ha='center', va='center')
        else:
            sns.histplot(data_vector, bins=bins, kde=True, ax=ax)

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Frequency')
        plt.tight_layout()

        file_path = self._save_plot(fig, filename)

        # Return manifest entry
        return {
            "name": title,
            "type": "histogram", # Use the generic 'histogram' type
            "file": os.path.basename(file_path),
            "interpretation_key": interpretation_key
        }