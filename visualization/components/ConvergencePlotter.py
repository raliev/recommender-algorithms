import os

from .BasePlotter import BasePlotter
import matplotlib.pyplot as plt

class ConvergencePlotter(BasePlotter):
    """Plots one or more convergence lines."""
    def plot(self, data_dict, title, y_label, filename, interpretation_key):
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, values in data_dict.items():
            if values:
                ax.plot(range(1, len(values) + 1),
                        values,
                        label=key)

        ax.set_title(title)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(y_label)
        if len(data_dict) > 1: ax.legend()
        ax.grid(True)
        plt.tight_layout()

        file_path = self._save_plot(fig, filename)

        # Return manifest entry
        return {
            "name": title,
            "type": "line_plot",
            "file": os.path.basename(file_path),
            "interpretation_key": interpretation_key
        }