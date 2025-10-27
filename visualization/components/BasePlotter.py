import os
import matplotlib.pyplot as plt
import seaborn as sns

class BasePlotter:
    """Base class for a reusable plotting component."""
    def __init__(self, visuals_dir):
        self.visuals_dir = visuals_dir
        os.makedirs(visuals_dir, exist_ok=True)

    def _save_plot(self, fig, filename):
        """Helper to save a plot to the correct directory."""
        path = os.path.join(self.visuals_dir, filename)
        fig.savefig(path)
        plt.close(fig)
        return path

    def add_to_manifest(self, manifest_list, name, type, file, interpretation_key):
        """Adds a simple file-based plot to the manifest."""
        manifest_list.append({
            "name": name,
            "type": type,
            "file": os.path.basename(file), # Store relative path
            "interpretation_key": interpretation_key
        })