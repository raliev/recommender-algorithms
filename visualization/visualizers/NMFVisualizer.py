import os
from .FunkSVDVisualizer import FunkSVDVisualizer

class NMFVisualizer(FunkSVDVisualizer):
    """
    Specific visualizer for NMF.
    Inherits from FunkSVDVisualizer and sets the correct algorithm name.
    """
    def __init__(self, k_factors, plot_interval=5):
        super().__init__(k_factors, plot_interval)

        self.algorithm_name = "NMF"
        self.visuals_dir = os.path.join(self.visuals_base_dir, self.algorithm_name, self.run_timestamp)
        os.makedirs(self.visuals_dir, exist_ok=True)