import os
from .PureSVDVisualizer import PureSVDVisualizer

class SVDVisualizer(PureSVDVisualizer):
    """
    Specific visualizer for SVD.
    Its logic is identical to PureSVDVisualizer, just with a different name.
    """
    def __init__(self, k_factors, **kwargs):
        super().__init__(k_factors, **kwargs)

        self.algorithm_name = "SVD"
        self.visuals_dir = os.path.join(self.visuals_base_dir, self.algorithm_name, self.run_timestamp)
        os.makedirs(self.visuals_dir, exist_ok=True)