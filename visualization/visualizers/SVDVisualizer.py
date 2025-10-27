# File: visualization/visualizers/SVDVisualizer.py
import os
from .PureSVDVisualizer import PureSVDVisualizer
from visualization.components.FactorMatrixPlotter import FactorMatrixPlotter
from visualization.components.SingularValuesPlotter import SingularValuesPlotter

class SVDVisualizer(PureSVDVisualizer): # Inherit from PureSVDVisualizer
    """
    Specific visualizer for SVD.
    Its logic is identical to PureSVDVisualizer, just with a different name.
    """
    def __init__(self, k_factors, **kwargs):
        # Call parent __init__ but it will set the name to PureSVD
        super().__init__(k_factors, **kwargs)

        # --- Override the algorithm name and directory ---
        self.algorithm_name = "SVD"
        self.visuals_dir = os.path.join(self.visuals_base_dir, self.algorithm_name, self.run_timestamp)
        os.makedirs(self.visuals_dir, exist_ok=True) # Ensure it exists

        # --- Re-instantiate plotters with the correct directory ---
        self.matrix_plotter = FactorMatrixPlotter(self.visuals_dir, self.k_factors)
        self.scree_plotter = SingularValuesPlotter(self.visuals_dir)

    # The visualize_fit_results method is inherited and will work correctly.