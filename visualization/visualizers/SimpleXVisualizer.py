import os
from .NCFVisualizer import NCFVisualizer

class SimpleXVisualizer(NCFVisualizer):
    """
    Specific visualizer for SimpleX.
    Inherits from NCFVisualizer as the plots (loss, snapshots, breakdown)
    are identical in structure.
    """
    def __init__(self, k_factors=0, plot_interval=1):
        # Call the parent __init__
        super().__init__(k_factors, plot_interval)

        self.algorithm_name = "SimpleX"
        self.visuals_dir = os.path.join(self.visuals_base_dir, self.algorithm_name, self.run_timestamp)
        os.makedirs(self.visuals_dir, exist_ok=True)

        self.convergence_plotter = self.convergence_plotter.__class__(self.visuals_dir)
        if self.matrix_plotter:
            self.matrix_plotter = self.matrix_plotter.__class__(self.visuals_dir, self.k_factors)
        self.breakdown_plotter = self.breakdown_plotter.__class__(self.visuals_dir)
