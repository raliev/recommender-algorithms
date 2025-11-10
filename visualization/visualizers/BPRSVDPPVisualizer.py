import os

from .BPRAdaptiveVisualizer import BPRAdaptiveVisualizer

class BPRSVDPPVisualizer(BPRAdaptiveVisualizer):
    """
    Specific visualizer for BPR+SVDPP.
    Inherits all plotting logic from BPRAdaptiveVisualizer.
    """
    def __init__(self, k_factors, plot_interval=5):
        super().__init__(k_factors, plot_interval)

        self.algorithm_name = "BPR+SVDPP"
        self.visuals_dir = os.path.join(self.visuals_base_dir, self.algorithm_name, self.run_timestamp)
        os.makedirs(self.visuals_dir, exist_ok=True)

        # All other history keys (p_change, q_change, auc, avg_negative_score)
        # and plotters (convergence_plotter, matrix_plotter, tsne_plotter, etc.)
        # are inherited from the parent classes (BPRAdaptiveVisualizer, BPRVisualizer,
        # and AlgorithmVisualizer)