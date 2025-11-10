import os

from .AlgorithmVisualizer import AlgorithmVisualizer
from .BPRAdaptiveVisualizer import BPRAdaptiveVisualizer
from .BPRVisualizer import BPRVisualizer
from ..components.ConvergencePlotter import ConvergencePlotter
from ..components.EmbeddingTSNEPlotter import EmbeddingTSNEPlotter
from ..components.FactorMatrixPlotter import FactorMatrixPlotter
from ..components.RecommendationBreakdownPlotter import RecommendationBreakdownPlotter


class BPRSVDPPVisualizer(BPRAdaptiveVisualizer):
    """Specific visualizer for BPR."""
    def __init__(self, k_factors, plot_interval=5):
        AlgorithmVisualizer.__init__(self, "BPR+SVDPP", plot_interval)
        self.k_factors = k_factors
        self.history['p_change'] = []
        self.history['q_change'] = []
        self.history['auc'] = [] # 2. ADD AUC TO HISTORY

        self.convergence_plotter = ConvergencePlotter(self.visuals_dir)
        self.matrix_plotter = FactorMatrixPlotter(self.visuals_dir,
                                                  self.k_factors)
        self.breakdown_plotter = RecommendationBreakdownPlotter(self.visuals_dir)
        self.tsne_plotter = EmbeddingTSNEPlotter(self.visuals_dir) # 3. INSTANTIATE TSNE PLOTTER

        self.R = None
        self.P = None
        self.Q = None
        self.algorithm_name = "BPR+SVDPP"
        self.visuals_dir = os.path.join(self.visuals_base_dir, self.algorithm_name, self.run_timestamp)
        os.makedirs(self.visuals_dir, exist_ok=True)

        self.convergence_plotter = ConvergencePlotter(self.visuals_dir)
        self.matrix_plotter = FactorMatrixPlotter(self.visuals_dir, self.k_factors) # <-- FIX
        self.breakdown_plotter = RecommendationBreakdownPlotter(self.visuals_dir) # <-- FIX
        self.tsne_plotter = EmbeddingTSNEPlotter(self.visuals_dir)

        self.history['avg_negative_score'] = []
        self.history['auc'] = []