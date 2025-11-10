import os
from .NCFVisualizationRenderer import NCFVisualizationRenderer

class DeepFMRenderer(NCFVisualizationRenderer):
    def __init__(self, run_dir, explanations):
        super().__init__(run_dir, explanations)
        self.algorithm_name = "DeepFM"
        self.run_timestamp = os.path.basename(run_dir)

        self.explanations.update({
            "Objective": "Shows the training loss (Binary Cross-Entropy) over epochs. DeepFM combines Linear, FM, and MLP components to predict interaction probability. A decreasing trend indicates convergence.",
            "Snapshots": "These plots visualize the distribution and relationships within the shared User (P) and Item (Q) embedding matrices at key epochs.",
            "Recommendation Breakdown": "This visualizes how DeepFM uses the learned embeddings and its combined (Linear + FM + MLP) architecture to generate final scores for a sample user."
        })