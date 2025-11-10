import os
from .NCFVisualizationRenderer import NCFVisualizationRenderer

class DeepFMRenderer(NCFVisualizationRenderer):
    """
    Renders visualizations specific to DeepFM.
    It inherits from NCFVisualizationRenderer because the plot types
    (Objective, Snapshots, Breakdown) are the same.
    """
    def __init__(self, run_dir, explanations):
        """
        Initialize the renderer.
        """
        super().__init__(run_dir, explanations) # Pass explanations to base
        self.algorithm_name = "DeepFM" # Set correct algorithm name
        self.run_timestamp = os.path.basename(run_dir) # Get timestamp from dir

        self.explanations.update({
            "Objective": "Shows the training loss (Binary Cross-Entropy) over epochs. DeepFM combines Linear, FM, and MLP components to predict interaction probability. A decreasing trend indicates convergence.",
            "Snapshots": "These plots visualize the distribution and relationships within the shared User (P) and Item (Q) embedding matrices at key epochs.",
            "Recommendation Breakdown": "This visualizes how DeepFM uses the learned embeddings and its combined (Linear + FM + MLP) architecture to generate final scores for a sample user."
        })
