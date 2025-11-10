import os
from .NCFVisualizationRenderer import NCFVisualizationRenderer

class SimpleXVisualizationRenderer(NCFVisualizationRenderer):
    def __init__(self, run_dir, explanations):
        super().__init__(run_dir, explanations)
        self.algorithm_name = "SimpleX"
        self.run_timestamp = os.path.basename(run_dir)

        self.explanations.update({
            "Objective": "Shows the combined training loss (InfoNCE Contrastive Loss + Consistency Regularization) over epochs. A decreasing trend indicates the model is successfully learning to pull positive pairs together and push in-batch negative pairs apart.",
            "Snapshots": "These plots visualize the distribution and relationships within the User (P) and Item (Q) embedding matrices at key epochs.",
            "Recommendation Breakdown": "This visualizes how SimpleX uses the learned embeddings and a simple dot product (P_u @ Q.T) to generate final ranking scores for a sample user."
        })
