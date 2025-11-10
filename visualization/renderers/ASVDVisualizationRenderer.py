import streamlit as st
import os
from .SVDppVisualizationRenderer import SVDppVisualizationRenderer
class ASVDVisualizationRenderer(SVDppVisualizationRenderer):
    def __init__(self, run_dir, explanations):
        super().__init__(run_dir, explanations)
        self.algorithm_name = "ASVD"

        self.explanations.update({
            "Objective": "Shows the Root Mean Squared Error (RMSE) calculated on the *observed ratings* in the training set over iterations. A decreasing trend indicates convergence.",
            "Factor Change": "Shows the Frobenius norm of the change in the item (Q), explicit item (X), and implicit item (Y) latent factor matrices between iterations. A decreasing trend indicates convergence.",
            "Snapshots": "These plots visualize the distributions and relationships within the latent factor matrices at key iterations.\n\n* **P (Heatmap):** This shows the **Explicit Item Factors (X)**.\n* **Q (Heatmap):** This shows the primary **Item Factors (Q)**.\n* **Y (Heatmap):** This shows the **Implicit Item Factors (Y)**."
        })

