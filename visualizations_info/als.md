---
### Objective
Shows the Root Mean Squared Error (RMSE) calculated on the *observed ratings* in the training set over iterations. A decreasing trend indicates the model is learning to better predict the known ratings and converging. Lower RMSE generally suggests better predictive accuracy on the training data.

---
### Factor Change
Shows the Frobenius norm (a measure of matrix magnitude) of the change in the user latent factor matrix (P) and the item latent factor matrix (Q) between consecutive iterations. A decreasing trend towards zero indicates that the factor matrices are stabilizing, which is another sign of convergence. Large initial changes are expected, followed by smaller adjustments.

---
### Snapshots
These plots visualize the distribution and relationships within the latent factor matrices (P for users, Q for items) at key iterations (usually the first and last).
* **Heatmaps:** Show the magnitude of values within the factor matrices (sampled if large). Patterns might reveal user/item groups or dominant factors.
* **Histograms:** Show the distribution of values within the P and Q matrices. Ideally, values should be reasonably distributed, not collapsed to zero or extremely large.
* **2D Latent Space (if k=2):** Plots users and items based on their first two latent factors. Proximity in this space suggests similarity as learned by the model.
  Comparing the first and last snapshots shows how the embeddings evolved during training.