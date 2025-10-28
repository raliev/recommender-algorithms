---
### Objective
Shows the Root Mean Squared Error (RMSE) calculated on the *observed ratings* in the training set over iterations, **including the bias terms** (global mean, user biases, item biases). A decreasing trend indicates the model (factors and biases) is learning to better predict the known ratings and converging.

---
### Factor Change
Shows the Frobenius norm of the change in the user latent factor matrix (P) and the item latent factor matrix (Q) between iterations. **Bias changes are not shown here.** A decreasing trend indicates that the factor matrices are stabilizing, a sign of convergence.

---
### Snapshots
These plots visualize the distribution and relationships within the latent factor matrices (P for users, Q for items) at key iterations. Bias vectors are not visualized here.
* **Heatmaps:** Show factor magnitudes (sampled).
* **Histograms:** Show value distributions within P and Q.
* **2D Latent Space (if k=2):** Plots users/items based on their first two factors.
  Comparing snapshots shows embedding evolution.