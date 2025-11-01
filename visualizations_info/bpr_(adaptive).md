---
### Avg. Negative Score
Shows the average predicted score (x_uj) of the 'hardest' negative item sampled during each epoch. This model uses **adaptive sampling**, meaning it actively looks for negative items (j) that the model *currently* predicts a high score for (making them "hard" examples).

* **What to look for:** A decreasing trend. This indicates the model is successfully learning to push down the scores of even these "hard" negatives. As the model improves, the average score of the hardest negatives it can find will get lower.

---
### Factor Change Norm Plot
This plot shows the **rate of change** of the User (P) and Item (Q) factor matrices at each iteration, measured by the Frobenius norm.
* **What it shows:** How much the factor matrices are being updated.
* **What to look for:**
    * **High values at the start:** The model is learning and making large adjustments.
    * **Decreasing values over time:** The model is stabilizing as it approaches a good solution.
    * **Values converging to (or near) zero:** The model has converged, and further iterations are providing diminishing returns.
* **Note:** This is a proxy for convergence. BPR uses Stochastic Gradient Descent (SGD), so the path is "noisy." We look for the overall downward trend, not a perfectly smooth line.

---
### Heatmaps (P and Q Snapshots)
These plots show the *actual values* inside the User-Factor (P) and Item-Factor (Q.T) matrices at the **first** and **last** iterations (and intervals in between, if any).
* **What it shows:** A "bird's-eye view" of the latent factors for a sample of users and items.
* **What to look for:**
    * **Iteration 1:** The matrix should look random, like static. This is the random initialization.
    * **Last Iteration:** The matrix should show clear patterns, structures, or "blocks." This indicates the model has learned relationships and grouped users/items by their latent features.
* **Comparison:** The difference between the first and last iteration heatmaps visually represents the learning process.

---
### Histograms (P and Q Snapshots)
These plots show the *distribution* of all values within the P and Q matrices at different iterations.
* **What it shows:** Whether the learned factors are diverse, clustered around zero, or have extreme outliers.
* **What to look for:**
    * **Iteration 1:** A tight, normal-like distribution around zero (from random initialization).
    * **Last Iteration:** The distribution will likely be wider, showing the model has learned a range of feature values to distinguish between users and items.
* **Impact of Regularization ($\lambda$):** Very strong regularization might keep this distribution tightly packed around zero, preventing the model from learning.

---
### 2D Latent Space Plot (if k=2)
This plot is only generated if you set **Latent Factors (k) = 2**. It plots every user and every item as a point on a 2D graph, using their two latent factors as (x, y) coordinates.
* **What it shows:** A direct visualization of the "recommendation space" the model has learned.
* **What to look for:**
    * **Proximity:** Users who are "close" to items in this space are more likely to be recommended those items.
    * **Clusters:** You may see "clusters" of users with similar tastes and the "clusters" of items they prefer. For example, a group of "Action Movie Lovers" might be plotted near a group of "Action Movies."
* **Comparison (First vs. Last Iteration):** The first iteration plot will show a random cloud of points. The last iteration plot should show meaningful structure.