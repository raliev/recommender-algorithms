---
### Objective
Shows the Root Mean Squared Error (RMSE) calculated on the *observed ratings* in the training set over iterations. A decreasing trend indicates the model is learning to better predict the known ratings and converging. Lower RMSE generally suggests better predictive accuracy on the training data.

---
### Factor Change Norm Plot
This plot shows the **rate of change** of the User (P) and Item (Q) factor matrices at each iteration, measured by the Frobenius norm.

* **What it shows:** How much the factor matrices are being updated after one full pass (iteration) over all known ratings.
* **What to look for:**
    * **High values at the start:** The model is learning and making large adjustments from its random start.
    * **Decreasing values over time:** The model is stabilizing as it approaches a good solution (a local minimum for the error).
    * **Values converging to (or near) zero:** The model has converged. Further iterations are providing diminishing returns, and the factor matrices are no longer changing significantly.
* **Note:** This is a proxy for convergence. `FunkSVD` uses Stochastic Gradient Descent (SGD), which updates one rating at a time, so the path is "noisy." We look for the overall downward trend of the *entire matrix* after each full iteration.

---
### Heatmaps (P and Q Snapshots)
These plots show the *actual values* inside the User-Factor (P) and Item-Factor (Q.T) matrices at the **first** and **last** iterations.

* **What it shows:** A "bird's-eye view" of the latent factors for a sample of users and items.
* **What to look for:**
    * **Iteration 1:** The matrix should look random, like static. This is the random initialization.
    * **Last Iteration:** The matrix should show clear patterns and structures. This indicates the model has learned relationships and grouped users/items by their latent features (e.g., a "comedy" factor, an "action" factor, etc.).
    * **Comparison:** The difference between the first and last iteration heatmaps visually represents the entire learning process.

---
### Histograms (P and Q Snapshots)
These plots show the *distribution* of all values within the P and Q matrices at different iterations.

* **What it shows:** Whether the learned factors are diverse, clustered around zero, or have extreme outliers.
* **What to look for:**
    * **Iteration 1:** A tight, normal-like distribution around zero (from random initialization).
    * **Last Iteration:** The distribution will likely be wider, showing the model has learned a range of feature values to distinguish between users and items.
    * **Impact of Regularization ($\lambda$):** Very strong regularization (`lambda_reg`)  will keep this distribution tightly packed around zero, preventing the model from learning complex patterns (i.e., preventing overfitting).

---
### 2D Latent Space Plot (if k=2)
This plot is only generated if you set **Latent Factors (k) = 2**. It plots every user and every item as a point on a 2D graph, using their two latent factors as (x, y) coordinates.

* **What it shows:** A direct visualization of the "recommendation space" the model has learned.
* **What to look for:**
    * **Proximity:** In an explicit model like FunkSVD, the dot product of a user vector and an item vector predicts the rating. Users will be "near" items they rate highly (in a dot-product sense, not just Euclidean distance).
    * **Clusters:** You will see clusters of users with similar tastes and the clusters of items they prefer. For example, a group of "Sci-Fi Lovers" might be plotted in a similar region, and "Sci-Fi Movies" will be in a corresponding region that results in a high dot product.
    * **Comparison (First vs. Last Iteration):** The first iteration plot will show a random cloud of points. The last iteration plot should show this meaningful structure.

---
### Snapshots
These plots visualize the distribution and relationships within the latent factor matrices (P for users, Q for items) at key iterations (usually the first and last).
* **Heatmaps:** Show the magnitude of values within the factor matrices (sampled).
* **Histograms:** Show the distribution of values within P and Q matrices.
* **2D Latent Space (if k=2):** Plots users and items based on their first two latent factors. Proximity suggests similarity.
  Comparing snapshots shows how the embeddings evolved.

---
### Recommendation Breakdown
This visualization breaks down how FunkSVD uses the learned latent factors (P and Q) to generate final predicted rating scores for a single sample user using the dot product $P_u \cdot Q^T$.
1.  **User History:** Shows the items the user has rated and the ratings given.
2.  **Aggregated Scores:** Shows the predicted rating scores calculated as $P_u \cdot Q^T$. These scores directly represent the model's prediction for each item's rating by that user.
3.  **Top-K Recommendations:** Highlights the items predicted to have the highest rating *that the user has not already rated*.