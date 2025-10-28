---
### Factor Change
Shows the Frobenius norm of the change in the item latent factor matrix P and the *other* item latent factor matrix Q between iterations. FISM learns two item-factor matrices to model item similarity based on user interaction patterns. A decreasing trend indicates that the factor matrices are stabilizing and the model is converging.

---
### Snapshots
These plots visualize the distribution and relationships within the two item latent factor matrices (P and Q) at key iterations (usually the first and last).
* **Heatmaps:** Show the magnitude of values within the factor matrices (sampled).
* **Histograms:** Show the distribution of values within P and Q matrices.
* **2D Latent Space (if k=2):** Plots items based on their first two factors from *both* P and Q matrices. Proximity might indicate similarity, although the relationship is complex (similarity is derived from $P \cdot Q^T$).
  Comparing snapshots shows how the item embeddings evolved during training.

---
### Recommendation Breakdown
This visualization breaks down how FISM generates recommendation scores for a single sample user. It approximates the FISM prediction formula $\hat{r}_{ui} = b_i + |\mathcal{R}_u \setminus \{i\}|^{-\alpha} \sum_{j \in \mathcal{R}_u \setminus \{i\}} \mathbf{p}_j \cdot \mathbf{q}_i^T$ by visualizing the contribution of the user's history ($R_u$) multiplied by the learned item-item similarity matrix ($S = P \cdot Q^T$).
1.  **User History:** Shows the items the user has interacted with (implicit feedback).
2.  **Aggregated Scores:** Shows scores calculated conceptually as $R_u \cdot (P \cdot Q^T)$. This reflects how items similar to the user's history contribute to the final score.
3.  **Top-K Recommendations:** Highlights the highest-scoring items *that the user has not already interacted with*.