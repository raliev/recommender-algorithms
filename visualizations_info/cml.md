---
### Factor Change
Shows the Frobenius norm of the change in the user (P) and item (Q) latent factor matrices between iterations. CML optimizes item distances in latent space based on user interactions. A decreasing trend in factor change indicates that the latent factors are stabilizing and the model is converging.

---
### Snapshots
These plots visualize the distribution and relationships within the latent factor matrices (P for users, Q for items) at key iterations (usually the first and last).
* **Heatmaps:** Show the magnitude of values within the factor matrices (sampled).
* **Histograms:** Show the distribution of values within P and Q matrices.
* **2D Latent Space (if k=2):** Plots users and items based on their first two latent factors. CML aims to place users closer to items they've interacted with than to items they haven't.
  Comparing snapshots shows how the embeddings were adjusted to satisfy the distance constraints.

---
### Recommendation Breakdown
This visualization breaks down how CML generates ranking scores for a single sample user, using the learned latent factors (P and Q). Since CML is based on distances (lower is better), scores are often represented as the *negative* Euclidean distance: $-\| P_u - Q_i \|$.
1.  **User History:** Shows the items the user has interacted with (implicit feedback).
2.  **Aggregated Scores:** Shows the negative distance scores. Items closer to the user in the latent space will have higher (less negative) scores.
3.  **Top-K Recommendations:** Highlights the highest-scoring (closest) items *that the user has not already interacted with*.