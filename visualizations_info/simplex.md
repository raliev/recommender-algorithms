---
### Objective
Shows the combined training loss over epochs. This loss is primarily the **InfoNCE Contrastive Loss** with an added **Consistency Regularization** term.

* **InfoNCE Loss:** This loss trains the model to maximize the similarity (dot product) between a user and their positive items, while simultaneously minimizing the similarity between that user and all other "in-batch negative" items.
* **What to look for:** A decreasing trend indicates the model is successfully learning to "contrast" positive and negative items, pulling user embeddings closer to items they've interacted with and pushing them away from items they haven't.

---
### Snapshots
These plots visualize the distribution and relationships within the learned User (P) and Item (Q) embedding matrices at key epochs. SimpleX is a bi-encoder model that learns these embeddings directly.

* **Heatmaps:** Show the magnitude of values within the embedding matrices (sampled).
* **Histograms:** Show the distribution of values within the P and Q embedding matrices.
* **2D Latent Space (if k=2):** Plots users and items based on their first two embedding dimensions. Since the recommendation score is a simple dot product ($s(u,i) = p_u^\top q_i$), proximity in this space (in a dot-product, not Euclidean, sense) indicates a high recommendation score.

---
### Recommendation Breakdown
This visualization breaks down how SimpleX uses the learned embeddings to generate final ranking scores for a single sample user.

1.  **User History:** Shows the items the user has interacted with (implicit feedback).
2.  **Aggregated Scores:** Shows the final output scores, calculated as a simple dot product of the user's embedding vector and all item embedding vectors ($P_u \cdot Q^T$).
3.  **Top-K Recommendations:** Highlights the highest-scoring items *that the user has not already interacted with*.