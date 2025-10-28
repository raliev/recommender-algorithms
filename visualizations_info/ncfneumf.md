---
### Objective
Shows the training loss (e.g., Binary Cross-Entropy) over epochs. NCF models (NeuMF, GMF, MLP) are typically trained using a classification or regression loss suitable for implicit or explicit feedback. A decreasing trend indicates that the model is learning the patterns in the data and converging.

---
### Snapshots
These plots visualize the distribution and relationships within the embedding matrices (User Embeddings, often denoted P, and Item Embeddings, Q) at key epochs (usually first and last). NCF learns dense vector representations for users and items.
* **Heatmaps:** Show the magnitude of values within the embedding matrices (sampled).
* **Histograms:** Show the distribution of values within the P and Q embedding matrices.
* **2D Latent Space (if k=2):** Plots users and items based on their first two embedding dimensions. Proximity might indicate similarity, depending on the NCF model type (GMF emphasizes dot product similarity, MLP learns complex interactions).
  Comparing snapshots shows how the embeddings evolved during training.

---
### Recommendation Breakdown
This visualization breaks down how NCF uses the learned embeddings and its neural network architecture to generate final prediction scores (often probabilities or logits) for a single sample user.
1.  **User History:** Shows the items the user has interacted with (implicit feedback) or rated (explicit feedback).
2.  **Aggregated Scores:** Shows the final output scores from the NCF model for all items for that user. Higher scores indicate a stronger recommendation.
3.  **Top-K Recommendations:** Highlights the highest-scoring items *that the user has not already interacted with*.