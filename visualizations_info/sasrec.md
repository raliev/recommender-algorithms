---
### Objective
Shows the training loss (e.g., Cross-Entropy Loss) over epochs. SASRec predicts the *next item* in a user's sequence. A decreasing loss indicates the model is getting better at predicting subsequent items based on the preceding sequence context and converging.

---
### Snapshots
These plots visualize the distribution and relationships within the Item Embedding matrix (often denoted Q) at key epochs (usually the first and last). SASRec learns dense vector representations for items, incorporating positional information and self-attention over the user's sequence history. It does not typically learn distinct static user embeddings (P) in the same way as matrix factorization models.
* **Heatmaps:** Show the magnitude of values within the item embedding matrix (sampled).
* **Histograms:** Show the distribution of values within the item embedding matrix.
* **2D Latent Space (if k=2):** Plots items based on their first two embedding dimensions. Proximity might indicate items that are often consumed sequentially or are similar in the context learned by the self-attention mechanism.
  Comparing snapshots shows how item embeddings evolved during training.