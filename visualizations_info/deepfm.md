---
### Objective
Shows the training loss (Binary Cross-Entropy) over epochs.
DeepFM is trained to predict the probability of an interaction.
A decreasing trend indicates that the model is learning the patterns in the data and converging.
---
### Snapshots
These plots visualize the distribution and relationships within the shared embedding matrices (User Embeddings, P, and Item Embeddings, Q) at key epochs.
These embeddings are used by all three components (Linear, FM, and MLP).
* **Heatmaps:** Show the magnitude of values within the embedding matrices (sampled).
* **Histograms:** Show the distribution of values within the P and Q embedding matrices.
* **2D Latent Space (if k=2):** Plots users and items based on their first two embedding dimensions.
  Proximity might indicate similarity, though the final score is a complex combination from all three model components.
  Comparing snapshots shows how the embeddings evolved during training.