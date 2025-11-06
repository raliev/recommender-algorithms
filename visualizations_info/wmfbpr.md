---
### Factor Change Norm Plot
This plot shows the **rate of change** of the User (P) and Item (Q) factor matrices at each iteration. A decreasing trend indicates that the factor matrices are stabilizing and the model is converging.
---
### AUC
Shows the **Area Under Curve (AUC)** on a validation set. AUC measures the model's ability to correctly rank a random positive item higher than a random negative item. A value of 1.0 is perfect, 0.5 is random. A rising curve indicates the model is learning to rank correctly.
---
### Snapshots
These plots visualize the distribution and relationships within the latent factor matrices (P for users, Q for items) at key iterations. Comparing the first and last snapshots shows how the embeddings evolved during training.
---
### Item Weights
This histogram shows the distribution of the global item importance weights (w_i) calculated using **PageRank** on the item co-occurrence graph. These weights are added to the item vectors *before* the dot product, effectively boosting the recommendation score for globally "important" items.
---
### Embedding t-SNE Plot
This plot shows a 2D t-SNE projection of the user (P) and item (Q) embedding vectors. It visualizes the 'interest map' the model has learned. Clusters of items/users indicate similarity in the latent space.
---
### Recommendation Breakdown
This visualization breaks down how WMFBPR generates final scores for a single sample user. The score is calculated using the weighted formula $\overline{r}_{ui} = P_u \cdot (Q_i + w_i)$, where $w_i$ is the item's global PageRank weight.