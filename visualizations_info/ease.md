---
### Final Similarity Heatmap
This heatmap visualizes the **learned item-item similarity matrix (B)**, which is the core of the EASE model.
* **What it shows:** The coefficients of the linear autoencoder. The value at `(i, j)` represents how much item `j` (column) contributes to the reconstruction of item `i` (row).
* **What to look for:** Unlike KNN similarity, these values are not bounded between [-1, 1] and can be negative.
    * **Positive Value (Bright):** Indicates that item `j`'s presence positively predicts the presence of item `i`.
    * **Negative Value (Dark):** Indicates that item `j`'s presence negatively predicts (or "suppresses") item `i`.
* **Note:** EASE is a **dense** model, so this matrix is not sparse like SLIM's.

---
### Histogram of Final Similarity Values
This histogram shows the distribution of all values from the learned similarity matrix `B`.
* **What it shows:** The magnitude and range of the learned item-item relationships.
* **What to look for:** You will typically see a distribution centered around zero, but with significant positive and negative tails, representing the positive and negative correlations the model has learned.

---
### Recommendation Breakdown
This plot explains **how a recommendation is made** using the final prediction formula $\hat{X} = XB$.
1.  **Top (Red)**: The user's original interaction history ($R_u$).
2.  **Middle (Blue)**: The *aggregated scores* ($\tilde{R}_u = R_u \cdot B$). For each item $i$ the user liked, the model "votes" for other items $j$ with the strength $B_{ij}$. This plot is the sum of all 'votes' from the user's history.
3.  **Bottom (Green)**: The final Top-K recommendations, after filtering out items the user already liked.