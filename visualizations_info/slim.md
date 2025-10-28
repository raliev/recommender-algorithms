### Final (Adjusted) Similarity Heatmap
This plot shows a sampled heatmap of the **learned item-item similarity matrix ($W$)**. This matrix is the "brain" of the SLIM model.

* **What it shows:** The coefficients learned by the ElasticNet regression. The value at $(i, j)$ represents how much item $j$ (column) is used to predict a user's rating for item $i$ (row).
* **What to look for:**
    * **Sparsity:** Because SLIM uses L1 regularization, this matrix should be *very sparse* (mostly zeros). The dark-blue/black cells represent a coefficient of 0.
    * **Bright Spots:** The non-zero (brighter) cells indicate the item-item relationships the model learned. A bright spot at $(i, j)$ means "item $j$ is a good predictor for item $i$."
    * **Positive-Only:** The `positive=True` setting in the model  ensures all coefficients are $\geq 0$, so we only see positive (or zero) relationships.

---
### Histogram of Final Similarity Values
This plot shows the *distribution* of all **non-zero** coefficients from the similarity matrix $W$.

* **What it shows:** The magnitude of the learned item-item relationships.
* **What to look for:**
    * This plot *only* shows non-zero values. The vast number of zero-value coefficients are excluded.
    * **Distribution Shape:** You can see if the learned relationships are all small and weak (clustered near zero) or if the model found some very strong, high-value relationships (a tail to the right).
    * **Impact of L1 ($\lambda_1$):** A higher `l1_reg` parameter will result in *fewer* non-zero coefficients (a sparser matrix), and this histogram will be built from a smaller set of values.

---
### W Sparsity
This is the **core idea** of SLIM. Each dot represents a non-zero coefficient ($W_{ij} > 0$) in the item-item matrix $W$. A sparse plot (mostly white) shows that the model has correctly learned relationships for only a small, relevant subset of item pairs, making it efficient and often more robust. The 'Sparsity' percentage shows how many of the *possible* connections are zero.

---
### W Distribution
This histogram shows the values of all the non-zero coefficients in $W$. It helps understand the *strength* of the learned item-to-item relationships. A distribution skewed towards zero means most relationships are weak, while a long tail indicates a few items are very strong predictors for others.

---
### SLIM Recommendation
This plot explains **how a recommendation is made**. <br>1. **Top (Red)**: The user's original interaction history ($R_u$). <br>2. **Middle (Blue)**: The *aggregated scores* ($\tilde{R}_u = R_u \cdot W$). For each item $i$ the user liked, the model 'votes' for related items $j$ with the strength $W_{ij}$. This plot is the sum of all 'votes' from the user's history. <br>3. **Bottom (Green)**: The final Top-K recommendations, after filtering out items the user already liked.
