---
### Raw Similarity Heatmap
This heatmap visualizes the initial similarity scores calculated between items (ItemKNN) or users (UserKNN) *before* any adjustments like minimum support or shrinkage are applied. It helps understand the raw relatedness based purely on the chosen metric (e.g., cosine, adjusted cosine, pearson). A sample subset may be shown for large matrices.

---
### Final (Adjusted) Similarity Heatmap
This heatmap displays the item-item (ItemKNN) or user-user (UserKNN) similarity matrix *after* applying adjustments like minimum support (zeroing out similarities based on too few co-ratings/co-rated users) and shrinkage (reducing similarities with low support). This is the matrix used for generating predictions. Comparing this to the raw heatmap shows the impact of these adjustments. A sample subset may be shown.

---
### Co-rated Counts Heatmap
This heatmap shows the number of users who have rated both item *i* and item *j* (for ItemKNN) or the number of items rated by both user *u* and user *v* (for UserKNN). It provides context for the similarity scores, as similarities based on more co-ratings are generally more reliable. This matrix is used for applying minimum support and shrinkage. A sample subset may be shown.

---
### Histogram of Final Similarity Values
This histogram shows the distribution of the non-zero values in the *final (adjusted)* similarity matrix. It helps understand the range and frequency of similarity scores used for prediction. For ItemKNN, it often shows many small positive values, reflecting item relatedness.

---
### Recommendation Breakdown
This visualization breaks down how KNN generates recommendation scores for a single sample user based on the final similarity matrix (S).
* **ItemKNN:** Scores are calculated as a weighted average of the user's ratings for items similar to the target item: $\hat{r}_{ui} = \frac{\sum_{j \in N_k(i) \cap R_u} s_{ij} r_{uj}}{\sum_{j \in N_k(i) \cap R_u} |s_{ij}|}$. The plot approximates this using $R_u \cdot S$.
* **UserKNN:** Scores are calculated based on ratings given by similar users to the target item: $\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N_k(u)} s_{uv} (r_{vi} - \bar{r}_v)}{\sum_{v \in N_k(u)} |s_{uv}|}$. The plot visualization might use a simplified representation.
1.  **User History:** Shows the items the user has rated (ItemKNN) or the user's rating vector (UserKNN).
2.  **Aggregated Scores:** Shows the calculated prediction scores based on the KNN logic.
3.  **Top-K Recommendations:** Highlights the highest-scoring items *that the user has not already interacted with*.

---
### Deviation Matrix Heatmap
(Slope One Specific) This heatmap shows the average difference in ratings between pairs of items (item *i* vs item *j*), calculated across all users who rated both. Cell (i, j) contains $avg(rating_i - rating_j)$. This matrix is central to Slope One predictions. A sample subset may be shown.

---
### Deviation Histogram
(Slope One Specific) This histogram shows the distribution of the non-zero average deviation values found in the Deviation Matrix. It helps understand the typical rating differences learned by the model.