# ItemKNN Visualizations Explained

## Model Internals

---

### Raw Similarity Heatmap

**What it shows:** This heatmap visualizes the raw item-item similarity matrix (e.g., using Adjusted Cosine or Pearson) *before* any adjustments like `min_support` or `shrinkage` are applied.

**How to interpret:**

* This is the "pure" mathematical similarity between items based on user ratings.
* Bright squares (close to 1) indicate items that are very similarly rated. Dark squares (close to -1) indicate items that are oppositely rated.
* The diagonal is 1 (an item is perfectly similar to itself).
* This matrix can be noisy. A pair of items might get a high similarity score based on only one or two user ratings, which is not statistically reliable.

---

### Co-rated Counts Heatmap

**What it shows:** This heatmap visualizes the number of users who have rated *both* items in a pair. For example, the cell at (item A, item B) shows how many users provided a rating for *both* movie A and movie B.

**How to interpret:**

* This matrix represents the "support" or "confidence" we have in the raw similarity score.
* Bright squares indicate item pairs that have been co-rated by many users, making their similarity score more reliable.
* Dark areas (close to 0) indicate pairs that were co-rated by very few (or no) users. The raw similarity for these pairs is unreliable.
* **Hyperparameters:** This plot directly shows why `min_support` is important. `min_support` forces all similarities in the dark areas (low support) to be 0, cleaning up the noise.

---

### Final (Adjusted) Similarity Heatmap

**What it shows:** This is the **final, learned model** (`W`). It is the raw similarity matrix after being filtered by `min_support` (setting low-support scores to 0) and adjusted by `shrinkage` (reducing the confidence of scores that still have relatively low support).

**How to interpret:**

* Compare this to the "Raw Similarity Heatmap." You should see that it is much *sparser*.
* The `min_support` threshold has removed many of the noisy similarities.
* The `shrinkage` factor has likely dampened the values of the remaining, less-confident similarities.
* This final matrix is what is used to make predictions. Bright squares represent the strong, reliable neighbors that will be used in the `k`-nearest neighbors calculation.

---

### Histogram of Final Similarity Values

**What it shows:** This histogram plots the distribution of all the *non-zero* values from the "Final Similarity Heatmap."

**How to interpret:**

* This shows the overall range and frequency of the learned similarity "weights."
* Typically, you will see a large number of similarities clustered near zero, with a tail of higher-value similarities.
* If `k` (the number of neighbors used in prediction) is 20, but the histogram shows that most items only have ~5 high-similarity neighbors, you might consider if `k` is set appropriately.