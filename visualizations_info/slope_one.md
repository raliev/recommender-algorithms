### Final (Adjusted) Similarity Heatmap
**(Average Deviation Matrix)**

This plot shows a sampled heatmap of the **Average Deviation Matrix (`dev_matrix`)**. This is the core "brain" of the SlopeOne model.

* **What it shows:** The value at `(i, j)` is the average difference `(rating(i) - rating(j))` calculated from all users who rated both items.
* **What to look for:**
    * **Positive Value (e.g., +1.5 at (i, j)):** Item `i` is, on average, rated 1.5 stars *higher* than item `j`.
    * **Negative Value (e.g., -0.8 at (i, j)):** Item `i` is, on average, rated 0.8 stars *lower* than item `j`.
    * **Zero Value:** Items `i` and `j` are rated similarly on average, or no users co-rated them.
    * **Anti-symmetry:** The matrix is anti-symmetric (i.e., `dev_matrix[i, j] = -dev_matrix[j, i]`).

---
### Co-rated Counts Heatmap
**(Frequency Matrix)**

This plot shows a sampled heatmap of the **Co-rated Frequency Matrix (`freq_matrix`)**. This matrix provides the "support" or "confidence" for the values in the Deviation Matrix.

* **What it shows:** The value at `(i, j)` is the *count* of users who provided a rating for *both* item `i` and item `j`.
* **What to look for:**
    * **Bright Spots:** Indicate item pairs that are frequently rated together, making their average deviation value more reliable.
    * **Dark Spots (Zeros):** Indicate item pairs that were *never* rated by the same user. The deviation value for these pairs will be 0, and they cannot be used to make predictions for each other.
    * **Symmetry:** This matrix is symmetric (`freq_matrix[i, j] = freq_matrix[j, i]`).

---
### Histogram of Final Similarity Values
**(Distribution of Average Deviations)**

This plot shows the *distribution* of all **non-zero** values from the Average Deviation Matrix.

* **What it shows:** The range and commonality of rating deviations across the entire dataset.
* **What to look for:**
    * **Central Peak:** Often centered around 0, indicating that many item pairs have small average deviations.
    * **Spread:** A wider spread indicates larger average differences between item ratings (e.g., a "great" item vs. a "terrible" item).
    * **Symmetry:** The distribution should be roughly symmetric around 0 due to the anti-symmetric nature of the deviation matrix (for every `+1.5`, there is a `-1.5`).