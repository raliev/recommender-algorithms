# WRMF Visualizations Explained

## Convergence Plots
---
### Objective
Shows the objective function value optimized by WRMF over iterations. This objective typically involves minimizing the squared error between predicted and actual interactions (where actual is 1 for interactions, 0 otherwise), weighted by a confidence score (C = 1 + alpha * R), plus regularization terms for P and Q. A decreasing trend indicates the model is converging towards a good implicit feedback solution.

---

### Objective Function Plot (Approximate)

**What it shows:** This plot tracks the value of the WRMF objective function (loss) over iterations. WRMF tries to minimize this function, which balances reconstructing the observed interactions (weighted by confidence) and keeping the latent factors small (regularization).

**How to interpret:**

* **Good:** The curve should generally decrease over iterations and eventually flatten out. A flattening curve indicates **convergence** – the model isn't changing much anymore.
* **Bad:** If the curve fluctuates wildly, stays flat from the beginning, or increases, it might indicate problems. Maybe the regularization (`lambda_reg`) is too high or too low, or there's an issue with the implementation.

**Hyperparameters:**

* `lambda_reg`: Higher regularization might lead to a higher final objective value but can prevent overfitting. Lower regularization might allow the objective to get lower but risks overfitting.
* `alpha`: Affects the confidence weights (`C` matrix). Higher alpha gives more importance to observed interactions. This indirectly influences the objective function's landscape.

---

### Factor Change Norm Plot

**What it shows:** This plots the magnitude (Frobenius norm) of the change in the User (P) and Item (Q) latent factor matrices between consecutive iterations.

**How to interpret:**

* 📉 **Good:** The change should decrease rapidly in early iterations and approach zero as the algorithm converges. This means the latent factors are stabilizing.
* 📈 **Bad:** If the change remains high or fluctuates significantly even after many iterations, the model hasn't converged. It might need more `iterations`, or the `lambda_reg` might be too low, allowing factors to keep changing drastically.

**Hyperparameters:**

* `iterations`: If the change norm is still high when training stops, you might need more iterations.
* `lambda_reg`: Higher regularization tends to make factors stabilize faster (smaller changes).

---

## Snapshot Comparison Plots
---

### Heatmaps (P and Q Snapshots)

**What they show:** These visualize the actual values within the User (P) and Item (Q) latent factor matrices at specific iterations (e.g., first and last). Each row/column corresponds to a user/item, and the columns/rows represent the latent factors. The colors indicate the magnitude of the factor values.

**How to interpret:**

* **Early Iterations:** Expect to see noisy, almost random patterns, reflecting the initial random values.
* **Later Iterations:** Look for emerging structures or patterns. Do certain users/items have distinctly high or low values for specific factors? Are there blocks of similar colors? This shows the model learning representations. Comparing the first and last iteration heatmaps clearly shows how much the factors have evolved.
* **Performance Insight:** It's harder to judge *absolute* performance directly from heatmaps alone, but drastic differences between users or items might indicate learned preferences or characteristics. Very uniform heatmaps might suggest the model isn't learning diverse features.

**Hyperparameters:** `k` (number of factors) directly impacts the width/height of these matrices. `lambda_reg` influences the range of values (higher regularization usually leads to smaller values).

---

### Histograms (P and Q Snapshots)

**What they show:** These show the distribution of values within the P and Q matrices at specific iterations.

**How to interpret:**

* **Early Iterations:** Often centered around zero with a certain spread, based on the random initialization.
* **Later Iterations:** The distribution might shift, spread out, or become more peaked. Look for changes between the first and last iteration. Are the values becoming extremely large or small? A relatively stable, perhaps multi-modal distribution often indicates learning. If all values collapse to near zero, `lambda_reg` might be too high. If they explode, `lambda_reg` might be too low.

**Hyperparameters:** `lambda_reg` strongly influences the range and variance of these values.

---

### 2D Latent Space Plot (if k=2)

**What it shows:** If you use only 2 latent factors (`k=2`), this scatter plot shows each user and item as a point in that 2D space.

**How to interpret:**

* Look for structure. Do users and items that seem related cluster together? Are there distinct groups?
* The distance between a user point and an item point *might* relate to preference (though WRMF uses dot products, not just distance). Users close together might have similar tastes. Items close together might be similar.
* Comparing the first and last iteration shows how the model organizes users and items in the latent space during training. Initially random points should move into more meaningful positions.

### Snapshots
These plots visualize the distribution and relationships within the latent factor matrices (P for users, Q for items) learned by WRMF at key iterations (usually first and last).
* **Heatmaps:** Show the magnitude of values within the factor matrices (sampled).
* **Histograms:** Show the distribution of values within P and Q matrices.
* **2D Latent Space (if k=2):** Plots users and items based on their first two latent factors. Proximity suggests similarity in the context of implicit feedback preferences.
  Comparing snapshots shows how embeddings evolved during WRMF training.

---
### Recommendation Breakdown
This visualization breaks down how WRMF uses the learned latent factors (P and Q) to generate final prediction scores for a single sample user using the dot product $P_u \cdot Q^T$.
1.  **User History:** Shows the items the user has interacted with (implicit feedback).
2.  **Aggregated Scores:** Shows the raw prediction scores calculated as $P_u \cdot Q^T$. Higher scores indicate items the model predicts the user is more likely to interact with.
3.  **Top-K Recommendations:** Highlights the items predicted to have the highest rating *that the user has not already rated*.