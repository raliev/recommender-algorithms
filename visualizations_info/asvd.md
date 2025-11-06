---
### Objective
Shows the Root Mean Squared Error (RMSE) calculated on the *observed ratings* in the training set over iterations. A decreasing trend indicates the model (biases and all three item-factor matrices Q, X, and Y) is learning to better predict known ratings and converging.
---
### Factor Change
Shows the Frobenius norm of the change in the item (Q), explicit item (X), and implicit item (Y) latent factor matrices between iterations. Bias changes are not typically shown here. A decreasing trend towards zero indicates that all three factor matrices are stabilizing, a sign of convergence.
---
### Snapshots
These plots visualize the distributions and relationships within the three latent factor matrices at key iterations.
* **P (Heatmap):** This shows the **Explicit Item Factors (X)**, which learn to represent an item's contribution to a user's profile based on the *rating* given.
* **Q (Heatmap):** This shows the primary **Item Factors (Q)**, which represent the item as something to *be rated*.
* **Y (Heatmap):** This shows the **Implicit Item Factors (Y)**, which represent an item's contribution to a user's profile simply by being *interacted with*.
* **Histograms/2D Plots:** Show the distributions and relationships of these factors.