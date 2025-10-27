import os
import numpy as np
import matplotlib.pyplot as plt
from .BasePlotter import BasePlotter

class RecommendationBreakdownPlotter(BasePlotter):
    """
    Plots a 3-panel visualization breaking down the recommendation
    process for a single user (e.g., R_u * W = R_tilde_u).
    This is adapted from the logic in SLIMVisualizer .
    """

    def plot(self, user_history_vector: np.ndarray,
             result_vector: np.ndarray,
             item_names: list,
             user_id: str,
             k: int,
             filename: str,
             interpretation_key: str,
             max_items_to_show: int = 50):
        """
        Generates and saves the breakdown plot.

        Args:
            user_history_vector (np.ndarray): 1D array (1xN) of the
                                              user's *original* interactions.
            result_vector (np.ndarray): 1D array (1xN) of the *final* predicted scores.
            item_names (list): List of item names/IDs for the x-axis.
            user_id (str): The ID of the user being plotted.
            k (int): The number of top-k recommendations to highlight.
            filename (str): The filename to save the plot as.
            interpretation_key (str): The key for the manifest.
            max_items_to_show (int, optional): Truncate x-axis for readability.

        Returns:
            dict: The manifest entry for this visualization.
        """
        try:
            R_u_dense = user_history_vector.ravel()
            R_tilde_u = result_vector.ravel()
            liked_item_indices = np.where(R_u_dense > 0)[0]

            fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

            # Determine range to plot
            max_idx = min(len(item_names), max_items_to_show)
            item_indices_range = range(max_idx)
            item_labels = item_names[:max_idx]

            # 1. Plot User History (R_u)
            ax1 = axes[0]
            R_u_plot = R_u_dense[:max_idx]
            ax1.stem(item_indices_range, R_u_plot,
                     linefmt='r-', markerfmt='ro', basefmt=' ')
            ax1.set_title(f'1. User {user_id}\'s History ($R_u$)')
            ax1.set_ylabel('Interaction (1=Liked)')

            # 2. Plot Aggregated Contributions
            ax2 = axes[1]
            R_tilde_plot_scores = R_tilde_u.copy()
            # Don't show scores for items already liked
            R_tilde_plot_scores[liked_item_indices] = 0
            ax2.bar(item_indices_range, R_tilde_plot_scores[:max_idx],
                    color='blue')
            ax2.set_title('2. Aggregated Recommendation Scores ($\tilde{R}_u$)')
            ax2.set_ylabel('Aggregated Score')

            # 3. Plot Top-K Recommendations
            ax3 = axes[2]
            # Get Top-K from the *filtered* scores
            top_k_indices = np.argsort(R_tilde_plot_scores)[-k:]
            top_k_scores = R_tilde_plot_scores[top_k_indices]

            rec_plot_final = np.zeros_like(R_tilde_u)
            rec_plot_final[top_k_indices] = top_k_scores

            ax3.bar(item_indices_range, rec_plot_final[:max_idx],
                    color='green')
            ax3.set_title(f'3. Final Top-{k} Recommendations '
                          '(Items already liked are excluded)')
            ax3.set_ylabel('Final Score')
            ax3.set_xlabel('Item Index / Name')

            if len(item_names) > max_items_to_show:
                ax3.set_xlim(-1, max_items_to_show)

            # Set text labels if not too crowded
            if max_idx <= 40:
                ax3.set_xticks(item_indices_range)
                ax3.set_xticklabels(item_labels, rotation=90)

            plt.tight_layout()
            file_path = self._save_plot(fig, filename)

            return {
                "name": f"Recommendation Breakdown for User {user_id}",
                "type": "recommendation_breakdown",
                "file": os.path.basename(file_path),
                "interpretation_key": interpretation_key
            }

        except Exception as e:
            print(f"Error plotting recommendation breakdown: {e}")
            # Create an empty plot with error text
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error plotting breakdown: {e}",
                    ha='center', va='center')
            file_path = self._save_plot(fig, filename)
            return {
                "name": "Breakdown Plot (Error)",
                "type": "recommendation_breakdown",
                "file": os.path.basename(file_path),
                "interpretation_key": "Error"
            }