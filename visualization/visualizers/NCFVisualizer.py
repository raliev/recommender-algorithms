import os
import json
import numpy as np

from .AlgorithmVisualizer import AlgorithmVisualizer
from visualization.components.ConvergencePlotter import ConvergencePlotter
from visualization.components.FactorMatrixPlotter import FactorMatrixPlotter
from visualization.components.RecommendationBreakdownPlotter import RecommendationBreakdownPlotter

class NCFVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for NCF / NeuMF.
    Plots loss convergence and embedding matrix snapshots.
    """
    def __init__(self, k_factors=0, plot_interval=1): # k_factors set by algorithm #
        super().__init__("NCFNeuMF", plot_interval) #
        self.k_factors = k_factors #
        # NCF just tracks objective (loss)
        self.history['objective'] = [] #

        self.convergence_plotter = ConvergencePlotter(self.visuals_dir) #
        # k_factors must be set *after* init but before record_iteration #
        # We handle this in the algorithm's fit method #
        self.matrix_plotter = None #
        self.breakdown_plotter = RecommendationBreakdownPlotter(self.visuals_dir)

        self.R = None
        self.R_predicted_final = None # To store final predictions

    def start_run(self, params, R=None):
        """Called at the beginning of the fit method."""
        super().start_run(params) # Call parent method
        self.R = R # Store the training matrix (R)

    def record_iteration(self, iteration_num, total_iterations, P, Q, objective, **kwargs): #
        """Records NCF data and saves snapshot plots."""
        # Lazily initialize matrix_plotter if k_factors is now known
        if self.matrix_plotter is None and self.k_factors > 0: #
            self.matrix_plotter = FactorMatrixPlotter(self.visuals_dir, self.k_factors) #

        super().record_iteration(iteration_num, objective=objective) #

        if self._should_plot_snapshot(iteration_num, total_iterations): #
            if self.matrix_plotter and P is not None and Q is not None: #
                # --- Delegate to FactorMatrixPlotter component ---
                manifest_entry = self.matrix_plotter.plot_snapshot( #
                    P=P, # User Embeddings #
                    Q=Q, # Item Embeddings #
                    iter_num=iteration_num, #
                    interpretation_key="Snapshots" # Use a common key #
                )
                self.visuals_manifest.append(manifest_entry) # Add to manifest #

    def _plot_convergence_graphs(self): #
        """Plots NCF convergence graphs (loss only)."""
        # Call the parent method which plots 'objective'
        super()._plot_convergence_graphs() #

    # --- ADD breakdown plotting method ---
    def _plot_recommendation_breakdown(self, R_predicted_final):
        """
        Plots the recommendation breakdown for a single sample user using final scores.
        """
        if self.R is None or R_predicted_final is None:
            print("Warning: R or final predicted scores not available. "
                  "Skipping recommendation breakdown plot.")
            return

        # 1. Find a suitable sample user (using original R for history)
        user_interaction_counts = (self.R > 0).sum(axis=1)
        sample_user_idx = np.where(
            (user_interaction_counts >= 5) & (user_interaction_counts <= 20)
        )[0]

        if len(sample_user_idx) > 0:
            sample_user_idx = sample_user_idx[0]
        elif user_interaction_counts.sum() > 0:
            sample_user_idx = np.argmax(user_interaction_counts)
        else:
            sample_user_idx = 0

        # 2. Get the necessary vectors
        user_history_vector = self.R[sample_user_idx, :]
        # Use the final predicted scores from the model
        result_vector = R_predicted_final[sample_user_idx, :]

        num_items = self.R.shape[1]
        item_names = [f"Item {i}" for i in range(num_items)]

        # 3. Call the plotter
        manifest_entry = self.breakdown_plotter.plot(
            user_history_vector=user_history_vector,
            result_vector=result_vector,
            item_names=item_names,
            user_id=str(sample_user_idx),
            k=10, # Plot Top-10
            filename="recommendation_breakdown.png",
            interpretation_key="Recommendation Breakdown"
        )
        self.visuals_manifest.append(manifest_entry)

    # --- ADD end_run method ---
    def end_run(self, R_predicted_final=None):
        """Called at the end of the fit method."""
        self.params_saved['iterations_run'] = self.iterations_run
        self._save_params() # Update with final iteration count

        self._plot_convergence_graphs() # Plot objective

        # Need final predictions for breakdown
        self._plot_recommendation_breakdown(R_predicted_final)

        self._save_history()
        self._save_visuals_manifest()