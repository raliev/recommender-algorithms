from visualization.visualizers.AlgorithmVisualizer import AlgorithmVisualizer


class FunkSVDVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for FunkSVD.
    """
    def __init__(self, k_factors, plot_interval=5):
        super().__init__("FunkSVD", plot_interval)
        self.k_factors = k_factors

        # FunkSVD-specific history keys
        self.history['objective'] = []
        self.history['p_change'] = []
        self.history['q_change'] = []

        self.R = None
        self.P = None
        self.Q = None

    def start_run(self, params, R=None):
        """Called at the beginning of the fit method."""
        super().start_run(params)
        self.R = R # Store the training matrix (R) 

    def record_iteration(self, iteration_num, total_iterations, P, Q, objective, p_change, q_change, **kwargs):
        """Records FunkSVD data and saves snapshot plots."""
        # --- MODIFIED ---
        # Call base to update iteration count AND store all history
        super().record_iteration(iteration_num,
                                 objective=objective,
                                 p_change=p_change,
                                 q_change=q_change)

        # Store final P and Q for breakdown plot
        self.P = P
        self.Q = Q

        # Call generic helper from base class
        self._plot_snapshot_if_needed(iteration_num, total_iterations, P, Q)

    def _plot_convergence_graphs(self):
        """Plots FunkSVD convergence graphs (objective and factor changes)."""
        # Call base to plot objective (RMSE)
        super()._plot_convergence_graphs()

        # Call base to plot factor changes
        self._plot_factor_change_convergence(keys=['p_change', 'q_change'])

    def _plot_recommendation_breakdown(self):
        """
        Plots the recommendation breakdown for a single sample user.
        """
        if self.R is None or self.P is None or self.Q is None:
            print("Warning: R, P, or Q not available. "
                  "Skipping recommendation breakdown plot.")
            return

        # 1. Use base helper to find the user
        user_idx, history_vec = self._find_sample_user(self.R)
        if user_idx is None:
            return

        # 2. Calculate the algorithm-specific score vector
        all_scores = self.P @ self.Q.T
        result_vec = all_scores[user_idx, :]

        # 3. Use base helper to plot
        self._plot_recommendation_breakdown_generic(
            user_id=str(user_idx),
            user_history_vector=history_vec,
            result_vector=result_vec,
            interpretation_key="Recommendation Breakdown"
        )

    def end_run(self, R_predicted_final=None):
        """Called at the end of the fit method."""
        self.params_saved['iterations_run'] = self.iterations_run
        self._save_params() # Update with final iteration count
        self._plot_convergence_graphs()
        self._plot_recommendation_breakdown()
        # Call base class method for error distribution
        self._plot_train_error_distribution(R_predicted_final, self.R)
        self._save_history()
        self._save_visuals_manifest(append=False) 