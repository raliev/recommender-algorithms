# app/algorithms/funksvd.py
import numpy as np
from .base import Recommender

class FunkSVDRecommender(Recommender):
    def __init__(self, k, iterations=100, learning_rate=0.005, lambda_reg=0.02):
        super().__init__(k)
        self.name = "FunkSVD"
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg

    def fit(self, R, progress_callback=None, visualizer = None):
        num_users, num_items = R.shape
        self.P = np.random.normal(scale=1./self.k, size=(num_users, self.k))
        self.Q = np.random.normal(scale=1./self.k, size=(num_items, self.k))

        if visualizer:
            params_to_save = {
                'algorithm': self.name, 'k': self.k, 'iterations_set': self.iterations,
                'learning_rate': self.learning_rate, 'lambda_reg': self.lambda_reg
            }
            visualizer.start_run(params_to_save)

        for i in range(self.iterations):
            P_old = self.P.copy() if visualizer else None
            Q_old = self.Q.copy() if visualizer else None

            for u, item_idx in np.argwhere(R > 0):
                error = R[u, item_idx] - self.P[u, :] @ self.Q[item_idx, :].T
                p_u = self.P[u, :]
                q_i = self.Q[item_idx, :]

                self.P[u, :] += self.learning_rate * (error * q_i - self.lambda_reg * p_u)
                self.Q[item_idx, :] += self.learning_rate * (error * p_u - self.lambda_reg * q_i)

            if visualizer:
                p_change_norm = np.linalg.norm(self.P - P_old, 'fro')
                q_change_norm = np.linalg.norm(self.Q - Q_old, 'fro')
                current_iteration = i + 1

                visualizer.record_iteration(
                    iteration_num=current_iteration,
                    total_iterations=self.iterations,
                    P=self.P,
                    Q=self.Q,
                    p_change=p_change_norm,
                    q_change=q_change_norm
                )

            if progress_callback:
                progress_callback((i + 1) / self.iterations)

        if visualizer:
            visualizer.end_run()

        return self

    def end_run(self):
        """
        Called at the end of the fit method.
        Explicitly saves params, plots, history, and the manifest.
        """
        self.params_saved['iterations_run'] = self.iterations_run
        self._save_params() # Update with final iteration count

        self._plot_convergence_graphs() # Plot CML's convergence graphs

        self._save_history()
        self._save_visuals_manifest()