import os
import json
import numpy as np
from datetime import datetime
# Import the new plotting component
from visualization.components.ConvergencePlotter import ConvergencePlotter

from visualization.components.EmbeddingTSNEPlotter import EmbeddingTSNEPlotter
from visualization.components.FactorMatrixPlotter import FactorMatrixPlotter
from visualization.components.RecommendationBreakdownPlotter import RecommendationBreakdownPlotter
from visualization.components.ErrorDistributionPlotter import ErrorDistributionPlotter
from visualization.components.LatentDistributionPlotter import LatentDistributionPlotter
from visualization.components.VectorHistogramPlotter import VectorHistogramPlotter
from visualization.components.SingularValuesPlotter import SingularValuesPlotter
from visualization.components.SimilarityMatrixPlotter import SimilarityMatrixPlotter
from visualization.components.SparsityPatternPlotter import SparsityPatternPlotter
from visualization.components.PopularityPlotter import PopularityPlotter


class AlgorithmVisualizer:
    """Base class for handling visualization generation."""
    def __init__(self, algorithm_name, plot_interval=5):
        self.algorithm_name = algorithm_name
        self.plot_interval = plot_interval
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.visuals_base_dir = 'visuals'
        self.visuals_dir = os.path.join(self.visuals_base_dir, self.algorithm_name, self.run_timestamp)
        self.history = {} # General history tracking
        self.params_saved = {}
        self.iterations_run = 0
        self.visuals_manifest = []

    @property
    def convergence_plotter(self):
        if not hasattr(self, '_convergence_plotter'):
            self._convergence_plotter = ConvergencePlotter(self.visuals_dir)
        return self._convergence_plotter

    @property
    def similarity_plotter(self):
        if not hasattr(self, '_similarity_plotter'):
            self._similarity_plotter = SimilarityMatrixPlotter(self.visuals_dir)
        return self._similarity_plotter


    @property
    def popularity_plotter(self):
        if not hasattr(self, '_popularity_plotter'):
            self._popularity_plotter = PopularityPlotter(self.visuals_dir)
        return self._popularity_plotter

    @property
    def sparsity_plotter(self):
        if not hasattr(self, '_sparsity_plotter'):
            self._sparsity_plotter = SparsityPatternPlotter(self.visuals_dir)
        return self._sparsity_plotter

    @property
    def matrix_plotter(self):
        if not hasattr(self, '_matrix_plotter'):
            # It needs k_factors, which must be set by the child
            k = getattr(self, 'k_factors', 0)
            self._matrix_plotter = FactorMatrixPlotter(self.visuals_dir, k)
        return self._matrix_plotter

    @property
    def scree_plotter(self):
        if not hasattr(self, '_scree_plotter'):
            self._scree_plotter = SingularValuesPlotter(self.visuals_dir)
        return self._scree_plotter

    @property
    def distribution_plotter(self):
        if not hasattr(self, '_distribution_plotter'):
            self._distribution_plotter = LatentDistributionPlotter(self.visuals_dir)
        return self._distribution_plotter

    @property
    def histogram_plotter(self):
        if not hasattr(self, '_histogram_plotter'):
            self._histogram_plotter = VectorHistogramPlotter(self.visuals_dir)
        return self._histogram_plotter

    @property
    def breakdown_plotter(self):
        if not hasattr(self, '_breakdown_plotter'):
            self._breakdown_plotter = RecommendationBreakdownPlotter(self.visuals_dir)
        return self._breakdown_plotter

    @property
    def error_plotter(self):
        if not hasattr(self, '_error_plotter'):
            self._error_plotter = ErrorDistributionPlotter(self.visuals_dir)
        return self._error_plotter

    @property
    def tsne_plotter(self):
        if not hasattr(self, '_tsne_plotter'):
            self._tsne_plotter = EmbeddingTSNEPlotter(self.visuals_dir)
        return self._tsne_plotter

    def start_run(self, params):
        """Called at the beginning of the fit method."""
        os.makedirs(self.visuals_dir, exist_ok=True)
        print(f"Saving visualizations to: {os.path.abspath(self.visuals_dir)}")
        self.params_saved = {**params, 'timestamp': self.run_timestamp}
        self._save_params()
        self.visuals_manifest = []

    def record_iteration(self, iteration_num, **kwargs):
        """
        Called within the fit loop to record data and potentially save plots.
        Subclasses should override this to handle specific plots.
        'iteration_num' is 1-based.
        """
        self.iterations_run = iteration_num
        # Base implementation can store common history items
        for key, value in kwargs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def _should_plot_snapshot(self, iteration_num, total_iterations):
        """Determines if snapshot plots should be saved for this iteration."""
        is_first = (iteration_num == 1)
        is_last = (iteration_num == total_iterations)
        is_interval = (self.plot_interval > 0 and iteration_num % self.plot_interval == 0)
        return is_first or is_last or is_interval

    def end_run(self):
        """Called at the end of the fit method."""
        self.params_saved['iterations_run'] = self.iterations_run
        self._save_params() # Update with final iteration count

        self._plot_convergence_graphs() # Plot common convergence graphs

        self._save_history()
        self._save_visuals_manifest()

    def get_run_directory(self):
        """Returns the directory path for the current run."""
        return self.visuals_dir

    def get_base_directory(self):
        """Returns the base directory for visualizations."""
        return self.visuals_base_dir

    def _save_params(self):
        """Saves hyperparameters to params.json."""
        params_path = os.path.join(self.visuals_dir, 'params.json')
        try:
            with open(params_path, 'w') as f:
                # Ensure numpy types are converted if necessary (though params usually aren't)
                serializable_params = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
                                       for k, v in self.params_saved.items()}
                json.dump(serializable_params, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save parameters to {params_path}: {e}")

    def _save_history(self):
        """Saves collected history data to history.json."""
        history_path = os.path.join(self.visuals_dir, 'history.json')
        try:
            with open(history_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                serializable_history = {k: [float(item) if isinstance(item, (np.number, int, float)) else item for item in v]
                                        for k, v in self.history.items()}
                json.dump(serializable_history, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save history to {history_path}: {e}")

    def _save_visuals_manifest(self, append=False):
        """Saves the visual manifest to visuals.json."""
        manifest_path = os.path.join(self.visuals_dir, 'visuals.json')
        data_to_save = self.visuals_manifest

        if append and os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r') as f:
                    existing_data = json.load(f)
                data_to_save = existing_data + self.visuals_manifest
            except Exception as e:
                print(f"Warning: Could not read existing manifest for appending: {e}")

        try:
            with open(manifest_path, 'w') as f:
                json.dump(data_to_save, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save visuals manifest to {manifest_path}: {e}")

    def _plot_convergence_graphs(self):
        """
        Plots common convergence metrics like objective.
        Now uses the ConvergencePlotter component.
        """
        if 'objective' in self.history and self.history['objective']:
            plotter = ConvergencePlotter(self.visuals_dir)
            manifest_entry = plotter.plot(
                data_dict={'Objective Value': self.history['objective']},
                title=f'{self.algorithm_name} Objective Function over Iterations',
                y_label='Objective Value',
                filename='objective_convergence.png',
                interpretation_key='Objective' # Key for renderer to find explanation
            )
            self.visuals_manifest.append(manifest_entry) # Add to manifest

    def _plot_train_error_distribution(self, R_predicted_final, R_train_actual):
        """
        (Protected) Plots the sorted error distribution on the training set.
        Called by child's end_run().
        """
        if not hasattr(self, 'error_plotter') or self.error_plotter is None:
            print(f"Warning: 'error_plotter' not initialized for {self.algorithm_name}. Skipping train error plot.")
            return

        if R_predicted_final is None or R_train_actual is None:
            print(f"Warning: R_predicted_final or R_train_actual not available for {self.algorithm_name}. Skipping train error plot.")
            return

        try:
            manifest_entry = self.error_plotter.plot(
                R_train_actual=R_train_actual,
                R_predicted_all=R_predicted_final,
                title="Distribution of Training Set Errors (Absolute)",
                filename="error_distribution.png",
                interpretation_key="Error Distribution"
            )
            if manifest_entry:
                self.visuals_manifest.append(manifest_entry)
        except Exception as e:
            print(f"Error plotting train error distribution: {e}")

    def plot_test_errors(self, R_predicted_all_df, test_df):
        """
        (Public) Calculates test errors and saves a new plot to the run directory.
        This is called *after* .fit() and .end_run() from the main script.
        """
        if not hasattr(self, 'error_plotter') or self.error_plotter is None:
            print(f"Warning: 'error_plotter' not initialized for {self.algorithm_name}. Skipping test error plot.")
            return

        if test_df is None or R_predicted_all_df is None:
            print(f"Warning: test_df or R_predicted_all_df not provided. Skipping test error plot.")
            return

        try:
            # Use test_df to get actuals and indices
            test_indices = test_df.to_numpy().nonzero()
            if test_indices[0].size == 0:
                print("Warning (plot_test_errors): No non-zero ratings in test data. Skipping test plot.")
                return

            # Call the plotter to save the test error plot
            test_manifest_entry = self.error_plotter.plot(
                R_train_actual=test_df.to_numpy(), # Pass test data here
                R_predicted_all=R_predicted_all_df.to_numpy(),
                title="Distribution of Test Set Errors (Absolute)",
                filename="error_distribution_test.png", # Use a new filename
                interpretation_key="Error Distribution" # Same key is fine
            )

            if test_manifest_entry:
                # IMPORTANT: Append this to the visuals.json, do not overwrite
                manifest_path = os.path.join(self.visuals_dir, 'visuals.json')
                current_manifest = []
                if os.path.exists(manifest_path):
                    try:
                        with open(manifest_path, 'r') as f:
                            current_manifest = json.load(f)
                    except Exception as e:
                        print(f"Warning: could not read manifest to append test plot: {e}")

                current_manifest.append(test_manifest_entry)

                with open(manifest_path, 'w') as f:
                    json.dump(current_manifest, f, indent=4)

        except Exception as e:
            print(f"Error in {self.algorithm_name}.plot_test_errors: {e}")

    def _plot_convergence_line(self, history_key, title, y_label, filename, interp_key):
        """Generic helper to plot any single line from history."""
        if history_key in self.history and self.history[history_key]:
            manifest_entry = self.convergence_plotter.plot(
                data_dict={title: self.history[history_key]},
                title=title,
                y_label=y_label,
                filename=filename,
                interpretation_key=interp_key
            )
            self.visuals_manifest.append(manifest_entry)

    def _plot_factor_change_convergence(self, keys=['p_change', 'q_change', 'y_change', 'x_change']):
        """Plots convergence for any factor change keys found in history."""
        data_dict = {}
        key_map = {
            'p_change': 'P Change Norm',
            'q_change': 'Q Change Norm',
            'y_change': 'Y Change Norm',
            'x_change': 'X Change Norm'
        }
        for key in keys:
            if key in self.history and self.history[key]:
                # Skip first value to avoid large initial jump
                data_to_plot = self.history[key][1:] if len(self.history[key]) > 1 else self.history[key]
                data_dict[key_map.get(key, key)] = data_to_plot

        if data_dict:
            manifest_entry = self.convergence_plotter.plot(
                data_dict=data_dict,
                title='Change in Latent Factors (Frobenius Norm)',
                y_label='Norm of Difference',
                filename='factor_change_convergence.png',
                interpretation_key='Factor Change'
            )
            self.visuals_manifest.append(manifest_entry)
    def _plot_snapshot_if_needed(self, iteration_num, total_iterations, P, Q, Y=None):
        """Helper to plot factor matrix snapshots if the iteration matches."""
        if self._should_plot_snapshot(iteration_num, total_iterations):
            manifest_entry = self.matrix_plotter.plot_snapshot(
                P=P,
                Q=Q,
                Y=Y,
                iter_num=iteration_num,
                interpretation_key="Snapshots"
            )
            if manifest_entry:
                self.visuals_manifest.append(manifest_entry)

    def _plot_tsne_if_needed(self, iteration_num, total_iterations, P, Q):
        """Helper to plot t-SNE if the iteration matches."""
        if self._should_plot_snapshot(iteration_num, total_iterations):
            manifest_entry = self.tsne_plotter.plot(
                Q=Q,
                P=P,
                iter_num=iteration_num,
                title=f'Embedding t-SNE (Iter {iteration_num})',
                filename=f'tsne_iter_{iteration_num}.png',
                interpretation_key='TSNE'
            )
            if manifest_entry:
                self.visuals_manifest.append(manifest_entry)

    def _find_sample_user(self, R):
        """Finds a suitable sample user index and their history vector."""
        if R is None:
            print("Warning: R matrix not available for breakdown plot.")
            return None, None

        try:
            user_interaction_counts = (R > 0).sum(axis=1)
            # Try to find a user with a "medium" number of interactions
            sample_user_idx_arr = np.where(
                (user_interaction_counts >= 5) & (user_interaction_counts <= 20)
            )[0]

            if len(sample_user_idx_arr) > 0:
                sample_user_idx = sample_user_idx_arr[0]
            elif user_interaction_counts.sum() > 0:
                sample_user_idx = np.argmax(user_interaction_counts) # Fallback to most active
            else:
                sample_user_idx = 0 # Fallback to user 0

            user_history_vector = R[sample_user_idx, :]
            return sample_user_idx, user_history_vector

        except Exception as e:
            print(f"Error finding sample user: {e}")
            return None, None

    def _plot_recommendation_breakdown_generic(self, user_id, user_history_vector, result_vector, interpretation_key="Recommendation Breakdown"):
        """Calls the breakdown plotter with the provided vectors."""
        if user_history_vector is None or result_vector is None:
            print("Warning: Skipping breakdown plot (missing history or result vector).")
            return

        # Ensure history vector is 1D
        if hasattr(user_history_vector, 'toarray'): # Handle sparse matrix row
            user_history_vector = user_history_vector.toarray().flatten()

        num_items = len(user_history_vector)
        item_names = [f"Item {i}" for i in range(num_items)]

        manifest_entry = self.breakdown_plotter.plot(
            user_history_vector=user_history_vector,
            result_vector=result_vector,
            item_names=item_names,
            user_id=str(user_id),
            k=10, # Plot Top-10
            filename="recommendation_breakdown.png",
            interpretation_key=interpretation_key
        )
        if manifest_entry:
            self.visuals_manifest.append(manifest_entry)
