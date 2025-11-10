import os
import json
import numpy as np
from datetime import datetime
# Import the new plotting component
from visualization.components.ConvergencePlotter import ConvergencePlotter

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

    def _save_visuals_manifest(self):
        """Saves the visual manifest to visuals.json."""
        manifest_path = os.path.join(self.visuals_dir, 'visuals.json')
        try:
            with open(manifest_path, 'w') as f:
                json.dump(self.visuals_manifest, f, indent=4)
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

