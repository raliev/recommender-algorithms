import os
import numpy as np
import matplotlib.pyplot as plt
from visualization.components.BasePlotter import BasePlotter

class ErrorDistributionPlotter(BasePlotter):
    """
    Plots the sorted distribution of errors (e.g., absolute error)
    on a percentile-based X-axis to visualize the "long tail" of errors.
    """

    def plot(self, R_train_actual: np.ndarray, R_predicted_all: np.ndarray,
             title: str, filename: str, interpretation_key: str):
        """
        Generates and saves the error distribution plot.

        Args:
            R_train_actual (np.ndarray): The training matrix with actual ratings (e.g., self.R).
            R_predicted_all (np.ndarray): The full predicted matrix (e.g., P @ Q.T).
            title (str): The main title for the plot.
            filename (str): The filename to save the plot as.
            interpretation_key (str): The key for the manifest.

        Returns:
            dict: The manifest entry for this visualization, or None if errors.
        """
        try:
            # Find the indices of rated items in the training set
            train_indices = R_train_actual.nonzero()
            if train_indices[0].size == 0:
                print("Warning (ErrorDistributionPlotter): No non-zero ratings found in training data. Skipping plot.")
                return None

            # Extract the actual and predicted ratings for these items
            train_actuals = R_train_actual[train_indices]
            train_preds = R_predicted_all[train_indices]

            # Calculate absolute errors
            train_errors = np.abs(train_actuals - train_preds)

            # Sort errors in descending order
            train_errors_sorted = np.sort(train_errors)[::-1]

            # Create the percentile X-axis (0% to 100%)
            train_x_percentile = np.linspace(0, 100, len(train_errors_sorted))

            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(train_x_percentile, train_errors_sorted, label='Abs. Error', color='blue')
            ax.set_xlabel("Prediction Percentile (Sorted by Error)")
            ax.set_ylabel("Absolute Error (|Actual - Predicted|)")
            ax.set_title(title)
            ax.legend()
            ax.grid(True)
            plt.tight_layout()

            # Save the plot
            file_path = self._save_plot(fig, filename)

            # Return manifest entry
            return {
                "name": title,
                "type": "error_distribution", # A new generic type
                "file": os.path.basename(file_path),
                "interpretation_key": interpretation_key
            }
        except Exception as e:
            print(f"Error in ErrorDistributionPlotter: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None