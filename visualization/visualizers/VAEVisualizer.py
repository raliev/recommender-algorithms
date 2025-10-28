import os
import numpy as np
from .AlgorithmVisualizer import AlgorithmVisualizer
from visualization.components.ConvergencePlotter import ConvergencePlotter
from visualization.components.LatentDistributionPlotter import LatentDistributionPlotter
from visualization.components.FactorMatrixPlotter import FactorMatrixPlotter # For reconstruction

class VAEVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for VAE.
    Plots loss convergence and the latent space (mu) distribution.
    """
    def __init__(self, k_factors, plot_interval=1):
        super().__init__("VAE", plot_interval)
        self.k_factors = k_factors
        self.history['objective'] = [] # For loss
        self.last_mu = None
        self.last_recon = None
        self.last_input = None

        # Instantiate components
        self.convergence_plotter = ConvergencePlotter(self.visuals_dir)
        self.distribution_plotter = LatentDistributionPlotter(self.visuals_dir)
        # We can reuse the FactorMatrixPlotter to plot the reconstruction
        self.matrix_plotter = FactorMatrixPlotter(self.visuals_dir, self.k_factors)


    def record_iteration(self, iteration_num, total_iterations, objective, mu, recon_x, x_input, **kwargs):
        """Records VAE data and stores final batch info."""
        # Call base to update iteration count and store objective
        super().record_iteration(iteration_num, objective=objective)

        # Store the outputs from the *last* iteration
        if iteration_num == total_iterations:
            self.last_mu = mu
            self.last_recon = recon_x
            self.last_input = x_input

    def _plot_latent_distribution(self):
        """Plots the final latent space distribution."""
        if self.last_mu is not None:
            manifest_entry = self.distribution_plotter.plot(
                latent_vectors=self.last_mu,
                title='Latent Space (μ) Distribution vs. Prior (N(0,1))',
                filename='latent_distribution.png',
                interpretation_key='Latent Distribution'
            )
            self.visuals_manifest.append(manifest_entry)
        else:
            print("Warning: last_mu not saved, skipping latent distribution plot.")

    def _plot_reconstruction_example(self):
        """Plots a heatmap of the original vs. reconstructed input for the last batch."""
        if self.last_recon is not None and self.last_input is not None:
            # The FactorMatrixPlotter was designed for P and Q
            # Let's just plot the reconstruction matrix directly
            manifest_entry_recon = self.matrix_plotter._plot_heatmap(
                self.last_recon,
                title=f'Reconstructed Batch (Last Epoch)',
                filename='recon_heatmap.png',
                sample_size=self.last_recon.shape[0] # Show full batch
            )
            manifest_entry_recon["type"] = "similarity_heatmap" # Use a generic type
            manifest_entry_recon["interpretation_key"] = "Reconstruction Heatmap"
            self.visuals_manifest.append(manifest_entry_recon)

            manifest_entry_orig = self.matrix_plotter._plot_heatmap(
                self.last_input,
                title=f'Original Batch (Last Epoch)',
                filename='original_heatmap.png',
                sample_size=self.last_input.shape[0] # Show full batch
            )
            manifest_entry_orig["type"] = "similarity_heatmap"
            manifest_entry_orig["interpretation_key"] = "Reconstruction Heatmap"
            self.visuals_manifest.append(manifest_entry_orig)


    def end_run(self):
        """Called at the end of the fit method."""
        self.params_saved['iterations_run'] = self.iterations_run
        self._save_params()

        # Plot objective (loss)
        self._plot_convergence_graphs()

        # Plot the new VAE-specific graphs
        self._plot_latent_distribution()
        self._plot_reconstruction_example()

        self._save_history()
        self._save_visuals_manifest()