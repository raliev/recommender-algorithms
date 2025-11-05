import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .BasePlotter import BasePlotter

try:
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class EmbeddingTSNEPlotter(BasePlotter):
    """
    Plots a 2D t-SNE visualization of an embedding matrix (e.g., Q).
    """

    def __init__(self, visuals_dir):
        super().__init__(visuals_dir)
        if not SKLEARN_AVAILABLE:
            print("Warning: scikit-learn not installed. t-SNE plots will be unavailable.")

    def plot(self, Q: np.ndarray, iter_num: int, title: str,
             filename: str, interpretation_key: str, P: np.ndarray = None):
        """
        Generates and saves a 2D t-SNE scatter plot.

        Args:
            Q (np.ndarray): Item embedding matrix (n_items x k_factors).
            iter_num (int): The current iteration number.
            title (str): The main title for the plot.
            filename (str): The filename to save the plot as.
            interpretation_key (str): The key for the manifest.
            P (np.ndarray, optional): User embedding matrix (n_users x k_factors).

        Returns:
            dict: The manifest entry for this visualization.
        """
        if not SKLEARN_AVAILABLE:
            return None # Cannot plot

        fig, ax = plt.subplots(figsize=(10, 8))

        # Combine P and Q if P is provided
        if P is not None:
            n_users = P.shape[0]
            n_items = Q.shape[0]
            embeddings = np.vstack([P, Q])
            labels = ['User'] * n_users + ['Item'] * n_items
            palette = {"User": "blue", "Item": "red"}
        else:
            embeddings = Q
            labels = ['Item'] * Q.shape[0]
            palette = {"Item": "red"}

        # Reduce sample size if too large (t-SNE is slow)
        sample_size = 1000
        if embeddings.shape[0] > sample_size:
            indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
            embeddings = embeddings[indices]
            labels = [labels[i] for i in indices]
            title += f" (Sampled {sample_size})"

        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, embeddings.shape[0] - 1))
            embeddings_2d = tsne.fit_transform(embeddings)

            plot_data = {"dim_1": embeddings_2d[:, 0], "dim_2": embeddings_2d[:, 1], "label": labels}

            sns.scatterplot(
                data=plot_data,
                x="dim_1",
                y="dim_2",
                hue="label",
                palette=palette,
                alpha=0.6,
                ax=ax
            )
            ax.set_title(title)
            ax.set_xlabel("t-SNE Dimension 1")
            ax.set_ylabel("t-SNE Dimension 2")

        except Exception as e:
            ax.text(0.5, 0.5, f't-SNE Error: {e}', ha='center', va='center')

        plt.tight_layout()
        file_path = self._save_plot(fig, filename)

        return {
            "name": title,
            "type": "tsne_plot",
            "file": os.path.basename(file_path),
            "iteration": iter_num,
            "interpretation_key": interpretation_key
        }