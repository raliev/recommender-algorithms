import os
import matplotlib.pyplot as plt
import seaborn as sns
from .BasePlotter import BasePlotter

class PopularityPlotter(BasePlotter):
    """
    Plots a horizontal bar chart for item popularity.
    """

    def plot(self, popularity_series, title, filename, interpretation_key, top_n=50):
        """
        Generates and saves a bar plot of the top_n most popular items.

        Args:
            popularity_series (pd.Series): A Series with item IDs as index and counts as values.
            top_n (int): Number of top items to display.

        Returns:
            dict: The manifest entry for this visualization.
        """

        top_items = popularity_series.nlargest(top_n).sort_values(ascending=True)

        if top_items.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'No popularity data to display.', horizontalalignment='center', verticalalignment='center')
        else:
            fig_height = max(5, len(top_items) * 0.3)
            fig, ax = plt.subplots(figsize=(10, fig_height))

            sns.barplot(
                x=top_items.values,
                y=top_items.index.astype(str),
                ax=ax,
                orient='h',
                palette="viridis"
            )

            ax.set_title(title)
            ax.set_xlabel('Total Interactions (Popularity Count)')
            ax.set_ylabel('Item ID')

        plt.tight_layout()
        file_path = self._save_plot(fig, filename)

        # Return manifest entry
        return {
            "name": title,
            "type": "bar_plot", # A new generic type
            "file": os.path.basename(file_path),
            "interpretation_key": interpretation_key
        }