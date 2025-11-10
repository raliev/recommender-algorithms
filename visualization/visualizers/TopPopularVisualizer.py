import os
import pandas as pd
from .AlgorithmVisualizer import AlgorithmVisualizer

class TopPopularVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for TopPopularRecommender.
    Saves a bar chart of the most popular items.
    """
    def __init__(self, **kwargs):
        super().__init__("Top Popular")
        self.movie_titles_df = None
        self.item_id_map = None

    def set_title_maps(self, movie_titles_df, item_id_map):
        """Receives the title/ID maps from the Recommender's fit method."""
        self.movie_titles_df = movie_titles_df
        self.item_id_map = item_id_map

    def visualize_fit_results(self, item_popularity, params):
        """
        Called once by the algorithm's fit method.
        """
        self.start_run(params)
        self.visuals_manifest = []

        item_names = None
        if self.item_id_map is not None and self.movie_titles_df is not None:
            if len(self.item_id_map) == len(item_popularity):
                try:
                    item_names = []
                    for movie_id in self.item_id_map:
                        try:
                            title = self.movie_titles_df.loc[movie_id, 'title']
                            item_names.append(title)
                        except KeyError:
                            item_names.append(f"Movie ID {movie_id} (Unknown)")
                except Exception as e:
                    print(f"Warning (TopPopVisualizer): Could not map all movie IDs to titles: {e}")
                    item_names = [str(mid) for mid in self.item_id_map]

        if item_names is None:
            item_names = [f"Item {i}" for i in range(len(item_popularity))]

        popularity_series = pd.Series(item_popularity, index=item_names)

        manifest_entry = self.popularity_plotter.plot(
            popularity_series=popularity_series,
            title="Top 50 Most Popular Items",
            filename="top_popular_items.png",
            interpretation_key="Popularity Plot",
            top_n=50
        )
        self.visuals_manifest.append(manifest_entry)

        self.params_saved['iterations_run'] = 1
        self._save_params()
        self._save_history()
        self._save_visuals_manifest()