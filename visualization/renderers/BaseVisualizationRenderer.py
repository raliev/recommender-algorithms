import streamlit as st
import os
import glob
import re
import json # <--- ADDED IMPORT
from abc import ABC, abstractmethod

from visualization import generic_renderers


class BaseVisualizationRenderer(ABC):
    """Abstract base class for algorithm-specific visualization renderers."""

    def __init__(self, run_dir, explanations):
        """
        Initialize the renderer.
        Args:
            run_dir (str): The path to the specific run directory containing visualization files.
            explanations (dict): A dictionary mapping plot keys to markdown explanations.
        """
        self.run_dir = run_dir
        self.explanations = explanations
        # --- ADDED ---
        self.algorithm_name = "Base" # Child classes should override this
        self.run_timestamp = os.path.basename(run_dir)
        # --- END ADD ---

    @abstractmethod
    def _render_plots(self, manifest): # <--- RENAMED & MADE ABSTRACT
        """
        Algorithm-specific plot rendering logic.
        This method is called by the generic render() after visuals.json is loaded.
        """
        pass

    def render(self): # <--- NEW GENERIC RENDER METHOD
        """
        Generic render method.
        Loads visuals.json and calls the child's _render_plots method.
        """
        st.header(f"{self.algorithm_name} Visualizations for {self.run_timestamp}")
        st.write(f"Displaying visualizations for algorithm: {self.algorithm_name}")

        manifest_path = os.path.join(self.run_dir, 'visuals.json')
        if not os.path.exists(manifest_path):
            st.warning(f"Visuals manifest 'visuals.json' not found in {self.run_dir}.")
            return

        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception as e:
            st.error(f"Error loading 'visuals.json': {e}")
            return

        if not manifest:
            st.info("No visualizations were generated for this run according to visuals.json.")
            return

        # Call the specific implementation
        self._render_plots(manifest)

    def show_error_distribution(self, manifest):
        st.divider()
        st.subheader("Error Distribution")
        col1, col2 = st.columns(2)

        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e["type"] == "error_distribution" and e["file"] == "error_distribution.png",
            renderer_func=generic_renderers.render_error_distribution,
            not_found_msg="No training error distribution plot found.",
            column=col1
        )

        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e["type"] == "error_distribution" and e["file"] == "error_distribution_test.png",
            renderer_func=generic_renderers.render_error_distribution,
            not_found_msg="No test error distribution plot found. (Run with a train/test split to generate this).",
            column=col2
        )
        st.divider()

    def show_factor_matrices(self, manifest):
        st.divider()
        st.subheader("Final Factor Matrices")
        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e.get("type") == "factor_snapshot",
            renderer_func=generic_renderers.render_factor_snapshot,
            not_found_msg="No latent factor snapshots found."
        )

    def show_scree_plot(self, manifest):
        st.subheader("Singular Value Analysis")
        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e.get("type") == "scree_plot",
            renderer_func=generic_renderers.render_scree_plot,
            not_found_msg="No singular value (scree) plot found."
        )

    def show_factor_snapshots(self, manifest):
        self._render_snapshot_comparison(
            manifest,
            plot_type="factor_snapshot",
            renderer_func=generic_renderers.render_factor_snapshot,
            not_found_msg="No embedding factor snapshots found."
        )

    def show_objective_plot(self, manifest):
        st.subheader("Convergence Plot")
        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e["type"] == "line_plot" and e["interpretation_key"] == "Objective",
            renderer_func=generic_renderers.render_line_plot,
            not_found_msg="No loss convergence plot found."
        )

    def show_sparcity_and_hist_plot(self, manifest):
        sparsity_plot = next((e for e in manifest if e.get("type") == "sparsity_plot"), None)
        st.subheader("Learned Model Properties (Matrix $W$)")
        st.info(
            "The algorithm learns a sparse item-item matrix $W$. These plots show the "
            "properties of this final learned model. The **Sparsity Plot** is the "
            "most important, showing *which* item pairs have a learned relationship."
        )
        col1, col2 = st.columns(2)
        if sparsity_plot:
            # Use a generic renderer for the image (this part is unique)
            generic_renderers._render_image_with_interpretation(
                os.path.join(self.run_dir, sparsity_plot["file"]),
                sparsity_plot["name"],
                self.explanations.get(sparsity_plot["interpretation_key"], "Explanation not found."),
                column=col1
            )
        else:
            col1.caption("Sparsity plot not found in manifest.")

        # Refactored histogram plot
        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e.get("type") == "histogram",
            renderer_func=generic_renderers.render_histogram,
            not_found_msg="Coefficient distribution plot not found in manifest.",
            column=col2
        )

    def show_convergence_plot(self, manifest):
        st.divider()
        st.subheader("Convergence Plots")
        col_conv1, col_conv2 = st.columns(2)

        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e["type"] == "line_plot" and e["interpretation_key"] == "Objective",
            renderer_func=generic_renderers.render_line_plot,
            not_found_msg="No objective convergence plot (RMSE) found.",
            column=col_conv1
        )

        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e["type"] == "line_plot" and e["interpretation_key"] == "Factor Change",
            renderer_func=generic_renderers.render_line_plot,
            not_found_msg="No factor change convergence plot found.",
            column=col_conv2
        )

    def show_heatmap_hist_plot(self, manifest):
        st.divider()
        st.subheader("Learned Model Properties (Matrix $B$)")
        col1, col2 = st.columns(2)

        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e.get("interpretation_key") == "Final Similarity Heatmap",
            renderer_func=generic_renderers.render_similarity_heatmap,
            not_found_msg="Learned B matrix heatmap not found.",
            column=col1
        )

        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e.get("interpretation_key") == "Histogram of Final Similarity Values",
            renderer_func=generic_renderers.render_histogram,
            not_found_msg="Coefficient distribution plot not found.",
            column=col2
        )

    def show_histograms_raw_final_corated_plot(self, manifest):
        col_s1, col_s2 = st.columns(2)

        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e["interpretation_key"] == "Raw Similarity Heatmap",
            renderer_func=generic_renderers.render_similarity_heatmap,
            not_found_msg="Raw similarity heatmap not found.",
            column=col_s1
        )
        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e["interpretation_key"] == "Final (Adjusted) Similarity Heatmap",
            renderer_func=generic_renderers.render_similarity_heatmap,
            not_found_msg="Final (Adjusted) similarity heatmap not found.",
            column=col_s2
        )

        st.divider()
        col_s3, col_s4 = st.columns(2)

        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e["interpretation_key"] == "Co-rated Counts Heatmap",
            renderer_func=generic_renderers.render_similarity_heatmap,
            not_found_msg="Co-rated counts heatmap not generated/found.",
            column=col_s3
        )
        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e["interpretation_key"] == "Histogram of Final Similarity Values",
            renderer_func=generic_renderers.render_histogram,
            not_found_msg="Histogram of Final Similarity Values not found.",
            column=col_s4
        )

    def show_factor_1column(self, manifest):
        st.subheader("Convergence Plots")
        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e["type"] == "line_plot" and e["interpretation_key"] == "Factor Change",
            renderer_func=generic_renderers.render_line_plot,
            not_found_msg="No factor change convergence plot found."
        )

    def show_factor_auc_2column(self, manifest):
        st.divider()
        st.subheader("Convergence Plots")
        col1, col2 = st.columns(2)

        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e["type"] == "line_plot" and e["interpretation_key"] == "Factor Change",
            renderer_func=generic_renderers.render_line_plot,
            not_found_msg="No factor convergence plot found.",
            column=col1
        )

        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e["type"] == "line_plot" and e["interpretation_key"] == "AUC",
            renderer_func=generic_renderers.render_line_plot,
            not_found_msg="No AUC convergence plot found. (Note: This must be calculated and passed by the algorithm's 'fit' method).",
            column=col2
        )

    def show_factor_auc_plots_3column(self, manifest):
        st.subheader("Convergence Plots")
        col1, col2, col3 = st.columns(3)

        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e["type"] == "line_plot" and e["interpretation_key"] == "Factor Change",
            renderer_func=generic_renderers.render_line_plot,
            not_found_msg="No factor convergence plot found.",
            column=col1
        )

        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e["type"] == "line_plot" and e["interpretation_key"] == "Avg. Negative Score",
            renderer_func=generic_renderers.render_line_plot,
            not_found_msg="No average negative score plot found.",
            column=col2
        )

        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e["type"] == "line_plot" and e["interpretation_key"] == "AUC",
            renderer_func=generic_renderers.render_line_plot,
            not_found_msg="No AUC convergence plot found. (Note: This must be calculated and passed by the algorithm's 'fit' method).",
            column=col3
        )


    def show_latent_space_distribution(self, manifest):
        st.divider()
        st.subheader("Latent Space Distribution")
        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e["type"] == "latent_distribution",
            renderer_func=generic_renderers.render_latent_distribution_plot,
            not_found_msg="No latent distribution plot found."
        )

    def show_original_reconstructed_plot(self, manifest):
        st.divider()
        st.subheader("Reconstruction Example (Last Batch)")
        col1, col2 = st.columns(2)

        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e["type"] == "similarity_heatmap" and "Original" in e["name"],
            renderer_func=generic_renderers.render_similarity_heatmap,
            not_found_msg="Original batch heatmap not found.",
            column=col1
        )

        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e["type"] == "similarity_heatmap" and "Reconstructed" in e["name"],
            renderer_func=generic_renderers.render_similarity_heatmap,
            not_found_msg="Reconstructed batch heatmap not found.",
            column=col2
        )

    def show_popularity_plot(self, manifest):
        st.divider()
        st.subheader("Popularity Plot")
        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e["type"] == "bar_plot",
            renderer_func=generic_renderers.render_line_plot, # Re-uses render_line_plot
            not_found_msg="No popularity plot found for this run."
        )

    def show_latent_factor_snapshots(self, manifest):
        self._render_snapshot_comparison(
            manifest,
            plot_type="factor_snapshot",
            renderer_func=generic_renderers.render_factor_snapshot,
            not_found_msg="No latent factor snapshots found.",
            subheader="Latent Factor Snapshots"
        )

    def show_item_weights_histogram_plot(self, manifest):
        st.divider()
        st.subheader("Global Item Weights (PageRank)")
        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e["type"] == "histogram" and e["interpretation_key"] == "Item Weights",
            renderer_func=generic_renderers.render_histogram,
            not_found_msg="No item weights histogram plot found."
        )

    def show_tsne_plot(self, manifest):
        self._render_snapshot_comparison(
            manifest,
            plot_type="tsne_plot",
            renderer_func=generic_renderers.render_tsne_plot,
            not_found_msg="No t-SNE plots found.",
            subheader="Embedding t-SNE Plot"
        )


    def show_breakdown_plot(self, manifest):
        st.divider()
        st.subheader("Recommendation Breakdown")
        self._find_and_render_plot(
            manifest,
            find_filter=lambda e: e["type"] == "recommendation_breakdown",
            renderer_func=generic_renderers.render_recommendation_breakdown,
            not_found_msg="No recommendation breakdown plot was generated for this run."
        )

    # --- Reusable Helper Methods ---

    def _find_and_render_plot(self, manifest, find_filter, renderer_func, not_found_msg, column=None):
        """
        (New Helper) Finds a single plot in the manifest and renders it, or shows a 'not found' message.
        """
        if column is None:
            column = st

        plot_entry = next((e for e in manifest if find_filter(e)), None)

        if plot_entry:
            renderer_func(self.run_dir, plot_entry, self.explanations, column)
        else:
            column.info(not_found_msg)

    def _render_snapshot_comparison(self, manifest, plot_type, renderer_func, not_found_msg, subheader=None, iteration_key="iteration"):
        """
        (New Helper) Renders a 1- or 2-column comparison for plots that
        are saved at different iterations (e.g., factor_snapshot, tsne_plot).
        """
        if subheader:
            st.divider()
            st.subheader(subheader)

        snapshots = [e for e in manifest if e.get("type") == plot_type]
        if not snapshots:
            st.info(not_found_msg)
            return

        snapshots.sort(key=lambda x: x.get(iteration_key, 0))

        first = snapshots[0]
        last = snapshots[-1] if len(snapshots) > 1 else snapshots[0]

        iter_first = first.get(iteration_key, 0)
        iter_last = last.get(iteration_key, 0)

        if iter_first == iter_last:
            st.subheader(f"Snapshot: Iteration {iter_first}")
            renderer_func(self.run_dir, first, self.explanations, st)
        else:
            st.subheader(
                f"Snapshot Comparison: Iteration {iter_first} vs Iteration {iter_last}")
            col1, col2 = st.columns(2)
            renderer_func(self.run_dir, first, self.explanations, col1)
            renderer_func(self.run_dir, last, self.explanations, col2)

    def _render_image(self, file_path, caption, explanation_key=None, column=None):
        """Helper function to render an image with optional caption and explanation."""
        container = column if column else st # Render in column or directly
        if os.path.exists(file_path):
            container.image(file_path, caption=caption)
            if explanation_key:
                with container.expander("Interpretation"):
                    st.markdown(self.explanations.get(explanation_key, 'Explanation not found.'), unsafe_allow_html=True)
        else:
            container.caption(f"{caption} not found.")