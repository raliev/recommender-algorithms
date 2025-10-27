# visualization/renderers/WRMFVisualizationRenderer.py
import streamlit as st
import os
import json
from .BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers

class WRMFVisualizationRenderer(BaseVisualizationRenderer):
    """Renders visualizations specific to WRMF, now reading from visuals.json."""

    def render(self):
        manifest_path = os.path.join(self.run_dir, 'visuals.json')
        if not os.path.exists(manifest_path):
            st.warning(f"Visuals manifest 'visuals.json' not found.")
            return
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception as e:
            st.error(f"Error loading 'visuals.json': {e}")
            return

        st.subheader("Convergence Plots")
        obj_plot = next((e for e in manifest if e["type"] == "line_plot"
                         and "objective" in e["file"]), None)
        factor_plot = next((e for e in manifest if e["type"] == "line_plot"
                            and "factor" in e["file"]), None)

        col_conv1, col_conv2 = st.columns(2)
        if obj_plot:
            generic_renderers.render_line_plot(self.run_dir, obj_plot,
                                               self.explanations, col_conv1)
        if factor_plot:
            generic_renderers.render_line_plot(self.run_dir, factor_plot,
                                               self.explanations, col_conv2)

        st.divider()

        snapshots = [e for e in manifest if e["type"] == "factor_snapshot"]
        snapshots.sort(key=lambda x: x["iteration"])
        if snapshots:
            first_snapshot, last_snapshot = snapshots[0], snapshots[-1]
            if first_snapshot["iteration"] == last_snapshot["iteration"]:
                st.subheader(f"Snapshot: Iteration "
                             f"{first_snapshot['iteration']}")
                generic_renderers.render_factor_snapshot(
                    self.run_dir, first_snapshot, self.explanations, st)
            else:
                st.subheader(f"Snapshot Comparison: Iteration "
                             f"{first_snapshot['iteration']} vs Iteration "
                             f"{last_snapshot['iteration']}")
                col_snap1, col_snap2 = st.columns(2)
                generic_renderers.render_factor_snapshot(
                    self.run_dir, first_snapshot,
                    self.explanations, col_snap1)
                generic_renderers.render_factor_snapshot(
                    self.run_dir, last_snapshot,
                    self.explanations, col_snap2)

        st.divider()
        st.subheader("Recommendation Breakdown")

        breakdown_plot = next((e for e in manifest if
                               e["type"] == "recommendation_breakdown"), None)

        if breakdown_plot:
            generic_renderers.render_recommendation_breakdown(
                self.run_dir, breakdown_plot, self.explanations, st
            )
        else:
            st.info("No recommendation breakdown plot was generated for this run.")
