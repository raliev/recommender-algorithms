import streamlit as st
import os
import json
import pandas as pd
from .BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers

class AssociationRuleVisualizationRenderer(BaseVisualizationRenderer):
    """
    Renders visualizations for Association Rule models.
    It loads and displays DataFrames from JSON files.
    """

    def _render_dataframe(self, manifest_entry, column=None):
        """Helper to load and render a DataFrame."""
        container = column if column else st
        file_path = os.path.join(self.run_dir, manifest_entry["file"])

        if not os.path.exists(file_path):
            container.caption(f"{manifest_entry['name']} (File not found)")
            return

        try:
            df = pd.read_json(file_path, orient='records')

            # Convert antecedent/consequent lists back to strings for display
            if 'antecedents' in df.columns:
                df['antecedents'] = df['antecedents'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
            if 'consequents' in df.columns:
                df['consequents'] = df['consequents'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

            container.subheader(manifest_entry['name'])
            container.dataframe(df)

            interp_key = manifest_entry["interpretation_key"]
            interp_text = self.explanations.get(interp_key, f"Explanation for '{interp_key}' not found.")
            with container.expander("Interpretation"):
                st.markdown(interp_text, unsafe_allow_html=True)

        except Exception as e:
            container.error(f"Error loading {manifest_entry['name']}: {e}")

    def render(self):
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

        st.header("Model Internals")

        itemsets_entry = next((e for e in manifest if e["interpretation_key"] == "Frequent Itemsets"), None)
        rules_entry = next((e for e in manifest if e["interpretation_key"] == "Association Rules"), None)

        if itemsets_entry:
            self._render_dataframe(itemsets_entry)
        else:
            st.info("No frequent itemsets data found.")

        st.divider()

        if rules_entry:
            self._render_dataframe(rules_entry)
        else:
            st.info("No association rules data found.")