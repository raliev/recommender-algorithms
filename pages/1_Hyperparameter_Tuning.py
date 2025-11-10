import streamlit as st
import pandas as pd
import numpy as np
from tuner_config import TUNER_CONFIG
from tuner_runner import run_tuning_process
from utils import get_generated_datasets

st.set_page_config(page_title="Hyperparameter Tuning", layout="wide")
st.title("Hyperparameter Tuning")

st.sidebar.header("Data Source")
base_data_options = ["Synthetic 20x20", "Load MovieLens CSV"]
generated_data_options = get_generated_datasets()
all_data_options = base_data_options + generated_data_options
data_source = st.sidebar.radio(
    "Select Data Source",
#    ["Synthetic 20x20", "Load MovieLens CSV"],
    all_data_options,
    key="tuner_data_source"
)

data_params = {}
if data_source == "Load MovieLens CSV":
    data_params['num_users'] = st.sidebar.slider("Number of Users", 50, 610, 100)
    data_params['num_movies'] = st.sidebar.slider("Number of Movies", 100, 2000, 500)
elif data_source in generated_data_options:
    st.sidebar.info(f"Using your custom generated dataset: '{data_source}'")
st.sidebar.header("Algorithm Selection")
selected_algorithms = {}
for algo_name, params in TUNER_CONFIG.items():
    if st.sidebar.checkbox(algo_name, value=True, key=f"select_{algo_name}"):
        selected_algorithms[algo_name] = params

st.header("Configure Hyperparameters")

if not selected_algorithms:
    st.warning("Please select at least one algorithm from the sidebar.")
    st.stop()

configured_params = {}
total_iterations = 0

n_trials = st.sidebar.slider(
    "Number of Trials per Algorithm (BO)",
    min_value=5,
    max_value=200,
    value=20,
    step=5,
    help="Number of iterations."
)


for algo_name, params in selected_algorithms.items():
    expander = st.expander(f"**{algo_name}**", expanded=True)

    algo_configs = {}

    param_names = list(params.keys())

    for param_name in param_names:
        config = params[param_name]

        if "options" in config:
            with expander:
                algo_configs[param_name] = st.multiselect(
                    f"{param_name}",
                    options=config["options"],
                    default=config["default"],
                    key=f"{algo_name}_{param_name}"
                )
        else:
            expander.markdown(f"**{param_name}**")
            cols = expander.columns(2)

            min_val, max_val, default_val = config['min'], config['max'], config['default']

            start_val = cols[0].number_input("Min Value", value=float(min_val), key=f"{algo_name}_{param_name}_start")
            end_val = cols[1].number_input("Max Value", value=float(max_val), key=f"{algo_name}_{param_name}_end")

            is_int = isinstance(default_val, int) and isinstance(min_val, int) and isinstance(max_val, int)

            algo_configs[param_name] = {'low': start_val, 'high': end_val, 'is_int': is_int}
            expander.divider()


    expander.info(f"For {algo_name} the system will run **{n_trials}** iterations (Bayes optimization).")
    total_iterations += n_trials
    configured_params[algo_name] = algo_configs

st.success(f"**Total interations for all algorithms: {total_iterations}**")

run_button = st.button("Run Tuning Process", use_container_width=True, type="primary")

if run_button:
    if total_iterations > 0:
        run_tuning_process(
            configured_params,
            data_source,
            data_params,
            n_trials # Pass the number of trials instead of total_iterations
        )
    else:
        st.error("Нет комбинаций для тестирования. Пожалуйста, выберите хотя бы один алгоритм.")