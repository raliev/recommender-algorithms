# Defines the hyperparameter search space for each algorithm.
# The default 'step' values are chosen as a compromise between search granularity and performance.
TUNER_CONFIG = {
    "SVD": {
        "k": {"min": 2, "max": 100, "step": 4, "default": 32}
    },
    "ALS": {
        "k": {"min": 2, "max": 100, "step": 4, "default": 32},
        "iterations": {"min": 5, "max": 30, "step": 5, "default": 10},
        "lambda_reg": {"min": 0.01, "max": 0.5, "step": 0.05, "default": 0.1}
    },
    "ALS (Improved)": {
        "k": {"min": 2, "max": 100, "step": 4, "default": 32},
        "iterations": {"min": 5, "max": 30, "step": 5, "default": 10},
        "lambda_reg": {"min": 0.01, "max": 0.5, "step": 0.05, "default": 0.05},
        "lambda_biases": {"min": 1.0, "max": 20.0, "step": 2.0, "default": 10.0}
    },
    "ALS (PySpark)": {
        "k": {"min": 2, "max": 100, "step": 4, "default": 32},
        "maxIter": {"min": 5, "max": 25, "step": 5, "default": 10},
        "regParam": {"min": 0.01, "max": 0.20,  "step": 0.05, "default": 0.05}
    },
    "BPR": {
        "k": {"min": 10, "max": 100, "step": 10, "default": 32},
        "iterations": {"min": 500, "max": 5000, "step": 500, "default": 1000},
        "learning_rate": {"min": 0.001, "max": 0.1, "step": 0.005, "default": 0.01},
        "lambda_reg": {"min": 0.001, "max": 0.1, "step": 0.005, "default": 0.01}
    },
    "ItemKNN": {
        "k": {"min": 5, "max": 100, "step": 5, "default": 20},
        # similarity_metric is categorical, so we list options instead of min/max/step
        "similarity_metric": {"options": ["cosine", "adjusted_cosine", "pearson"], "default": "cosine"},
        "min_support": {"min": 0, "max": 10, "step": 1, "default": 2},
        "shrinkage": {"min": 0.0, "max": 100.0, "step": 10.0, "default": 0.0}
    },
    "UserKNN": {
        "k": {"min": 5, "max": 100, "step": 5, "default": 20},
        "similarity_metric": {"options": ["cosine", "adjusted_cosine", "pearson"], "default": "cosine"}
    },
    "NMF": {
        "k": {"min": 2, "max": 100, "step": 4, "default": 32},
        "max_iter": {"min": 50, "max": 500, "step": 50, "default": 200}
    },
    "FunkSVD": {
        "k": {"min": 2, "max": 100, "step": 4, "default": 32},
        "iterations": {"min": 5, "max": 30, "step": 5, "default": 10},
        "learning_rate": {"min": 0.001, "max": 0.01, "step": 0.001, "default": 0.005},
        "lambda_reg": {"min": 0.01, "max": 0.5, "step": 0.05, "default": 0.02}
    },
    "SVD++": {
        "k": {"min": 10, "max": 200, "step": 5, "default": 30},
        "iterations": {"min": 5, "max": 30, "step": 5, "default": 20},
        "learning_rate": {"min":0.0001 , "max": 0.1, "step": 0.01, "default": 0.005},
        "lambda_p": {"min": 0.001, "max": 0.5, "step": 0.01, "default": 0.02},
        "lambda_q": {"min": 0.001, "max": 0.5, "step": 0.01, "default": 0.02},
        "lambda_y": {"min": 0.001, "max": 0.5, "step": 0.01, "default": 0.02},
        "lambda_bu": {"min": 0.1, "max": 20.0, "step": 0.5, "default": 5.0}, # Wider range for biases
        "lambda_bi": {"min": 0.1, "max": 20.0, "step": 0.5, "default": 5.0}  # Wider range for biases
    },
    "WRMF": {
        "k": {"min": 10, "max": 100, "step": 10, "default": 32},
        "iterations": {"min": 5, "max": 30, "step": 5, "default": 10},
        "lambda_reg": {"min": 0.01, "max": 1.0, "step": 0.1, "default": 0.1},
        "alpha": {"min": 1, "max": 100, "step": 10, "default": 40}
    },
}