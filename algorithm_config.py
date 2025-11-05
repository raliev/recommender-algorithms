import streamlit as st # Needed for widget functions

from algorithms import (
    SVDRecommender,
    ALSRecommender,
    ALSImprovedRecommender,
    ALSPySparkRecommender,
    BPRRecommender,
    BPRAdaptiveRecommender,
    ItemKNNRecommender,
    UserKNNRecommender,
    SlopeOneRecommender,
    NMFRecommender,
    FunkSVDRecommender,
    PureSVDRecommender,
    SVDppRecommender,
    WRMFRecommender,
    CMLRecommender,
    NCFRecommender,
    SASRecRecommender,
    VAERecommender,
    SLIMRecommender,
    FISMRecommender
)
from algorithms import AprioriRecommender, EclatRecommender, TopPopularRecommender, FPGrowthRecommender
from visualization.renderers.AssociationRuleVisualizationRenderer import AssociationRuleVisualizationRenderer
from visualization.renderers.TopPopularVisualizationRenderer import TopPopularVisualizationRenderer
from visualization.visualizers.AssociationRuleVisualizer import AssociationRuleVisualizer
from visualization.visualizers.TopPopularVisualizer import TopPopularVisualizer
from visualization.renderers.NMFVisualizationRenderer import NMFVisualizationRenderer
from visualization.visualizers.NMFVisualizer import NMFVisualizer
from visualization.renderers.BPRAdaptiveVisualizationRenderer import BPRAdaptiveVisualizationRenderer
from visualization.visualizers.BPRAdaptiveVisualizer import BPRAdaptiveVisualizer
from visualization.renderers.SASRecVisualizationRenderer import SASRecVisualizationRenderer
from visualization.visualizers.SASRecVisualizer import SASRecVisualizer
from visualization.renderers.VAEVisualizationRenderer import VAEVisualizationRenderer
from visualization.visualizers.VAEVisualizer import VAEVisualizer
from visualization.renderers.PureSVDVisualizationRenderer import PureSVDVisualizationRenderer
from visualization.renderers.SVDVisualizationRenderer import SVDVisualizationRenderer
from visualization.visualizers.PureSVDVisualizer import PureSVDVisualizer
from visualization.visualizers.SVDVisualizer import SVDVisualizer
from visualization.renderers.ALSImprovedVisualizationRenderer import ALSImprovedVisualizationRenderer
from visualization.visualizers.ALSImprovedVisualizer import ALSImprovedVisualizer
from visualization.renderers.ALSVisualizationRenderer import ALSVisualizationRenderer
from visualization.visualizers.ALSVisualizer import ALSVisualizer
from visualization.renderers.NCFVisualizationRenderer import NCFVisualizationRenderer
from visualization.visualizers.NCFVisualizer import NCFVisualizer
from visualization.renderers.CMLVisualizationRenderer import CMLVisualizationRenderer
from visualization.visualizers.CMLVisualizer import CMLVisualizer
from visualization.renderers.FISMVisualizationRenderer import FISMVisualizationRenderer
from visualization.visualizers.FISMVisualizer import FISMVisualizer
from visualization.visualizers.SlopeOneVisualizer import SlopeOneVisualizer
from visualization.renderers.SVDppVisualizationRenderer import SVDppVisualizationRenderer
from visualization.visualizers.SVDppVisualizer import SVDppVisualizer
from visualization.renderers.FunkSVDVisualizationRenderer import FunkSVDVisualizationRenderer
from visualization.visualizers.FunkSVDVisualizer import FunkSVDVisualizer
from visualization.renderers.SLIMVisualizationRenderer import SLIMVisualizationRenderer
from visualization.visualizers.SLIMVisualizer import SLIMVisualizer
from visualization.renderers.BPRVisualizationRenderer import BPRVisualizationRenderer
from visualization.renderers.KNNVisualizationRenderer import KNNVisualizationRenderer
from visualization.renderers.WRMFVisualizationRenderer import WRMFVisualizationRenderer
from visualization.visualizers.BPRVisualizer import BPRVisualizer
from visualization.visualizers.ItemKNNVisualizer import ItemKNNVisualizer
from visualization.visualizers.UserKNNVisualizer import UserKNNVisualizer
from visualization.visualizers.WRMFVisualizer import WRMFVisualizer

WIDGET_MAP = {
    "slider": st.sidebar.slider,
    "selectbox": st.sidebar.selectbox,
    "select_slider": st.sidebar.select_slider,
}

ALGORITHM_CONFIG = {
    "Top Popular": {
        "is_implicit": True,
        "model_class": TopPopularRecommender,
        "parameters": {},
        "info": "Recommends items based on their overall popularity. A non-personalized baseline. ",
        "result_type": "other",
        "visualizer_class": TopPopularVisualizer,
        "visualization_renderer_class": TopPopularVisualizationRenderer
    },
    "Apriori": {
        "is_implicit": True,
        "model_class": AprioriRecommender,
        "parameters": {
            "min_support": {"type": "slider", "label": "Min Support", "min": 0.02, "max": 1.0, "default": 0.3, "step": 0.01, "format": "%.2f"},
            "min_confidence": {"type": "slider", "label": "Min Confidence", "min": 0.0, "max": 1.0, "default": 0.2, "step": 0.01}
        },
        "info": "Finds frequent itemsets using the Apriori algorithm  and generates 'if-then' rules. This is a 'from scratch' implementation for educational purposes.",
        "result_type": "association_rules",
        "visualizer_class": AssociationRuleVisualizer,
        "visualization_renderer_class": AssociationRuleVisualizationRenderer
    },
    "FP-Growth": {
        "is_implicit": True,
        "model_class": FPGrowthRecommender,
        "parameters": {
            "min_support": {"type": "slider", "label": "Min Support", "min": 0.01, "max": 1.0, "default": 0.2, "step": 0.01, "format": "%.2f"},
            "min_confidence": {"type": "slider", "label": "Min Confidence", "min": 0.0, "max": 1.0, "default": 0.2, "step": 0.01}
        },
        "info": "Finds frequent itemsets using a tree-based FP-tree structure . This implementation uses the `mlxtend` library.",
        "result_type": "association_rules",
        "visualizer_class": AssociationRuleVisualizer,
        "visualization_renderer_class": AssociationRuleVisualizationRenderer
    },
    "Eclat": {
        "is_implicit": True,
        "model_class": EclatRecommender,
        "parameters": {
            "min_support": {"type": "slider", "label": "Min Support", "min": 0.01, "max": 1.0, "default": 0.2, "step": 0.01, "format": "%.2f"},
            "min_confidence": {"type": "slider", "label": "Min Confidence", "min": 0.0, "max": 1.0, "default": 0.2, "step": 0.01}
        },
        "info": "Finds frequent itemsets using a vertical data layout and set intersections . This is a 'from scratch' implementation for educational purposes.",
        "result_type": "association_rules",
        "visualizer_class": AssociationRuleVisualizer,
        "visualization_renderer_class": AssociationRuleVisualizationRenderer
    },
    "SVD": {
        "is_implicit": False,
        "model_class": SVDRecommender, 
        "parameters": {
            "k": {"type": "slider", "label": "Latent Factors (k)", "min": 1, "max": 100, "default": 20}
        },
        "result_type": "matrix_factorization",
        "visualizer_class": SVDVisualizer,
        "visualization_renderer_class": SVDVisualizationRenderer
    },
    "ALS": {
        "model_class": ALSRecommender,
        "is_implicit": False,
        "parameters": {
            "k": {"type": "slider", "label": "Latent Factors (k)", "min": 1, "max": 100, "default": 32},
            "iterations": {"type": "slider", "label": "Iterations", "min": 1, "max": 30, "default": 10},
            "lambda_reg": {"type": "slider", "label": "Regularization (lambda)", "min": 0.0, "max": 1.0, "default": 0.1, "step": 0.01}
        },
        "result_type": "matrix_factorization",
        "visualizer_class": ALSVisualizer,
        "visualization_renderer_class": ALSVisualizationRenderer
    },
    "ALS (Improved)": {
        "model_class": ALSImprovedRecommender,
        "is_implicit": False,
        "parameters": {
            "k": {"type": "slider", "label": "Latent Factors (k)", "min": 1, "max": 100, "default": 32},
            "iterations": {"type": "slider", "label": "Iterations", "min": 1, "max": 30, "default": 10},
            "lambda_reg": {"type": "slider", "label": "Regularization (Factors)", "min": 0.0, "max": 1.0, "default": 0.05, "step": 0.01},
            "lambda_biases": {"type": "slider", "label": "Regularization (Biases)", "min": 0.0, "max": 20.0, "default": 10.0, "step": 0.5}
        },
        "result_type": "matrix_factorization",
        "visualizer_class": ALSImprovedVisualizer,
        "visualization_renderer_class": ALSImprovedVisualizationRenderer
    },
    "ALS (PySpark)": {
        "model_class": ALSPySparkRecommender,
        "is_implicit": False,
        "parameters": {
            "k": {"type": "slider", "label": "Latent Factors (k)", "min": 1, "max": 100, "default": 32},
            "iterations": {"type": "slider", "label": "Iterations (maxIter)", "min": 5, "max": 25, "default": 10},
            "lambda_reg": {"type": "slider", "label": "Regularization (regParam)", "min": 0.01, "max": 0.2, "default": 0.1, "step": 0.01}
        },
        "result_type": "matrix_factorization",
        "visualizer_class": None
    },
    "BPR": {
        "model_class": BPRRecommender,
        "is_implicit": True,
        "parameters": {
            "k": {"type": "slider", "label": "Latent Factors (k)", "min": 1, "max": 100, "default": 32},
            "iterations": {"type": "slider", "label": "Iterations", "min": 100, "max": 5000, "default": 200, "step": 100},
            "learning_rate": {"type": "slider", "label": "Learning Rate (alpha)", "min": 0.001, "max": 0.1, "default": 0.01, "step": 0.001, "format": "%.3f"},
            "lambda_reg": {"type": "slider", "label": "Regularization (lambda)", "min": 0.0, "max": 0.1, "default": 0.01, "step": 0.001, "format": "%.3f"}
        },
        "result_type": "bpr",
        "visualizer_class": BPRVisualizer,
        "visualization_renderer_class": BPRVisualizationRenderer
    },
    "BPR (Adaptive)": {
        "model_class": BPRAdaptiveRecommender,
        "is_implicit": True,
        "parameters": {
            "k": {"type": "slider", "label": "Latent Factors (k)", "min": 1, "max": 100, "default": 32},
            "iterations": {"type": "slider", "label": "Iterations", "min": 100, "max": 5000, "default": 200, "step": 100},
            "learning_rate": {"type": "slider", "label": "Learning Rate (alpha)", "min": 0.001, "max": 0.1, "default": 0.01, "step": 0.001, "format": "%.3f"},
            "lambda_reg": {"type": "slider", "label": "Regularization (lambda)", "min": 0.0, "max": 0.1, "default": 0.01, "step": 0.001, "format": "%.3f"},
            "negative_sample_pool_size": {"type": "slider", "label": "Adaptive Sample Pool", "min": 1, "max": 50, "default": 5, "step": 1, "help": "Size of the random pool to sample from to find the 'hardest' negative."}
        },
        "result_type": "bpr",
        "visualizer_class": BPRAdaptiveVisualizer,
        "visualization_renderer_class": BPRAdaptiveVisualizationRenderer
    },
    "ItemKNN": {
        "model_class": ItemKNNRecommender,
        "is_implicit": False,
        "parameters": {
            "k": {"type": "slider", "label": "Number of Neighbors (k)", "min": 5, "max": 100, "default": 20},
            "similarity_metric": {"type": "selectbox", "label": "Similarity Metric", "options": ["cosine", "adjusted_cosine", "pearson"], "default": "cosine"},
            "min_support": {"type": "slider", "label": "Minimum Support", "min": 0, "max": 10, "default": 2},
            "shrinkage": {"type": "slider", "label": "Shrinkage", "min": 0.0, "max": 100.0, "default": 0.0, "step": 1.0}
        },
        "result_type": "knn_similarity",
        "visualizer_class": ItemKNNVisualizer,
        "visualization_renderer_class": KNNVisualizationRenderer
    },
    "UserKNN": {
        "model_class": UserKNNRecommender,
        "is_implicit": False,
        "parameters": {
            "k": {"type": "slider", "label": "Number of Neighbors (k)", "min": 5, "max": 100, "default": 20},
            "similarity_metric": {"type": "selectbox", "label": "Similarity Metric", "options": ["cosine", "adjusted_cosine", "pearson"], "default": "cosine"}
        },
        "result_type": "knn_similarity",
        "visualizer_class": UserKNNVisualizer,
        "visualization_renderer_class": KNNVisualizationRenderer
    },
    "Slope One": {
        "model_class": SlopeOneRecommender,
        "is_implicit": False,
        "parameters": {},
        "result_type": "other",
        "visualizer_class": SlopeOneVisualizer,
        "visualization_renderer_class": KNNVisualizationRenderer
    },
    "NMF": {
        "model_class": NMFRecommender,
        "is_implicit": False,
        "parameters": {
            "k": {"type": "slider", "label": "Latent Factors (k)", "min": 1, "max": 100, "default": 32},
            "iterations": {"type": "slider", "label": "Iterations", "min": 1, "max": 100, "default": 50},
            "learning_rate": {"type": "slider", "label": "Learning Rate", "min": 0.001, "max": 0.1, "default": 0.005, "step": 0.001, "format": "%.3f"},
            "lambda_reg": {"type": "slider", "label": "Regularization (lambda)", "min": 0.0, "max": 1.0, "default": 0.02, "step": 0.01}
        },
        "result_type": "matrix_factorization",
        "visualizer_class": NMFVisualizer,
        "visualization_renderer_class": NMFVisualizationRenderer
    },
    "FunkSVD": {
        "model_class": FunkSVDRecommender,
        "is_implicit": False,
        "parameters": {
            "k": {"type": "slider", "label": "Latent Factors (k)", "min": 1, "max": 100, "default": 32},
            "iterations": {"type": "slider", "label": "Iterations", "min": 1, "max": 100, "default": 50},
            "learning_rate": {"type": "slider", "label": "Learning Rate", "min": 0.001, "max": 0.1, "default": 0.005, "step": 0.001},
            "lambda_reg": {"type": "slider", "label": "Regularization (lambda)", "min": 0.0, "max": 1.0, "default": 0.02, "step": 0.01}
        },
        "result_type": "matrix_factorization",
        "visualizer_class": FunkSVDVisualizer,
        "visualization_renderer_class": FunkSVDVisualizationRenderer
    },
    "PureSVD": {
        "model_class": PureSVDRecommender,
        "is_implicit": False,
        "parameters": {
            "k": {"type": "slider", "label": "Latent Factors (k)", "min": 1, "max": 100, "default": 32}
        },
        "result_type": "matrix_factorization",
        "visualizer_class": PureSVDVisualizer,
        "visualization_renderer_class": PureSVDVisualizationRenderer
    },
    "SVD++": {
        "model_class": SVDppRecommender,
        "is_implicit": False,
        "parameters": {
            "k": {"type": "slider", "label": "Latent Factors (k)", "min": 1, "max": 100, "default": 100},
            "iterations": {"type": "slider", "label": "Iterations", "min": 1, "max": 30, "default": 26},
            "learning_rate": {"type": "slider", "label": "Learning Rate", "min": 0.001, "max": 0.1, "default": 0.088375599, "step": 0.001, "format": "%.8f"},
            "lambda_p": {"type": "slider", "label": "Reg. P (User Factor)",
                         "min": 0.0, "max": 1.0, "default": 0.045728541, "step": 0.01, "format": "%.8f"},
            "lambda_q": {"type": "slider", "label": "Reg. Q (Item Factor)",
                         "min": 0.0, "max": 1.0, "default": 0.00997979, "step": 0.01, "format": "%.8f"},
            "lambda_y": {"type": "slider", "label": "Reg. Y (Implicit Factor)",
                         "min": 0.0, "max": 1.0, "default": 0.349586186, "step": 0.01, "format": "%.8f"},
            "lambda_bu": {"type": "slider", "label": "Reg. Bu (User Bias)",
                          "min": 0.0, "max": 20.0, "default": 8.233157889, "step": 0.5, "format": "%.8f"},
            "lambda_bi": {"type": "slider", "label": "Reg. Bi (Item Bias)",
                          "min": 0.0, "max": 20.0, "default": 12.93152091, "step": 0.5, "format": "%.8f"}
        },
        "result_type": "matrix_factorization",
        "visualizer_class": SVDppVisualizer,
        "visualization_renderer_class": SVDppVisualizationRenderer
    },
    "WRMF": {
        "model_class": WRMFRecommender,
        "is_implicit": True,
        "parameters": {
            "k": {"type": "slider", "label": "Latent Factors (k)", "min": 1, "max": 100, "default": 32},
            "iterations": {"type": "slider", "label": "Iterations", "min": 1, "max": 30, "default": 10},
            "lambda_reg": {"type": "slider", "label": "Regularization (lambda)", "min": 0.0, "max": 1.0, "default": 0.1, "step": 0.01},
            "alpha": {"type": "slider", "label": "Alpha", "min": 1, "max": 100, "default": 40}
        },
        "result_type": "wrmf",
        "visualizer_class": WRMFVisualizer,
        "visualization_renderer_class": WRMFVisualizationRenderer
    },
    "CML": {
        "model_class": CMLRecommender,
        "is_implicit": True,
        "parameters": {
            "k": {"type": "slider", "label": "Latent Factors (k)", "min": 1, "max": 100, "default": 32},
            "iterations": {"type": "slider", "label": "Iterations", "min": 1, "max": 200, "default": 100, "step": 10},
            "learning_rate": {"type": "slider", "label": "Learning Rate", "min": 0.001, "max": 0.1, "default": 0.01, "step": 0.001},
            "lambda_reg": {"type": "slider", "label": "Regularization (lambda)", "min": 0.0, "max": 1.0, "default": 0.01, "step": 0.001},
            "margin": {"type": "slider", "label": "Margin", "min": 0.1, "max": 2.0, "default": 0.5, "step": 0.1}
        },
        "result_type": "cml",
        "visualizer_class": CMLVisualizer,
        "visualization_renderer_class": CMLVisualizationRenderer
    },
    "NCFNeuMF": {
        "model_class": NCFRecommender,
        "is_implicit": True,
        "parameters": {
            "k": {"type": "slider", "label": "Latent Factors (k)", "min": 1, "max": 100, "default": 32},
            "model_type": {"type": "selectbox", "label": "Model Type", "options": ['NeuMF', 'GMF', 'NCF'], "default": 'NeuMF'},
            "epochs": {"type": "slider", "label": "Epochs", "min": 1, "max": 50, "default": 10},
            "batch_size": {"type": "select_slider", "label": "Batch Size", "options": [16, 32, 64, 128, 256], "default": 64},
            "learning_rate": {"type": "slider", "label": "Learning Rate", "min": 0.0001, "max": 0.01, "default": 0.001, "format": "%.4f"}
        },
        "result_type": "neural",
        "visualizer_class": NCFVisualizer,
        "visualization_renderer_class": NCFVisualizationRenderer
    },
    "SASRec": {
        "model_class": SASRecRecommender,
        "is_implicit": True,
        "parameters": {
            "k": {"type": "slider", "label": "Latent Factors (k)", "min": 1, "max": 100, "default": 32},
            "epochs": {"type": "slider", "label": "Epochs", "min": 5, "max": 100, "default": 30},
            "batch_size": {"type": "select_slider", "label": "Batch Size", "options": [32, 64, 128, 256], "default": 128},
            "learning_rate": {"type": "slider", "label": "Learning Rate", "min": 0.0001, "max": 0.01, "default": 0.001, "format": "%.4f"},
            "max_len": {"type": "slider", "label": "Max Sequence Length", "min": 10, "max": 200, "default": 50},
            "num_blocks": {"type": "slider", "label": "Number of Attention Blocks", "min": 1, "max": 4, "default": 2},
            "num_heads": {"type": "slider", "label": "Number of Attention Heads", "min": 1, "max": 4, "default": 1},
            "dropout_rate": {"type": "slider", "label": "Dropout Rate", "min": 0.0, "max": 0.5, "default": 0.2, "step": 0.05}
        },
        "result_type": "neural",
        "visualizer_class": SASRecVisualizer,
        "visualization_renderer_class": SASRecVisualizationRenderer
    },
    "VAE": {
        "model_class": VAERecommender,
        "is_implicit": True,
        "parameters": {
            "k": {"type": "slider", "label": "Latent Factors (k)", "min": 1, "max": 100, "default": 32},
            "epochs": {"type": "slider", "label": "Epochs", "min": 1, "max": 100, "default": 20},
            "batch_size": {"type": "select_slider", "label": "Batch Size", "options": [16, 32, 64, 128, 256], "default": 64},
            "learning_rate": {"type": "slider", "label": "Learning Rate", "min": 0.0001, "max": 0.01, "default": 0.001, "format": "%.4f"}
        },
        "result_type": "vae",
        "visualizer_class": VAEVisualizer,
        "visualization_renderer_class": VAEVisualizationRenderer
    },
    "SLIM": {
        "model_class": SLIMRecommender,
        "is_implicit": True,
        "parameters": {
            "l1_reg": {"type": "slider", "label": "L1 Regularization", "min": 0.0, "max": 0.1, "default": 0.001, "format": "%.4f"},
            "l2_reg": {"type": "slider", "label": "L2 Regularization", "min": 0.0, "max": 0.1, "default": 0.0001, "format": "%.5f"}
        },
        "info": "SLIM learns a sparse item-item similarity matrix. L1 encourages sparsity, L2 prevents large weights.",
        "result_type": "slim",
        "visualizer_class": SLIMVisualizer,
        "visualization_renderer_class": SLIMVisualizationRenderer
    },
    "FISM": {
        "model_class": FISMRecommender,
        "is_implicit": True,
        "parameters": {
            "k": {"type": "slider", "label": "Latent Factors (k)", "min": 1, "max": 100, "default": 32},
            "iterations": {"type": "slider", "label": "Iterations", "min": 10, "max": 200, "default": 100, "step": 10},
            "learning_rate": {"type": "slider", "label": "Learning Rate", "min": 0.001, "max": 0.1, "default": 0.01, "step": 0.001, "format": "%.3f"},
            "lambda_reg": {"type": "slider", "label": "Regularization (lambda)", "min": 0.0, "max": 0.1, "default": 0.01, "step": 0.001, "format": "%.3f"},
            "alpha": {"type": "slider", "label": "Alpha", "min": 0.0, "max": 1.0, "default": 0.5, "step": 0.1}
        },
        "result_type": "fism",
        "visualizer_class": FISMVisualizer,
        "visualization_renderer_class": FISMVisualizationRenderer
    },
}