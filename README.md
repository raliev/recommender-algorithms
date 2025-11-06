# Recommender System Laboratory

> **Interactive companion application for [Recommender Algorithms in 2026: A Practitioner's Guide](https://testmysearch.com/books/recommender-algorithms.html)**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Book](https://img.shields.io/badge/Book-Recommender%20Algorithms%20in%202026-orange.svg)](https://testmysearch.com/books/recommender-algorithms.html)

A comprehensive Streamlit application for experimenting with state-of-the-art recommendation algorithms. This interactive laboratory provides real-time training, visualization, and analysis of various recommender system algorithms, making it an ideal companion to the book.

![Book Cover](https://testmysearch.com/img/ra-ws.jpg)

## Key Features

* **Interactive Algorithm Lab**: Train and compare over 25 recommendation algorithms in real-time.
* **Real-time Visualizations**: Observe latent factor evolution, convergence curves, and recommendation breakdowns as models train.
* **Hyperparameter Tuning**: Utilize Bayesian optimization (Optuna) for automated parameter search.
* **Dataset Generation Wizard**: Create, configure, and save synthetic datasets based on ground-truth user preferences (P) and item features (Q).
* **Performance Metrics**: Track RMSE, MAE, Precision@k, Recall@k, and other standard evaluation metrics.
* **Report & Run Viewing**: Review and compare results from past hyperparameter tuning runs and individual lab experiments.

You can automatically generate a synthetic dataset that adheres to specific logic based on underlying item properties. For example, if you select the 'recipes_and_tastes' ground-truth template, the system utilizes a predefined set of recipes, each associated with one or more of 20 latent tastes. The wizard then generates users with randomized taste preferences (a ground-truth P matrix) and, based on the item-taste associations (a ground-truth Q matrix), calculates ideal user-recipe ratings. 

This generation process is highly configurable, allowing you to control parameters ranging from the number of tastes an average user possesses and the contrast of their preferences to the level of random noise and sparsity applied to the final output dataset. 

This generated dataset can then be used to train and test the algorithms. The recommender model will only receive the final ratings or interactions as input; the underlying characteristics (the 'tastes') are withheld. However, these ground-truth preferences and features are used within the 'Lab' page to provide a deeper analysis of the recommendation quality, supplementing the standard train/test metrics by showing why a recommendation was or was not correct based on the user's known interests.

## Getting Started

### Prerequisites

* Python 3.8 or higher
* pip package manager

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/recommender-lab.git
    cd recommender-lab
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

    Or using a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  Run the application:
    ```bash
    streamlit run app.py
    ```

    The application will automatically open in your default browser at `http://localhost:8501`.

## How to Use

This application is divided into several key modules, accessible from the sidebar:

1.  **Lab Page**: Interactively experiment with a single algorithm. Adjust hyperparameters, enable visualizations, and train models in real-time.
2.  **Hyperparameter Tuning Page**: Configure and run automated Bayesian optimization (Optuna) for one or more algorithms to find the best-performing configurations.
3.  **Dataset Wizard Page**: Design and generate synthetic datasets. Define user-preference (P) and item-feature (Q) matrices, apply noise and sparsity, and save the resulting dataset for use in the Lab and Tuner.
4.  **Report Viewer Page**: Load and analyze the results from previous Hyperparameter Tuning runs.
5.  **All Recent Lab Runs Page**: Browse the history and view visualizations from individual experiments conducted on the Lab page.

## Table of Contents

* [Getting Started](#getting-started)
* [How to Use](#how-to-use)
* [Supported Algorithms](#supported-algorithms)
* [Visualizations](#visualizations)
* [About the Book](#about-the-book)
* [Project Structure](#project-structure)
* [Configuration](#configuration)
* [Contributing](#contributing)
* [License](#license)

## Supported Algorithms

### Matrix Factorization Methods
* **SVD** - Singular Value Decomposition
* **PureSVD** - SVD decomposition for implicit feedback
* **FunkSVD** - Stochastic Gradient Descent MF
* **SVD++** - Factorization with implicit feedback
* **ASVD** - Asymmetric SVD (no explicit user factors)
* **ALS** - Alternating Least Squares
* **ALS (Improved)** - ALS with bias regularization
* **ALS (PySpark)** - Distributed implementation
* **NMF** - Non-negative Matrix Factorization
* **WRMF** - Weighted Regularized Matrix Factorization
* **FISM** - Factorized Item Similarity Models

### Neighborhood-Based Methods
* **ItemKNN** - Item-based Collaborative Filtering
* **UserKNN** - User-based Collaborative Filtering
* **Slope One** - Fast, simple collaborative filtering

### Ranking & Metric Learning
* **BPR** - Bayesian Personalized Ranking
* **BPR (Adaptive)** - BPR with adaptive negative sampling
* **WMFBPR** - Weighted MF with BPR (uses PageRank)
* **CML** - Collaborative Metric Learning
* **SLIM** - Sparse Linear Methods

### Deep Learning & Sequential Methods
* **NCF/NeuMF** - Neural Collaborative Filtering
* **SASRec** - Self-Attentive Sequential Recommendation
* **VAE** - Variational Autoencoders for Collaborative Filtering

### Association Rule Mining
* **Apriori** - Classic association rule mining
* **FP-Growth** - Efficient tree-based rule mining
* **Eclat** - Vertical layout-based rule mining

### Baseline Methods
* **Top Popular** - Non-personalized popularity baseline

## Visualizations

The application provides rich visualizations for understanding algorithm behavior:

* **Convergence Plots**: Track objective functions (e.g., RMSE, Loss) and factor change norms over iterations.
* **Latent Factor Snapshots**: Visualize user (P) and item (Q) factor matrices via heatmaps, histograms, and 2D projections (for k=2).
* **Similarity/Sparsity Matrices**: Analyze learned item-item (SLIM, ItemKNN) or user-user (UserKNN) relationships.
* **Recommendation Breakdowns**: Deconstruct the score generation process for a single sample user.
* **Association Rules**: Review frequent itemsets and generated rules (support, confidence, lift) for Apriori/FP-Growth.

## About the Book

This application is a companion to **[Recommender Algorithms in 2026: A Practitioner's Guide](https://testmysearch.com/books/recommender-algorithms.html)** by Rauf Aliev.
The book provides:
* Mathematical foundations of each algorithm
* Implementation details and optimization techniques
* Production-ready architectures
* Deep dive into LLM-based and multimodal approaches
* Real-world applications and best practices

### Purchase Options

Available on:
-  [Amazon US](https://www.amazon.com/dp/B0FVGLS1ZK)
-  [Amazon UK](https://www.amazon.co.uk/dp/B0FVGK1H36)
-  [Amazon Germany](https://www.amazon.de/dp/B0FVGK1H36)
-  [Amazon Italy](https://www.amazon.it/dp/B0FVGK1H36)
-  [Amazon Spain](https://www.amazon.es/dp/B0FVGK1H36)
-  [Amazon Poland](https://www.amazon.pl/dp/B0FVGK1H36)
-  [Amazon Canada](https://www.amazon.ca/dp/B0FVGK1H36)
-  [Amazon Australia](https://www.amazon.au/dp/B0FVGK1H36)
-  [Amazon India](https://www.amazon.in/Recommender-Algorithms-Rauf-Aliev/dp/B0FVTDRGSW)

Read sample: [https://testmysearch.com/pdfs/books/recommenders-sample.pdf](https://testmysearch.com/pdfs/books/recommenders-sample.pdf)

-  [Pay with Paypal and get a PDF for $9.99](https://testmysearch.com/books/recommender-algorithms.html)


## Configuration

* **Algorithm Parameters**: Configure in `algorithm_config.py`
* **Tuner Settings**: Adjust in `tuner_config.py`
* **Visualization Settings**: Modify in `visualization/` components

## Contributing

We welcome contributions. See our [Contributing Guide](CONTRIBUTING.md) for details.

Contributions can include:
* Reporting bugs
* Suggesting features
* Submitting pull requests
* Improving documentation
* Adding new algorithms
* Enhancing visualizations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
## Acknowledgments

* Built as a companion to the "Recommender Algorithms in 2026" book
* Uses [Streamlit](https://streamlit.io) for the interactive interface
* Employs [Optuna](https://optuna.org) for hyperparameter optimization
* Visualizations powered by [Plotly](https://plotly.com) and [Matplotlib](https://matplotlib.org)

## Contact

For questions or support related to the book or this application:
* Book Website: [testmysearch.com/books](https://testmysearch.com/books/recommender-algorithms.html)
* Author: Rauf Aliev

## Screenshots

![screenshots/scr1.png](screenshots/scr1.png)
![screenshots/scr2.png](screenshots/scr2.png)
![screenshots/scr3.png](screenshots/scr3.png)
![screenshots/scr4.png](screenshots/scr4.png)
![screenshots/scr5.png](screenshots/scr5.png)
![screenshots/scr6.png](screenshots/scr6.png)
![screenshots/scr7.png](screenshots/scr7.png)
![screenshots/scr8.png](screenshots/scr8.png)
![screenshots/scr9.png](screenshots/scr9.png)
![screenshots/scrA.png](screenshots/scrA.png)