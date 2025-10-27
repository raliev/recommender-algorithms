# Recommender System Laboratory

> **Interactive companion app for [Recommender Algorithms in 2026: A Practitioner's Guide](https://testmysearch.com/books/recommender-algorithms.html)**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Book](https://img.shields.io/badge/Book-Recommender%20Algorithms%20in%202026-orange.svg)](https://testmysearch.com/books/recommender-algorithms.html)

A comprehensive Streamlit application for experimenting with state-of-the-art recommendation algorithms. This interactive laboratory provides real-time training, visualization, and analysis of various recommender system algorithms, making it an ideal companion to the book.

![Book Cover](https://testmysearch.com/img/ra-ws.jpg)

## Key Features

- ** Interactive Algorithm Lab**: Train and compare 20+ recommendation algorithms with live visualization
- ** Real-time Visualizations**: Watch latent factor evolution, convergence curves, and recommendation breakdowns
- ** Hyperparameter Tuning**: Built-in Bayesian optimization with Optuna for automatic parameter search
- ** Performance Metrics**: Track RMSE, MAE, Precision@k, Recall@k, and more
- ** Educational Focus**: Perfect companion to understand algorithms from the book through hands-on experimentation

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/raliev/recommender-lab.git
cd recommender-lab
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

   Or using a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

## How to Use

This app has two main interfaces:

### 1. Lab Page - Interactive Experimentation

1. **Select Data Source**: Choose from synthetic data, MovieLens filtered subset, or full dataset
2. **Choose Algorithm**: Pick from 20+ available recommendation algorithms
3. **Adjust Hyperparameters**: Use sliders to tune algorithm parameters in real-time
4. **Enable Visualizations**: Toggle internals to see algorithm training in action
5. **Train Model**: Click "Train Model" to start training
6. **Explore Results**: View performance metrics, visualizations, and previous runs

### 2. Hyperparameter Tuning Page - Automated Optimization

Best hyperparameters for your algorithms - using Optuna

1. **Select Algorithms**: Choose which algorithms to tune
2. **Set Trial Count**: Configure the number of optimization trials
3. **Run Optimization**: Let Optuna automatically find the best hyperparameters
4. **Compare Results**: View performance comparisons across tuned configurations

## Table of Contents

- [Getting Started](#-getting-started)
- [How to Use](#-how-to-use)
- [Supported Algorithms](#-supported-algorithms)
- [Visualizations](#-visualizations)
- [About the Book](#-about-the-book)
- [Project Structure](#️-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

## Supported Algorithms

The list below is a portion of the algorithms explained in the book.

### Matrix Factorization Methods
- **SVD** - Singular Value Decomposition
- **FunkSVD** - Biased Matrix Factorization
- **PureSVD** - Eigenvalue decomposition approach
- **SVD++** - Factorization with implicit feedback
- **ALS** - Alternating Least Squares
- **ALS (Improved)** - Enhanced ALS with bias regularization
- **ALS (PySpark)** - Distributed implementation
- **NMF** - Non-negative Matrix Factorization
- **WRMF** - Weighted Regularized Matrix Factorization
- **CML** - Collaborative Metric Learning

### Neighborhood-Based Methods
- **ItemKNN** - Item-based Collaborative Filtering
- **UserKNN** - User-based Collaborative Filtering
- **Slope One** - Fast collaborative filtering

### Deep Learning Methods
- **BPR** - Bayesian Personalized Ranking
- **SLIM** - Sparse Linear Methods
- **FISM** - Factorized Item Similarity Models
- **NCF/NeuMF** - Neural Collaborative Filtering
- **SASRec** - Self-Attentive Sequential Recommendation
- **VAE** - Variational Autoencoders

## Visualizations

The app provides rich visualizations for understanding algorithm behavior:

- **Latent Factor Heatmaps**: See how user and item factors evolve during training
- **Convergence Graphs**: Track loss and objective function over iterations
- **Factor Histograms**: Analyze the distribution of latent factor values
- **Recommendation Breakdowns**: Understand how recommendations are generated
- **Similarity Matrices**: Visualize item-item and user-user relationships
- **Sparsity Patterns**: Explore matrix sparsity in learning-based methods

## About the Book

This application is a companion to **[Recommender Algorithms in 2026: A Practitioner's Guide](https://testmysearch.com/books/recommender-algorithms.html)** by Rauf Aliev.

The book provides:
- Mathematical foundations of each algorithm
- Implementation details and optimization techniques
- Production-ready architectures
- Deep dive into LLM-based and multimodal approaches
- Real-world applications and best practices

### Purchase Options

Available on:
-  [Amazon US](https://www.amazon.com/dp/B0FVGLS1ZK)
-  [Amazon UK](https://www.amazon.co.uk/dp/B0FVGK1H36)
-  [Other Amazon markets](https://testmysearch.com/books/recommender-algorithms.html)

## Project Structure

```
app/
├── algorithms/          # Implementation of recommender algorithms
├── visualization/       # Visualization components and renderers
├── pages/              # Streamlit pages (Lab, Hyperparameter Tuning)
├── datasets/           # Sample datasets
├── visualizations_info/ # Algorithm visualization documentation
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Configuration

- **Algorithm Parameters**: Configure in `algorithm_config.py`
- **Tuner Settings**: Adjust in `tuner_config.py`
- **Visualization Settings**: Modify in `visualization/` components

## Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

Contributions can include:
-  Reporting bugs
-  Suggesting features
-  Submitting pull requests
-  Improving documentation
-  Adding new algorithms
-  Enhancing visualizations

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- Built as a companion to the "Recommender Algorithms in 2026" book
- Uses [Streamlit](https://streamlit.io) for the interactive interface
- Employs [Optuna](https://optuna.org) for hyperparameter optimization
- Visualizations powered by [Plotly](https://plotly.com) and [Matplotlib](https://matplotlib.org)

## Contact

For questions or support related to the book or this application:
- Book Website: [testmysearch.com/books](https://testmysearch.com/books/recommender-algorithms.html)
- Author: Rauf Aliev r.aliev@gmail.com [https://testmysearch.com/raufaliev](https://testmysearch.com/raufaliev)

**Happy Recommending! 🎉**

