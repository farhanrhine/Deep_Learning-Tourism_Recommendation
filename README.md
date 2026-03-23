# 🌍 Tourism Recommendation System Using Deep Learning

A personalized tourism recommendation system that uses hybrid deep learning to predict user preferences and recommend travel destinations. Built with PyTorch and Streamlit.

## 📋 Objective

Design a personalized tourism recommendation system using deep learning that:
- Integrates multiple data sources (user profiles, destination attributes, interaction history)
- Predicts user preferences accurately using neural collaborative filtering + content-based models
- Provides adaptive recommendations through a learnable attention fusion mechanism
- Demonstrates AI-driven solutions for enhancing travel planning experiences

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Hybrid Recommender                  │
│                                                      │
│  ┌──────────────────┐    ┌──────────────────┐       │
│  │   NCF Branch     │    │  Content Branch  │       │
│  │  (Collaborative) │    │  (Attribute)     │       │
│  │                  │    │                  │       │
│  │ User Embedding   │    │ User Feature MLP │       │
│  │ Dest Embedding   │    │ Dest Feature MLP │       │
│  │ GMF + MLP Path   │    │ Concat + Predict │       │
│  └────────┬─────────┘    └────────┬─────────┘       │
│           │                       │                  │
│           └───────┬───────────────┘                  │
│                   │                                  │
│         ┌─────────▼──────────┐                      │
│         │  Attention Gate    │                      │
│         │  (Learned Weights) │                      │
│         └─────────┬──────────┘                      │
│                   │                                  │
│         ┌─────────▼──────────┐                      │
│         │  Fusion Layer      │                      │
│         │  → Rating (1-5)    │                      │
│         └────────────────────┘                      │
└─────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Web Interface | Streamlit |
| Visualization | Matplotlib, Seaborn |
| Package Manager | uv |

## 📁 Project Structure

```
Tourism-Recommendation/
├── main.py                    # CLI entry point
├── pyproject.toml             # Dependencies (managed by uv)
├── data/
│   ├── generate_dataset.py    # Synthetic dataset generator
│   ├── users.csv              # Generated user profiles
│   ├── destinations.csv       # Generated destinations
│   └── interactions.csv       # Generated user-destination interactions
├── models/
│   ├── ncf.py                 # Neural Collaborative Filtering
│   ├── content_model.py       # Content-Based Feature Extractor
│   └── hybrid.py              # Hybrid Recommender (NCF + Content)
├── training/
│   ├── train.py               # Training pipeline with early stopping
│   └── evaluate.py            # Evaluation metrics & visualization
├── utils/
│   ├── config.py              # Central configuration
│   ├── logger.py              # Custom logging
│   ├── custom_exception.py    # Custom exception hierarchy
│   └── preprocess.py          # Data preprocessing & feature engineering
├── app/
│   └── streamlit_app.py       # Streamlit web interface
├── saved_models/              # Model checkpoints
├── plots/                     # Evaluation plots
└── logs/                      # Training logs
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
uv sync
```

### 2. Generate Dataset
```bash
uv run python main.py generate-data
```

### 3. Train Models
```bash
# Full training (50 epochs)
uv run python main.py train

# Quick training
uv run python main.py train --epochs 10
```

### 4. Evaluate Models
```bash
uv run python main.py evaluate
```

### 5. Launch Web App
```bash
uv run python main.py run-app
```

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Squared Error (rating prediction accuracy) |
| MAE | Mean Absolute Error |
| Precision@K | Fraction of relevant items in top-K |
| Recall@K | Fraction of relevant items found |
| NDCG@K | Normalized Discounted Cumulative Gain (ranking quality) |
| Hit Rate@K | At least one relevant item in top-K |

## 📦 Models

### Neural Collaborative Filtering (NCF)
Learns latent user-destination interaction patterns through dual GMF + MLP pathways (NeuMF architecture).

### Content-Based Model
Uses explicit user attributes (age, travel style, budget) and destination features (category, climate, cost) for predictions.

### Hybrid Recommender
Combines NCF and Content-Based models via a learnable attention gate that dynamically weights collaborative vs. content signals.

## 🔧 Skills Used
- **Python** — Core language
- **Deep Learning** — PyTorch neural networks (NCF, MLP, Attention)
- **Recommendation Systems** — Collaborative filtering, content-based, hybrid fusion
- **Data Analytics** — Feature engineering, data preprocessing, statistical analysis
- **User Behavior Analysis** — Preference modeling, interaction pattern learning