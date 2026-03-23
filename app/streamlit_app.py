"""
Streamlit Web Application for the Tourism Recommendation System.
Provides an interactive interface for getting personalized destination recommendations,
exploring destinations, and viewing model performance analytics.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from models.hybrid import HybridRecommender
from utils.config import (
    DATA_DIR, MODEL_DIR, PLOTS_DIR,
    EMBEDDING_DIM, GMF_DIM, MLP_HIDDEN_LAYERS,
    USER_FEATURE_DIM, DEST_FEATURE_DIM,
)
from utils.preprocess import DataPreprocessor


# ─────────────────────────── Page Config ───────────────────────────
st.set_page_config(
    page_title="🌍 Tourism Recommendation System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Custom CSS ───────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .dest-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        transition: transform 0.2s;
    }
    .dest-card:hover {
        transform: translateY(-2px);
    }
    .score-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    .category-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Category Colors ───────────────────────────
CATEGORY_COLORS = {
    "beach": "#00b4d8",
    "mountain": "#2d6a4f",
    "city": "#e63946",
    "historical": "#d4a373",
    "nature": "#52b788",
    "island": "#0077b6",
    "desert": "#e9c46a",
}

CATEGORY_ICONS = {
    "beach": "🏖️",
    "mountain": "⛰️",
    "city": "🏙️",
    "historical": "🏛️",
    "nature": "🌿",
    "island": "🏝️",
    "desert": "🏜️",
}


# ─────────────────────────── Load Data & Model ───────────────────────────

@st.cache_data
def load_datasets():
    """Load the tourism datasets."""
    try:
        users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
        destinations = pd.read_csv(os.path.join(DATA_DIR, "destinations.csv"))
        interactions = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
        return users, destinations, interactions
    except FileNotFoundError:
        return None, None, None


@st.cache_resource
def load_model(num_users, num_dests):
    """Load the trained hybrid model."""
    model_path = os.path.join(MODEL_DIR, "hybrid_best.pt")
    if not os.path.exists(model_path):
        return None

    model = HybridRecommender(
        num_users=num_users,
        num_destinations=num_dests,
        embedding_dim=EMBEDDING_DIM,
        gmf_dim=GMF_DIM,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


@st.cache_resource
def get_preprocessor():
    """Get a fitted preprocessor."""
    preprocessor = DataPreprocessor()
    users, destinations, _ = load_datasets()
    if users is not None:
        preprocessor.encode_features(users, destinations)
    return preprocessor


def get_recommendations(model, user_profile, destinations, preprocessor, top_k=10):
    """Generate top-K recommendations for a user profile."""
    # Encode user features
    travel_styles = ["adventure", "cultural", "family", "luxury", "relaxation"]
    budget_levels = ["budget", "luxury", "moderate", "premium"]
    climates = ["arid", "cold", "mediterranean", "temperate", "tropical"]
    seasons = ["autumn", "spring", "summer", "winter"]
    categories = ["beach", "city", "desert", "historical", "island", "mountain", "nature"]

    ts_enc = travel_styles.index(user_profile["travel_style"]) if user_profile["travel_style"] in travel_styles else 0
    bl_enc = budget_levels.index(user_profile["budget_level"]) if user_profile["budget_level"] in budget_levels else 0

    user_feat = np.array([[ts_enc, bl_enc, user_profile["age"], user_profile["past_trips"]]], dtype=np.float32)
    user_feat = preprocessor.user_scaler.transform(user_feat)
    user_feat_tensor = torch.FloatTensor(user_feat)

    scores = []
    with torch.no_grad():
        for _, dest in destinations.iterrows():
            dest_id = dest["dest_id"]

            # Encode destination features
            cat_enc = categories.index(dest["category"]) if dest["category"] in categories else 0
            clim_enc = climates.index(dest["climate"]) if dest["climate"] in climates else 0
            seas_enc = seasons.index(dest["best_season"]) if dest["best_season"] in seasons else 0

            dest_feat = np.array([[cat_enc, clim_enc, seas_enc, dest["avg_cost"], dest["popularity_score"]]], dtype=np.float32)
            dest_feat = preprocessor.dest_scaler.transform(dest_feat)
            dest_feat_tensor = torch.FloatTensor(dest_feat)

            # Use a fixed user_id (0) for new user profiles
            user_id_tensor = torch.LongTensor([0])
            dest_id_tensor = torch.LongTensor([dest_id])

            pred = model(user_id_tensor, dest_id_tensor, user_feat_tensor, dest_feat_tensor)
            scores.append(pred.item())

    destinations = destinations.copy()
    destinations["predicted_score"] = scores
    recommendations = destinations.nlargest(top_k, "predicted_score")
    return recommendations


# ─────────────────────────── Main App ───────────────────────────

def main():
    # Header
    st.markdown('<p class="main-header">🌍 Tourism Recommendation System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Personalized Travel Recommendations Using Deep Learning</p>', unsafe_allow_html=True)

    # Load data
    users, destinations, interactions = load_datasets()

    if users is None:
        st.error("⚠️ Dataset not found! Run `python main.py generate-data` first.")
        return

    # Sidebar - User Profile
    st.sidebar.markdown("## 👤 Your Travel Profile")
    st.sidebar.markdown("---")

    age = st.sidebar.slider("🎂 Age", 18, 70, 28)
    travel_style = st.sidebar.selectbox(
        "✈️ Travel Style",
        ["adventure", "cultural", "relaxation", "family", "luxury"],
        index=0,
    )
    budget_level = st.sidebar.selectbox(
        "💰 Budget Level",
        ["budget", "moderate", "premium", "luxury"],
        index=1,
    )
    past_trips = st.sidebar.slider("🗺️ Past Trips", 0, 30, 5)
    preferred_climate = st.sidebar.selectbox(
        "🌤️ Preferred Climate",
        ["any", "tropical", "temperate", "arid", "cold", "mediterranean"],
        index=0,
    )
    top_k = st.sidebar.slider("📊 Number of Recommendations", 5, 20, 10)

    user_profile = {
        "age": age,
        "travel_style": travel_style,
        "budget_level": budget_level,
        "past_trips": past_trips,
    }

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Recommendations", "📊 Analytics", "🔍 Explore"])

    # ─── Tab 1: Recommendations ───
    with tab1:
        model = load_model(len(users), len(destinations))
        if model is None:
            st.warning("⚠️ No trained model found. Run `python main.py train` first.")
            st.info("After training, refresh this page to get personalized recommendations.")
        else:
            preprocessor = get_preprocessor()

            with st.spinner("🔮 Generating recommendations..."):
                recs = get_recommendations(model, user_profile, destinations, preprocessor, top_k)

            # Filter by climate if specified
            if preferred_climate != "any":
                climate_recs = recs[recs["climate"] == preferred_climate]
                if len(climate_recs) >= 3:
                    recs = climate_recs

            st.markdown(f"### 🎯 Top {len(recs)} Recommendations for You")
            st.markdown(f"_Based on your profile: **{travel_style.title()}** traveler, "
                        f"**{budget_level.title()}** budget, age **{age}**_")
            st.markdown("---")

            # Display recommendations in grid
            for i in range(0, len(recs), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(recs):
                        dest = recs.iloc[idx]
                        cat = dest["category"]
                        icon = CATEGORY_ICONS.get(cat, "📍")
                        color = CATEGORY_COLORS.get(cat, "#666")
                        score = dest["predicted_score"]

                        with col:
                            st.markdown(f"""
                            <div class="dest-card">
                                <h3>{icon} {dest['name']}</h3>
                                <p>📍 {dest['country']} &nbsp;│&nbsp;
                                <span class="category-tag" style="background:{color}20; color:{color};">
                                    {cat.title()}
                                </span></p>
                                <p>🌤️ {dest['climate'].title()} &nbsp;│&nbsp; 
                                   📅 Best in {dest['best_season'].title()} &nbsp;│&nbsp;
                                   💰 ${dest['avg_cost']:,.0f}</p>
                                <p>⭐ Popularity: {dest['popularity_score']:.0%}</p>
                                <span class="score-badge">Match Score: {score:.2f}/5.0</span>
                            </div>
                            """, unsafe_allow_html=True)

    # ─── Tab 2: Analytics ───
    with tab2:
        st.markdown("### 📊 Model Performance Analytics")
        st.markdown("---")

        # Dataset overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👥 Total Users", f"{len(users):,}")
        with col2:
            st.metric("🏖️ Destinations", f"{len(destinations):,}")
        with col3:
            st.metric("📝 Interactions", f"{len(interactions):,}")

        st.markdown("---")

        # Show saved plots if they exist
        plot_files = {
            "Training Curves": "training_curves.png",
            "Model Comparison": "model_comparison.png",
            "Prediction Distribution": "prediction_dist.png",
        }

        plots_found = False
        for title, filename in plot_files.items():
            path = os.path.join(PLOTS_DIR, filename)
            if os.path.exists(path):
                plots_found = True
                st.markdown(f"#### {title}")
                st.image(path)
                st.markdown("---")

        if not plots_found:
            st.info("📈 No evaluation plots found. Run `python main.py evaluate` to generate them.")

        # Rating distribution
        st.markdown("#### Rating Distribution in Dataset")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(interactions["rating"], bins=20, color="#667eea", alpha=0.7, edgecolor="white")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Ratings", fontweight="bold")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

        # Travel style distribution
        st.markdown("#### User Travel Style Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        style_counts = users["travel_style"].value_counts()
        colors_list = ["#667eea", "#764ba2", "#f093fb", "#4facfe", "#43e97b"]
        ax.bar(style_counts.index, style_counts.values, color=colors_list)
        ax.set_xlabel("Travel Style")
        ax.set_ylabel("Count")
        ax.set_title("Travel Styles", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close()

    # ─── Tab 3: Explore ───
    with tab3:
        st.markdown("### 🔍 Explore Destinations")
        st.markdown("---")

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            cat_filter = st.multiselect(
                "Category", destinations["category"].unique().tolist(),
                default=destinations["category"].unique().tolist(),
            )
        with col2:
            climate_filter = st.multiselect(
                "Climate", destinations["climate"].unique().tolist(),
                default=destinations["climate"].unique().tolist(),
            )
        with col3:
            cost_range = st.slider(
                "Cost Range ($)",
                int(destinations["avg_cost"].min()),
                int(destinations["avg_cost"].max()),
                (int(destinations["avg_cost"].min()), int(destinations["avg_cost"].max())),
            )

        filtered = destinations[
            (destinations["category"].isin(cat_filter)) &
            (destinations["climate"].isin(climate_filter)) &
            (destinations["avg_cost"].between(cost_range[0], cost_range[1]))
        ]

        st.markdown(f"**Showing {len(filtered)} destinations**")

        # Display as styled table
        display_df = filtered[["name", "country", "category", "climate", "avg_cost", "popularity_score", "best_season"]].copy()
        display_df.columns = ["Name", "Country", "Category", "Climate", "Avg Cost ($)", "Popularity", "Best Season"]
        display_df["Avg Cost ($)"] = display_df["Avg Cost ($)"].round(0).astype(int)
        display_df["Popularity"] = (display_df["Popularity"] * 100).round(1).astype(str) + "%"

        st.dataframe(display_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
