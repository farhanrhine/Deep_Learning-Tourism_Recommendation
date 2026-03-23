"""
Synthetic Tourism Dataset Generator.
Creates realistic user profiles, destinations, and user-destination interactions
with biased ratings that reflect real-world travel preferences.
"""

import os
import numpy as np
import pandas as pd
from utils.logger import get_project_logger
from utils.custom_exception import DataGenerationError
from utils.config import (
    DATA_DIR, NUM_USERS, NUM_DESTINATIONS, NUM_INTERACTIONS, RANDOM_SEED
)

logger = get_project_logger(__name__)

# ─────────────────────────── Constants ───────────────────────────
TRAVEL_STYLES = ["adventure", "cultural", "relaxation", "family", "luxury"]
BUDGET_LEVELS = ["budget", "moderate", "premium", "luxury"]
NATIONALITIES = [
    "American", "British", "German", "French", "Japanese",
    "Chinese", "Indian", "Australian", "Brazilian", "Canadian",
    "Italian", "Spanish", "Korean", "Thai", "Mexican",
]
DEST_CATEGORIES = ["beach", "mountain", "city", "historical", "nature", "island", "desert"]
CLIMATES = ["tropical", "temperate", "arid", "cold", "mediterranean"]
SEASONS = ["spring", "summer", "autumn", "winter"]
COUNTRIES = [
    "France", "Italy", "Spain", "Japan", "Thailand", "USA", "Australia",
    "Brazil", "India", "Greece", "Turkey", "Mexico", "Switzerland",
    "New Zealand", "Iceland", "Morocco", "Peru", "Vietnam", "Portugal", "Croatia",
]
DESTINATION_NAMES = [
    "Paris", "Rome", "Barcelona", "Tokyo", "Bangkok", "New York", "Sydney",
    "Rio de Janeiro", "Goa", "Santorini", "Istanbul", "Cancun", "Zurich",
    "Queenstown", "Reykjavik", "Marrakech", "Cusco", "Hanoi", "Lisbon", "Dubrovnik",
    "Bali", "Maldives", "Kyoto", "Venice", "Prague", "Dubai", "Cape Town",
    "Machu Picchu", "Petra", "Angkor Wat", "Great Barrier Reef", "Serengeti",
    "Fiji Islands", "Amalfi Coast", "Phuket", "Havana", "Cairo", "Amsterdam",
    "Vienna", "Munich", "Edinburgh", "Seville", "Osaka", "Seoul", "Singapore",
    "Banff", "Patagonia", "Sahara Oasis", "Zanzibar", "Bora Bora",
    "Lake Como", "Cinque Terre", "Cappadocia", "Ha Long Bay", "Luang Prabang",
    "Jaipur", "Udaipur", "Kerala Backwaters", "Ladakh", "Rishikesh",
    "Mount Fuji", "Nikko", "Nara", "Hiroshima", "Okinawa",
    "Yosemite", "Grand Canyon", "Yellowstone", "Niagara Falls", "Hawaii",
    "Galápagos Islands", "Easter Island", "Torres del Paine", "Iguazu Falls", "Amazon Rainforest",
    "Swiss Alps", "Dolomites", "Norwegian Fjords", "Scottish Highlands", "Black Forest",
    "Tulum", "Cartagena", "Medellín", "Buenos Aires", "Montevideo",
    "Kruger Park", "Victoria Falls", "Kilimanjaro", "Madagascar", "Seychelles",
    "Lofoten Islands", "Faroe Islands", "Azores", "Madeira", "Canary Islands",
    "Chiang Mai", "Siem Reap", "Yogyakarta", "Borneo", "Komodo Island",
    "Blue Lagoon", "Ring Road", "Golden Circle", "Tromsø", "Lapland",
    "Dubrovnik", "Kotor", "Split", "Mostar", "Ohrid",
    "Taj Mahal", "Hampi", "Varanasi", "Darjeeling", "Andaman Islands",
    "Great Wall Trek", "Zhangjiajie", "Guilin", "Xi'an", "Lijiang",
    "Uluru", "Tasmania", "Melbourne", "Perth", "Cairns",
    "Rotorua", "Milford Sound", "Abel Tasman", "Hobbiton", "Tongariro",
    "Bruges", "Luxembourg", "Monaco", "San Marino", "Liechtenstein",
    "Doha", "Oman", "Jordan", "Lebanon", "Georgia",
    "Tbilisi", "Batumi", "Yerevan", "Baku", "Samarkand",
    "Kathmandu", "Pokhara", "Bhutan", "Sri Lanka", "Maldives Reef",
    "Tahiti", "Samoa", "Tonga", "Vanuatu", "New Caledonia",
    "Dakar", "Fez", "Tunis", "Luxor", "Nairobi",
    "Petra Caves", "Dead Sea", "Wadi Rum", "Tel Aviv", "Jerusalem",
    "Greenland", "Svalbard", "Antarctica", "South Georgia", "Falkland Islands",
    "Havana Nights", "Trinidad", "Jamaica", "Barbados", "St. Lucia",
    "Bermuda", "Bahamas", "Turks and Caicos", "Aruba", "Curaçao",
    "Alaska", "Canadian Rockies", "Churchill", "Whitehorse", "Dawson City",
    "Provence", "Bordeaux", "Loire Valley", "Normandy", "Champagne",
    "Tuscany", "Sardinia", "Sicily", "Lake Garda", "Bologna",
    "Andalusia", "Basque Country", "Mallorca", "Ibiza", "Tenerife",
]


# ─────────────── Preference Affinity Matrix ───────────────
# Defines how much each travel style likes each destination category
# Higher = stronger preference
STYLE_CATEGORY_AFFINITY = {
    "adventure":  {"beach": 0.5, "mountain": 0.9, "city": 0.3, "historical": 0.4, "nature": 0.9, "island": 0.6, "desert": 0.8},
    "cultural":   {"beach": 0.3, "mountain": 0.4, "city": 0.8, "historical": 0.9, "nature": 0.5, "island": 0.3, "desert": 0.5},
    "relaxation": {"beach": 0.9, "mountain": 0.5, "city": 0.3, "historical": 0.3, "nature": 0.7, "island": 0.9, "desert": 0.2},
    "family":     {"beach": 0.8, "mountain": 0.5, "city": 0.7, "historical": 0.6, "nature": 0.7, "island": 0.7, "desert": 0.3},
    "luxury":     {"beach": 0.7, "mountain": 0.4, "city": 0.8, "historical": 0.5, "nature": 0.4, "island": 0.8, "desert": 0.3},
}


def generate_users(n: int = NUM_USERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate synthetic user profiles."""
    logger.info(f"Generating {n} user profiles...")
    rng = np.random.RandomState(seed)

    users = pd.DataFrame({
        "user_id": range(n),
        "age": rng.randint(18, 70, size=n),
        "nationality": rng.choice(NATIONALITIES, size=n),
        "travel_style": rng.choice(TRAVEL_STYLES, size=n),
        "budget_level": rng.choice(BUDGET_LEVELS, size=n),
        "past_trips": rng.poisson(lam=5, size=n),
    })

    logger.info(f"  ✓ Generated {len(users)} users")
    return users


def generate_destinations(n: int = NUM_DESTINATIONS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate synthetic destination profiles."""
    logger.info(f"Generating {n} destination profiles...")
    rng = np.random.RandomState(seed + 1)

    # Pick unique destination names
    chosen_names = rng.choice(DESTINATION_NAMES, size=n, replace=False)
    chosen_countries = rng.choice(COUNTRIES, size=n)

    destinations = pd.DataFrame({
        "dest_id": range(n),
        "name": chosen_names,
        "country": chosen_countries,
        "category": rng.choice(DEST_CATEGORIES, size=n),
        "avg_cost": rng.uniform(500, 5000, size=n).round(2),
        "popularity_score": rng.uniform(0.1, 1.0, size=n).round(3),
        "climate": rng.choice(CLIMATES, size=n),
        "best_season": rng.choice(SEASONS, size=n),
    })

    logger.info(f"  ✓ Generated {len(destinations)} destinations")
    return destinations


def generate_interactions(
    users: pd.DataFrame,
    destinations: pd.DataFrame,
    n: int = NUM_INTERACTIONS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate synthetic user-destination interactions with realistic rating bias.
    Ratings are influenced by the affinity between user travel style and destination category.
    """
    logger.info(f"Generating {n} interactions with preference-biased ratings...")
    rng = np.random.RandomState(seed + 2)

    user_ids = rng.choice(users["user_id"].values, size=n)
    dest_ids = rng.choice(destinations["dest_id"].values, size=n)

    ratings = []
    for uid, did in zip(user_ids, dest_ids):
        user_style = users.loc[users["user_id"] == uid, "travel_style"].values[0]
        dest_cat = destinations.loc[destinations["dest_id"] == did, "category"].values[0]

        # Base rating from affinity
        affinity = STYLE_CATEGORY_AFFINITY[user_style][dest_cat]
        base_rating = 1.0 + affinity * 4.0  # Scale to 1-5

        # Add noise
        noise = rng.normal(0, 0.5)
        rating = np.clip(base_rating + noise, 1.0, 5.0)
        ratings.append(round(rating, 1))

    interactions = pd.DataFrame({
        "user_id": user_ids,
        "dest_id": dest_ids,
        "rating": ratings,
        "visit_duration": rng.randint(1, 15, size=n),
        "season_visited": rng.choice(SEASONS, size=n),
        "would_revisit": rng.choice([0, 1], size=n, p=[0.35, 0.65]),
    })

    logger.info(f"  ✓ Generated {len(interactions)} interactions")
    logger.info(f"  ✓ Rating distribution: mean={interactions['rating'].mean():.2f}, "
                f"std={interactions['rating'].std():.2f}")
    return interactions


def generate_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate and save the complete dataset.
    
    Returns:
        Tuple of (users, destinations, interactions) DataFrames.
    
    Raises:
        DataGenerationError: If generation fails.
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting dataset generation...")
        logger.info("=" * 60)

        users = generate_users()
        destinations = generate_destinations()
        interactions = generate_interactions(users, destinations)

        # Save to CSV
        users.to_csv(os.path.join(DATA_DIR, "users.csv"), index=False)
        destinations.to_csv(os.path.join(DATA_DIR, "destinations.csv"), index=False)
        interactions.to_csv(os.path.join(DATA_DIR, "interactions.csv"), index=False)

        logger.info("=" * 60)
        logger.info(f"Dataset saved to {DATA_DIR}")
        logger.info(f"  • users.csv        : {len(users)} rows")
        logger.info(f"  • destinations.csv : {len(destinations)} rows")
        logger.info(f"  • interactions.csv : {len(interactions)} rows")
        logger.info("=" * 60)

        return users, destinations, interactions

    except Exception as e:
        raise DataGenerationError(
            message="Failed to generate dataset",
            details=str(e),
        ) from e


if __name__ == "__main__":
    generate_all()
