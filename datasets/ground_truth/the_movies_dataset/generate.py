import os
import zipfile
import pandas as pd
import json
import csv
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    api = KaggleApi()
    api.authenticate()
    dataset_dir = "the-movies-dataset"
    if not os.path.exists(dataset_dir):
        print("Downloading The Movies Dataset from Kaggle...")
        api.dataset_download_files("rounakbanik/the-movies-dataset", path=dataset_dir, unzip=True)
    return dataset_dir


def load_data(dataset_dir):
    movies_path = os.path.join(dataset_dir, "movies_metadata.csv")
    movies = pd.read_csv(movies_path, low_memory=False)
    movies = movies.dropna(subset=["title", "release_date"])
    movies["release_year"] = movies["release_date"].apply(
        lambda x: str(x).split("-")[0] if isinstance(x, str) else None
    )
    movies["release_year"] = pd.to_numeric(movies["release_year"], errors="coerce")
    movies = movies.dropna(subset=["release_year"])
    return movies


FEATURES = [
    "Classic (Before 1980)",
    "Old Film (1980-1999)",
    "Modern Film (2000-2010)",
    "Recent Film (After 2010)",

    # Genres
    "Action / Fast-Paced",
    "Adventure / Epic",
    "Comedy",
    "Romantic",
    "Drama",
    "Horror",
    "Thriller / Crime",
    "Sci-Fi / Space",
    "Fantasy / Mythic",
    "Animation / Family",
    "Mystery",
    "Musical",
    "War / Historical",
    "Documentary",
    "Western",

    # Aesthetic / Tone
    "Visually Stunning",
    "Dark / Gritty",
    "Lighthearted / Feel-Good",
    "Philosophical / Existential",
    "Character-Driven",
    "Plot-Driven / Complex",
    "Emotional / Heartfelt",
    "Violent / Intense",

    # Commercial & Audience
    "High Budget",
    "Low Budget",
    "Popular",
    "Critically Acclaimed",
    "Cult / Niche",
    "Family-Friendly",
    "Social / Political Theme",
    "Based on a True Story"
]


def extract_features(movies):
    feature_map = {}

    for _, row in movies.iterrows():
        title = row["title"]
        genres = str(row.get("genres", "")).lower()
        overview = str(row.get("overview", "")).lower()
        budget = float(row.get("budget", 0))
        popularity = float(row.get("popularity", 0))
        vote_average = float(row.get("vote_average", 0))
        year = int(row.get("release_year", 0))

        movie_features = []

        if year < 1980:
            movie_features.append("Classic (Before 1980)")
        elif 1980 <= year < 2000:
            movie_features.append("Old Film (1980-1999)")
        elif 2000 <= year <= 2010:
            movie_features.append("Modern Film (2000-2010)")
        elif year > 2010:
            movie_features.append("Recent Film (After 2010)")

        if "action" in genres:
            movie_features.append("Action / Fast-Paced")
        if "adventure" in genres:
            movie_features.append("Adventure / Epic")
        if "comedy" in genres:
            movie_features.append("Comedy")
        if "romance" in genres:
            movie_features.append("Romantic")
        if "drama" in genres:
            movie_features.append("Drama")
        if "horror" in genres:
            movie_features.append("Horror")
        if "thriller" in genres or "crime" in genres:
            movie_features.append("Thriller / Crime")
        if "science fiction" in genres or "sci-fi" in genres or "space" in genres:
            movie_features.append("Sci-Fi / Space")
        if "fantasy" in genres or "myth" in genres:
            movie_features.append("Fantasy / Mythic")
        if "animation" in genres or "family" in genres:
            movie_features.append("Animation / Family")
        if "mystery" in genres:
            movie_features.append("Mystery")
        if "music" in genres or "musical" in genres:
            movie_features.append("Musical")
        if "war" in genres or "history" in genres:
            movie_features.append("War / Historical")
        if "documentary" in genres:
            movie_features.append("Documentary")
        if "western" in genres:
            movie_features.append("Western")

        if "beautiful" in overview or "cinematography" in overview:
            movie_features.append("Visually Stunning")
        if "dark" in overview or "gritty" in overview or "noir" in overview:
            movie_features.append("Dark / Gritty")
        if "funny" in overview or "light-hearted" in overview or "joyful" in overview:
            movie_features.append("Lighthearted / Feel-Good")
        if "philosophical" in overview or "existential" in overview:
            movie_features.append("Philosophical / Existential")
        if "character" in overview or "relationships" in overview:
            movie_features.append("Character-Driven")
        if "plot" in overview or "mystery" in overview or "twist" in overview:
            movie_features.append("Plot-Driven / Complex")
        if "emotional" in overview or "heartfelt" in overview or "touching" in overview:
            movie_features.append("Emotional / Heartfelt")
        if "violence" in overview or "intense" in overview or "fight" in overview:
            movie_features.append("Violent / Intense")
        if "based on a true story" in overview or "true story" in overview:
            movie_features.append("Based on a True Story")

        if budget > 80_000_000:
            movie_features.append("High Budget")
        elif 0 < budget < 10_000_000:
            movie_features.append("Low Budget")

        if popularity > 10:
            movie_features.append("Popular")
        if vote_average >= 7.5:
            movie_features.append("Critically Acclaimed")
        if 5.5 < vote_average < 7 and popularity < 5:
            movie_features.append("Cult / Niche")
        if "family" in genres or "animation" in genres:
            movie_features.append("Family-Friendly")
        if "war" in overview or "politics" in overview or "society" in overview:
            movie_features.append("Social / Political Theme")

        # Map to feature indices
        feature_indices = [FEATURES.index(f) + 1 for f in movie_features if f in FEATURES]
        feature_map[title] = feature_indices

    return feature_map


def write_output(feature_map):
    items = sorted(list(feature_map.keys()))

    # items.csv
    with open("items.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Item Name"])
        for item in items:
            writer.writerow([item])

    # latent_characteristics.csv
    with open("latent_characteristics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature Dimension"])
        for feat in FEATURES:
            writer.writerow([feat])

    # item_feature_map.json
    with open("item_feature_map.json", "w", encoding="utf-8") as f:
        json.dump(feature_map, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    dataset_dir = download_dataset()
    movies = load_data(dataset_dir)
    print(f"Loaded {len(movies)} movies from dataset.")
    feature_map = extract_features(movies)
    print(f"Extracted interpretable features for {len(feature_map)} movies.")
    write_output(feature_map)
    print("Files created: items.csv, latent_characteristics.csv, item_feature_map.json")