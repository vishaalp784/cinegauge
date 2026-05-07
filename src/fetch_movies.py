import requests
import pandas as pd
import os
import time
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TMDB_KEY")

def fetch_movies(total_pages=100):
    movies = []
    print("🎬 Fetching movies from TMDB...")
    
    for page in range(1, total_pages + 1):
        url = "https://api.themoviedb.org/3/discover/movie"
        params = {
            "api_key": API_KEY,
            "sort_by": "popularity.desc",
            "page": page,
            "language": "en-US",
            "vote_count.gte": 100
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        if "results" in data:
            movies.extend(data["results"])
        
        if page % 10 == 0:
            print(f"  ✓ Fetched {len(movies)} movies so far...")
        
        time.sleep(0.1)  # be nice to the API
    
    return pd.DataFrame(movies)

def clean_movies(df):
    print("🧹 Cleaning data...")
    
    # Keep useful columns
    cols = ["id","title","release_date","popularity",
            "vote_average","vote_count","genre_ids",
            "overview","original_language"]
    df = df[cols].copy()
    
    # Clean up
    df = df.dropna(subset=["title","release_date"])
    df = df[df["release_date"] != ""]
    df["release_year"] = pd.to_datetime(
        df["release_date"], errors="coerce"
    ).dt.year
    df = df.dropna(subset=["release_year"])
    df["release_year"] = df["release_year"].astype(int)
    df = df.drop_duplicates(subset=["id"])
    
    print(f"  ✓ Clean dataset: {len(df)} movies")
    return df

if __name__ == "__main__":
    # Fetch
    df_raw = fetch_movies(total_pages=100)
    
    # Clean
    df_clean = clean_movies(df_raw)
    
    # Save
    os.makedirs("data", exist_ok=True)
    df_clean.to_csv("data/movies.csv", index=False)
    print(f"\n✅ Saved {len(df_clean)} movies to data/movies.csv")
    print(df_clean.head())