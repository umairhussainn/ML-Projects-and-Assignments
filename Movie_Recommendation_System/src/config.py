

import os

# Project Information
PROJECT_NAME = "Movie Recommendation System"
VERSION = "1.0.0"
AUTHOR = "Umair Hussain"
GITHUB_URL = "https://github.com/umairhussainn"  

# Data Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "ml-latest-small")
MOVIES_PATH = os.path.join(DATA_DIR, "movies.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")
TAGS_PATH = os.path.join(DATA_DIR, "tags.csv")
LINKS_PATH = os.path.join(DATA_DIR, "links.csv")

# Model Parameters
DEFAULT_NUM_RECOMMENDATIONS = 10
MIN_RECOMMENDATIONS = 5
MAX_RECOMMENDATIONS = 20

# UI Configuration
PAGE_TITLE = "Movie Recommender"
PAGE_ICON = "🎬"
LAYOUT = "wide"

# Color Scheme
PRIMARY_COLOR = "#FF6347"  # Tomato Red
SECONDARY_COLOR = "#808080"  # Gray
BACKGROUND_COLOR = "#f0f2f6"  # Light Gray

# Messages
SUCCESS_MESSAGE = "✅ Found {count} movies similar to '{movie}'"
ERROR_MESSAGE = "❌ '{movie}' not found in the database."
LOADING_MESSAGE = "Loading movie database..."
FINDING_MESSAGE = "Finding similar movies..."
