# 🎵 Spotify Music Recommender System

A content-based music recommendation system using Scikit-learn's K-Nearest Neighbors (KNN) algorithm. Recommends songs based on audio features similarity within the same genre.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5.0%2B-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.2.2%2B-brightgreen)

## Features

- **Content-based filtering** using audio features (danceability, energy, valence, tempo, acousticness)
- **Genre-aware recommendations** - finds similar songs within the same genre
- **Artist filtering** option for more precise matching
- **Duplicate prevention** ensures diverse recommendations
- **Configurable parameters** for number of recommendations and search scope

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`:

```bash
pandas>=2.2.2
scikit-learn>=1.5.0
kagglehub[pandas-datasets]>=0.1.8
```
## Installation
### Clone the repository:
```bash
git clone https://github.com/your-username/spotify-recommender.git
cd spotify-recommender
```
### Install dependencies:
``` bash
pip install -r requirements.txt
```
### Download the dataset:
Place your dataset.csv in the data/ folder

### Run the recommender system:
``` bash
python spotify_recommender.py
```
## Configuration
Modify these global variables in spotify_recommender.py to customize:

``` python
DATASET_PATH = 'data/dataset.csv'  # Path to dataset
FEATURES = ['danceability', 'energy', 'valence', 'tempo', 'acousticness']  # Audio features
N_RECOMMENDATIONS = 5  # Number of recommendations to display
N_QUERY_NEIGHBORS = 50  # Initial pool of similar songs to consider
TEST_SONG = "I Ain't Worried"  # Default song for testing
TEST_ARTIST = "OneRepublic"  # Default artist for testing
```

## Example Output
``` plaintext
Loading data and training model...

Finding recommendations similar to: 'I Ain't Worried'

Top 5 Recommendations:
1. You're Only Human (Second Wind) - Billy Joel
2. Hold Me Closer - Purple Disco Machine Remix - Elton John;Britney Spears;Purple Disco Machine
3. We Didn't Start the Fire - Billy Joel
4. Sunshine - OneRepublic
5. Over My Head (Cable Car) - The Fray
```

## How It Works
Data Loading: Loads the Spotify tracks dataset with audio features
Preprocessing: Normalizes features using MinMaxScaler
Model Training: Creates a KNN model using cosine similarity

## Recommendation:
Finds the query song in the dataset
Filters songs by the same genre
Computes similarity based on audio features
Returns the most similar songs (excluding duplicates)

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
Dataset from Spotify Tracks Dataset on Kaggle
Scikit-learn for the machine learning framework
