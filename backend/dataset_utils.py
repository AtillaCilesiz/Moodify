import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Dataset yolu
DATASET_PATH = '/Users/ati/Documents/Çalışmalarım/Coding/dataset/dataset.csv'

# Global cache
CACHED_DATASET = None

def load_and_clean_dataset(dataset_path=DATASET_PATH):
    """
    Spotify audio features datasetini yükler, temizler ve normalize eder.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset bulunamadı: {dataset_path}")

    df = pd.read_csv(dataset_path)

    features = [
        'danceability', 'energy', 'valence', 'tempo',
        'acousticness', 'speechiness', 'instrumentalness', 'liveness', 'loudness'
    ]
    required_columns = ['track_id'] + features

    try:
        df = df[required_columns]
    except KeyError as e:
        raise KeyError(f"Dataset kolonlarında eksik var. Beklenen kolonlar: {required_columns}. Hata: {e}")

    df = df.dropna()
    df = df[df['track_id'].notna()]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])

    df_scaled = pd.DataFrame(scaled_features, columns=features)
    df_scaled['track_id'] = df['track_id'].values
    df_scaled = df_scaled[['track_id'] + features]

    return df_scaled

def initialize_dataset():
    global CACHED_DATASET
    if CACHED_DATASET is None:
        print("Dataset yükleniyor...")
        CACHED_DATASET = load_and_clean_dataset()
        print(f"Dataset {len(CACHED_DATASET)} şarkı ile yüklendi.")
    else:
        print("Dataset zaten yüklenmiş.")


def get_audio_features_for_track(track_id):
    """
    Hazırdaki cache'den bir track_id için audio features verilerini getirir.
    """
    if CACHED_DATASET is None:
        raise ValueError("Dataset yüklenmemiş. Önce initialize_dataset() çağır.")

    track_data = CACHED_DATASET[CACHED_DATASET['track_id'] == track_id]

    if track_data.empty:
        return None

    features = track_data.drop(columns=['track_id']).iloc[0].to_dict()
    return features
