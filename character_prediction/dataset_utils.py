import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

class MovieDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_movie_data(filepath, movie_name, max_features=1000):
    df = pd.read_csv(filepath)
    df = df[df["movieTitle"] == movie_name].copy()

    vectorizer = TfidfVectorizer(max_features=max_features)
    label_encoder = LabelEncoder()

    df["label"] = label_encoder.fit_transform(df["character"])
    X = vectorizer.fit_transform(df["text"]).toarray()
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder, vectorizer
