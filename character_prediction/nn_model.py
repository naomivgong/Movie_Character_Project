import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from dataset_utils import MovieDataset
import numpy as np

class CharacterPrediction(nn.Module):
    def __init__(self, input_dim, hid_dim1, hid_dim2, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim1)
        self.fc2 = nn.Linear(hid_dim1, hid_dim2)
        self.fc3 = nn.Linear(hid_dim2, output_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)

def train_nn(X_train, y_train, X_test, y_test, output_dim, epochs=10, hid_dim1_ratio=2, hid_dim2_ratio=6, lr=0.03):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]
    model = CharacterPrediction(input_dim, input_dim // hid_dim1_ratio, input_dim // hid_dim2_ratio, output_dim).to(device)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(MovieDataset(X_train, y_train), batch_size=32, shuffle=True)
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    return model

def predict_nn(model, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    loader = DataLoader(MovieDataset(X_test, y_test), batch_size=32)
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            probs.extend(model(x).exp().cpu().numpy())
    return np.array(probs)
