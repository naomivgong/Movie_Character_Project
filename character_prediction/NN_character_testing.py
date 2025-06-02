from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score



class MovieDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class CharacterPrediction(nn.Module):
    def __init__(self, input_dim, hid_dim1 = 128, hid_dim2 = 64, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim1)
        self.fc2 = nn.Linear(hid_dim1, hid_dim2)
        self.fc3 = nn.Linear(hid_dim2, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x),dim =1 )
        return x

vectorizer = TfidfVectorizer(max_features=5000)
characters = pd.read_csv("/Users/naomigong/Coding/Movie_Character_Project/character_prediction/character_lines.csv")
movie_name = "playback"
df_movies = characters[characters["movieTitle"] == movie_name].copy()
X = vectorizer.fit_transform(df_movies["text"]).toarray()
label_encoder = LabelEncoder()
df_movies["label"] = label_encoder.fit_transform(df_movies["character"])

df_trainval, df_test = train_test_split(df_movies, test_size=0.2, stratify=df_movies["label"], random_state=42)
df_train, df_val = train_test_split(df_trainval, test_size=0.25, stratify=df_trainval["label"], random_state=42)
vectorizer.fit(df_train["text"])

X_train = vectorizer.transform(df_train["text"]).toarray()
y_train = df_train["label"].values

X_val = vectorizer.transform(df_val["text"]).toarray()
y_val = df_val["label"].values

X_test = vectorizer.transform(df_test["text"]).toarray()
y_test = df_test["label"].values

train_dataset = MovieDataset(X_train, y_train)
test_dataset = MovieDataset(X_test, y_test)
val_dataset = MovieDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle =True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

X_full_train = np.concatenate([X_train, X_val])
y_full_train = np.concatenate([y_train, y_val])
full_train_loader = DataLoader(MovieDataset(X_full_train, y_full_train), batch_size=32, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = nn.NLLLoss()

def train_model(epochs, hid_dim1_ratio, hid_dim2_ratio, lr):
    model = CharacterPrediction(input_dim = X_train.shape[1], hid_dim1 = X_train.shape[1]//hid_dim1_ratio, hid_dim2 = X_train.shape[1]//hid_dim2_ratio,  output_dim = len(label_encoder.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.float().to(device), batch_y.long().to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return model

def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.float().to(device), y.long().to(device)
            preds = model(x)
            predicted = preds.argmax(dim=1)          
            all_preds.extend(predicted.cpu().numpy())  
            all_labels.extend(y.cpu().numpy())

    return accuracy_score(all_labels, all_preds)


def find_best_model():
    param_grid = {
        "hid_dim1_ratio": [2, 3, 4],
        "hid_dim2_ratio":[4, 6, 8],
        "lr": [0.01, 0.005, 0.03],
        "epochs": [10, 12, 15, 20, 25]
    }
    best_params = None
    highest_accuracy = 0

    for params in ParameterGrid(param_grid):
        hid_dim1 = X_train.shape[1] // params["hid_dim1_ratio"]
        hid_dim2 = X_train.shape[1]// params["hid_dim2_ratio"]
        model = CharacterPrediction(input_dim=X_train.shape[1], hid_dim1=hid_dim1, hid_dim2=hid_dim2 ,output_dim=len(label_encoder.classes_)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = params["lr"])

        for epoch in range(params["epochs"]):
            model.train() #in training mode
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.float().to(device), batch_y.long().to(device)
                optimizer.zero_grad() #clears gradient from previous batch
                output = model(batch_x) #feeds batch_x into the model and gets predicted log probabilities
                loss = loss_fn(output, batch_y) #compares true to prediction
                loss.backward() #computes gradient
                optimizer.step() #updates the model parameters using the gradients in loss.backwards

        accuracy = evaluate_model(model, val_loader)
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_params = params
        
    return best_params, highest_accuracy


def train_full_model(epochs, hid_dim1_ratio, hid_dim2_ratio, lr):
    input_dim_x = X_full_train.shape[1]
    best_model = CharacterPrediction(input_dim = input_dim_x, hid_dim1= input_dim_x // hid_dim1_ratio, hid_dim2 = input_dim_x//hid_dim2_ratio, output_dim=len(label_encoder.classes_)).to(device)
    optimizer = torch.optim.Adam(best_model.parameters(), lr = lr)

    for epoch in range(epochs):
        best_model.train()
        for batch_x, batch_y in full_train_loader:
            batch_x, batch_y = batch_x.float().to(device), batch_y.long().to(device)
            optimizer.zero_grad()
            output = best_model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()

    accuracy = evaluate_model(best_model, test_loader)
    return accuracy, best_model

def for_ensemble(model,loader):
    model.eval()
    nn_probs = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.float().to(device), y.long().to(device)
            probs = model(x).exp().cpu().numpy() #softmax
            nn_probs.extend(probs)
            

    return np.array(nn_probs)

best_model = find_best_model()

print(f"The best parameters are ", best_model[0])
print("The accuracy score is ", best_model[1] )
best_params = best_model[0]
epochs = best_params["epochs"]
hid_dim1_ratio = best_params["hid_dim1_ratio"]
hid_dim2_ratio = best_params["hid_dim2_ratio"]
lr = best_params["lr"]
final_acc, final_best_model = train_full_model(epochs, hid_dim1_ratio, hid_dim2_ratio, lr)
print(f"The final accuracy is : {final_acc}")
# model = train_model(25, 2, 6, 0.005)
# test_model(model)
y_pred_proba_nn = for_ensemble(final_best_model, test_loader)

def get_best_model():
    return final_best_model