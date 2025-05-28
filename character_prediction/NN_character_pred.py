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

#vectorizer = TfidfVectorizer(max_features=5000)
characters = pd.read_csv("/Users/naomigong/Coding/Movie_Character_Project/character_prediction/character_lines.csv")
movie_name = "playback"
df_movies = characters[characters["movieTitle"] == movie_name]
#X = vectorizer.fit_transform(df_movies["text"]).toarray()

#try using spacy instead
nlp = spacy.load("en_core_web_md")
def get_embedding(text):
    return nlp(text).vector

X = np.array(df_movies["text"].apply(get_embedding).tolist())
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_movies["character"])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
train_dataset = MovieDataset(x_train, y_train)
test_dataset = MovieDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharacterPrediction(input_dim = x_train.shape[1], hid_dim1 = x_train.shape[1]//2, hid_dim2 = x_train.shape[1]//4,  output_dim = len(label_encoder.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
loss_fn = nn.NLLLoss()

for epoch in range(25):
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


model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for test_x, test_y in test_loader:
        test_x, test_y = test_x.float().to(device), test_y.long().to(device)
        preds = model(test_x)
        predicted = preds.argmax(dim=1)          
        all_preds.extend(predicted.cpu().numpy())  
        all_labels.extend(test_y.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))