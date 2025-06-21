from dataset_utils import load_movie_data
from rf_model import train_rf
from xgb_model import train_xgb, test_xgb
from nn_model import train_nn, predict_nn
import numpy as np
from sklearn.metrics import accuracy_score

# === Load Data ===
X_train, X_test, y_train, y_test, label_encoder, _ = load_movie_data(
    "/Users/naomigong/Coding/Movie_Character_Project/character_prediction/character_lines.csv",
    "playback",
    max_features=1000
)
# === Train Models ===
# Choose which model you want:
ensemble_accs = []
for i in range(3):
    use_rf = False
    if use_rf:
        base_model = train_rf(X_train, y_train, X_test, y_test)
        proba_rf = base_model.predict_proba(X_test)
    else:
        base_model = train_xgb(X_train, y_train, X_test, y_test)
        proba_rf = base_model.predict_proba(X_test)

    nn_model = train_nn(X_train, y_train, X_test, y_test, output_dim=len(label_encoder.classes_))
    proba_nn = predict_nn(nn_model, X_test, y_test)

    # === Ensemble ===
    ensemble_proba = (proba_rf + proba_nn) / 2
    ensemble_preds = np.argmax(ensemble_proba, axis=1)
    ensemble_acc = accuracy_score(y_test, ensemble_preds)
    ensemble_accs.append(ensemble_acc)
    print(f"Ensemble accuracy: {ensemble_acc:.4f}")

print(f"Average ensemble accuracy: {np.mean(ensemble_accs):.4f}")

# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from xgboost import XGBClassifier
# #Neural Network
# from sklearn.feature_extraction.text import TfidfVectorizer
# import spacy
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import classification_report
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import ParameterGrid
# from sklearn.metrics import accuracy_score



# #create Random Forest Classifier
# characters = pd.read_csv("/Users/naomigong/Coding/Movie_Character_Project/character_prediction/character_lines.csv")
# movie_name = "playback"
# df_movies = characters[characters["movieTitle"] == movie_name]
# #initialize objects
# vectorizer = TfidfVectorizer(max_features=1000)
# label_encoder = LabelEncoder()

# X = vectorizer.fit_transform(df_movies["text"]).toarray()
# y = label_encoder.fit_transform(df_movies["character"])
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)


# testing_rf = False
# if testing_rf == True:
#     param_grid_rf = {
#         "n_estimators" : [50, 100],
#         "criterion" : ["entropy"],
#         "max_depth": [60, 90],
#     }
#     random_forest = RandomForestClassifier(random_state=0)
#     grid_search = GridSearchCV(estimator=random_forest, cv = 5, param_grid=param_grid_rf, scoring="accuracy")
#     grid_search.fit(X, y)
#     results = grid_search.cv_results_

#     print("TESTING RANDOM FOREST")
#     for mean_score, params in zip(results["mean_test_score"], results["params"]):
#         print(f"{mean_score:.4f} --> {params}")

#     print("Best Parameters is ", grid_search.best_params_)
#     print("Best Score: ", grid_search.best_score_)

# best_rf_model = RandomForestClassifier(criterion="entropy", max_depth=90, n_estimators=50)
# best_rf_model.fit(x_train, y_train)
# y_pred = best_rf_model.predict(x_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("The accuracy of the model is: ", accuracy)

# testing_xgb = False

# if testing_xgb == True:
#     print("TESTING XGBOOST")
#     param_grid_xgb = {
#         "n_estimators" : [100, 150, 170],
#         "max_depth": [11, 12],
#         "learning_rate" : [ 0.3, 0.35, 0.35]
#     }
#     xgb = XGBClassifier(random_state=0)

#     grid_search_xgb = GridSearchCV(estimator=xgb, cv = 5, param_grid=param_grid_xgb, scoring="accuracy")
#     grid_search_xgb.fit(X, y)
#     results = grid_search_xgb.cv_results_

#     for mean_score, params in zip(results["mean_test_score"], results["params"]):
#         print(f"{mean_score:.4f} --> {params}")

#     print("Best Parameters is ", grid_search_xgb.best_params_)
#     print("Best Score: ", grid_search_xgb.best_score_)

# best_xgb_model = XGBClassifier(learning_rate = 0.3, max_depth = 11, n_estimators = 170)
# best_xgb_model.fit(x_train, y_train)
# y_pred = best_xgb_model.predict_proba(x_test)
# y_pred_max = np.argmax(y_pred, axis =1)
# accuracy = accuracy_score(y_test, y_pred_max)
# print("The accuracy of the model is: ", accuracy)





# class MovieDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X, dtype=torch.float32)  # explicitly float32
#         self.y = torch.tensor(y, dtype=torch.long)

#     def __len__(self):
#         return len(self.X)
    
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]
    

# class CharacterPrediction(nn.Module):
#     def __init__(self, input_dim, hid_dim1 = 128, hid_dim2 = 64, output_dim=4):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hid_dim1)
#         self.fc2 = nn.Linear(hid_dim1, hid_dim2)
#         self.fc3 = nn.Linear(hid_dim2, output_dim)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.log_softmax(self.fc3(x),dim =1 )
#         return x
# print("PREPPING DATA")
# df_movies["label"] = y

# df_trainval, df_test = train_test_split(df_movies, test_size=0.2, stratify=df_movies["label"], random_state=42)
# df_train, df_val = train_test_split(df_trainval, test_size=0.25, stratify=df_trainval["label"], random_state=42)
# vectorizer.fit(df_train["text"])
# print("Split")

# X_train = vectorizer.transform(df_train["text"]).toarray()
# Y_train = df_train["label"].values
# print("split1")

# X_val = vectorizer.transform(df_val["text"]).toarray()
# Y_val = df_val["label"].values
# print("split2")

# X_test = vectorizer.transform(df_test["text"]).toarray()
# Y_test = df_test["label"].values
# print("vector")
# print("X_train dtype:", X_train.dtype, "shape:", X_train.shape)
# print("y_train dtype:", y_train.dtype, "shape:", y_train.shape)
# print("Checking for NaNs:", np.any(np.isnan(X_train)))
# print("Creating MovieDataset now...")


# train_dataset = MovieDataset(X_train, Y_train)
# print(1)
# test_dataset = MovieDataset(X_test, Y_test)
# print(2)
# val_dataset = MovieDataset(X_val, Y_val)
# print("split3")

# train_loader = DataLoader(train_dataset, batch_size = 32, shuffle =True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# X_full_train = np.concatenate([X_train, X_val])
# y_full_train = np.concatenate([Y_train, Y_val])
# full_train_loader = DataLoader(MovieDataset(X_full_train, y_full_train), batch_size=32, shuffle=True)
# print("split4")


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loss_fn = nn.NLLLoss()
# print("FINISH DATA PREP")
# def train_model(epochs, hid_dim1_ratio, hid_dim2_ratio, lr):
#     model = CharacterPrediction(input_dim = X_train.shape[1], hid_dim1 = X_train.shape[1]//hid_dim1_ratio, hid_dim2 = X_train.shape[1]//hid_dim2_ratio,  output_dim = len(label_encoder.classes_)).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr = lr)
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         for batch_x, batch_y in train_loader:
#             batch_x, batch_y = batch_x.float().to(device), batch_y.long().to(device)
#             optimizer.zero_grad()
#             output = model(batch_x)
#             loss = loss_fn(output, batch_y)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

#     return model

# def evaluate_model(model, loader):
#     model.eval()
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for x, y in loader:
#             x, y = x.float().to(device), y.long().to(device)
#             preds = model(x)
#             predicted = preds.argmax(dim=1)          
#             all_preds.extend(predicted.cpu().numpy())  
#             all_labels.extend(y.cpu().numpy())

#     return accuracy_score(all_labels, all_preds)


# def find_best_model():
#     param_grid = {
#         "hid_dim1_ratio": [2, 3, 4],
#         "hid_dim2_ratio":[4, 6, 8],
#         "lr": [0.01, 0.005, 0.03],
#         "epochs": [10, 12, 15, 20, 25]
#     }
#     best_params = None
#     highest_accuracy = 0

#     for params in ParameterGrid(param_grid):
#         hid_dim1 = X_train.shape[1] // params["hid_dim1_ratio"]
#         hid_dim2 = X_train.shape[1]// params["hid_dim2_ratio"]
#         model = CharacterPrediction(input_dim=X_train.shape[1], hid_dim1=hid_dim1, hid_dim2=hid_dim2 ,output_dim=len(label_encoder.classes_)).to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr = params["lr"])

#         for epoch in range(params["epochs"]):
#             model.train() #in training mode
#             for batch_x, batch_y in train_loader:
#                 batch_x, batch_y = batch_x.float().to(device), batch_y.long().to(device)
#                 optimizer.zero_grad() #clears gradient from previous batch
#                 output = model(batch_x) #feeds batch_x into the model and gets predicted log probabilities
#                 loss = loss_fn(output, batch_y) #compares true to prediction
#                 loss.backward() #computes gradient
#                 optimizer.step() #updates the model parameters using the gradients in loss.backwards

#         accuracy = evaluate_model(model, val_loader)
#         if accuracy > highest_accuracy:
#             highest_accuracy = accuracy
#             best_params = params
        
#     return best_params, highest_accuracy


# def train_full_model(epochs, hid_dim1_ratio, hid_dim2_ratio, lr):
#     input_dim_x = X_full_train.shape[1]
#     best_model = CharacterPrediction(input_dim = input_dim_x, hid_dim1= input_dim_x // hid_dim1_ratio, hid_dim2 = input_dim_x//hid_dim2_ratio, output_dim=len(label_encoder.classes_)).to(device)
#     optimizer = torch.optim.Adam(best_model.parameters(), lr = lr)

#     for epoch in range(epochs):
#         best_model.train()
#         for batch_x, batch_y in full_train_loader:
#             batch_x, batch_y = batch_x.float().to(device), batch_y.long().to(device)
#             optimizer.zero_grad()
#             output = best_model(batch_x)
#             loss = loss_fn(output, batch_y)
#             loss.backward()
#             optimizer.step()

#     accuracy = evaluate_model(best_model, test_loader)
#     return accuracy, best_model

# def for_ensemble(model,loader):
#     model.eval()
#     nn_probs = []

#     with torch.no_grad():
#         for x, y in loader:
#             x, y = x.float().to(device), y.long().to(device)
#             probs = model(x).exp().cpu().numpy() #softmax
#             nn_probs.extend(probs)
            

#     return np.array(nn_probs)

# testing_nn = False
# if testing_nn == True:
#     best_model = find_best_model()
#     print(f"The best parameters are ", best_model[0])
#     print("The accuracy score is ", best_model[1] )
#     best_params = best_model[0]
#     epochs = best_params["epochs"]
#     hid_dim1_ratio = best_params["hid_dim1_ratio"]
#     hid_dim2_ratio = best_params["hid_dim2_ratio"]
#     lr = best_params["lr"]
#     final_acc, final_best_model = train_full_model(epochs, hid_dim1_ratio, hid_dim2_ratio, lr)
#     print(f"The final accuracy is : {final_acc}")
#     # model = train_model(25, 2, 6, 0.005)
#     # test_model(model)

# def get_best_model():
#     return final_best_model



# #===Ensemble====
# final_best_model = train_full_model(epochs=10, hid_dim1_ratio=2, hid_dim2_ratio=6, lr=0.03)
# y_pred_proba_xgb = best_xgb_model.predict_proba(x_test)
# y_pred_proba_nn = for_ensemble(final_best_model, test_loader)

# # Ensure both are numpy arrays of the same shape
# ensemble_proba = (y_pred_proba_xgb + y_pred_proba_nn) / 2
# ensemble_preds = np.argmax(ensemble_proba, axis=1)

# ensemble_acc = accuracy_score(y_test, ensemble_preds)
# print(f"Ensemble accuracy (average prob): {ensemble_acc:.4f}")