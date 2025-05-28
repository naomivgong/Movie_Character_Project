import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


#create Random Forest Classifier
characters = pd.read_csv("/Users/naomigong/Coding/Movie_Character_Project/character_prediction/character_lines.csv")
movie_name = "playback"
df_movies = characters[characters["movieTitle"] == movie_name]
#initialize objects
vectorizer = TfidfVectorizer(max_features=5000)
label_encoder = LabelEncoder()

X = vectorizer.fit_transform(df_movies["text"]).toarray()
y = label_encoder.fit_transform(df_movies["character"])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)


testing_rf = False
if testing_rf == True:
    param_grid_rf = {
        "n_estimators" : [50, 100],
        "criterion" : ["entropy"],
        "max_depth": [60, 90],
    }
    random_forest = RandomForestClassifier(random_state=0)
    grid_search = GridSearchCV(estimator=random_forest, cv = 5, param_grid=param_grid_rf, scoring="accuracy")
    grid_search.fit(X, y)
    results = grid_search.cv_results_

    print("TESTING RANDOM FOREST")
    for mean_score, params in zip(results["mean_test_score"], results["params"]):
        print(f"{mean_score:.4f} --> {params}")

    print("Best Parameters is ", grid_search.best_params_)
    print("Best Score: ", grid_search.best_score_)

best_rf_model = RandomForestClassifier(criterion="entropy", max_depth=90, n_estimators=50)
best_rf_model.fit(x_train, y_train)
y_pred = best_rf_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("The accuracy of the model is: ", accuracy)

testing_xgb = False

if testing_xgb == True:
    print("TESTING XGBOOST")
    param_grid_xgb = {
        "n_estimators" : [100, 150, 170],
        "max_depth": [11, 12],
        "learning_rate" : [ 0.3, 0.35, 0.35]
    }
    xgb = XGBClassifier(random_state=0)

    grid_search_xgb = GridSearchCV(estimator=xgb, cv = 5, param_grid=param_grid_xgb, scoring="accuracy")
    grid_search_xgb.fit(X, y)
    results = grid_search_xgb.cv_results_

    for mean_score, params in zip(results["mean_test_score"], results["params"]):
        print(f"{mean_score:.4f} --> {params}")

    print("Best Parameters is ", grid_search_xgb.best_params_)
    print("Best Score: ", grid_search_xgb.best_score_)

best_xgb_model = XGBClassifier(learning_rate = 0.3, max_depth = 11, n_estimators = 170)
best_xgb_model.fit(x_train, y_train)
y_pred = best_xgb_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("The accuracy of the model is: ", accuracy)


