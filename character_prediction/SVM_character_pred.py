import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

character_prediction_models = {}

#returns the rows of the movie that matches the movie title specified 
def extract_movie(movie_name, dataframe):
    movie_rows = dataframe[dataframe["movieTitle"]==movie_name]
    return movie_rows

# passes in the rows of the dataframe that matches the movie title
def train_model(movie_data):

    classifier = svm.SVC(decision_function_shape='ovo', probability=True)
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("model", classifier)
    ])

    movie_text = movie_data["text"]
    movie_character = movie_data["character"]
    pipeline.fit(movie_text, movie_character)
    return pipeline

#create a model for all movies 
def create_models(dataframe):
    movie_titles = dataframe["movieTitle"].unique()
    for movie in movie_titles:
        movie_data = extract_movie(movie, dataframe)
        model = train_model(movie_data)
        #add the model to the cache
        character_prediction_models[movie] = model


def check_movie_cache(movie_name):
    if movie_name in character_prediction_models:
        return True
    else:
        return False
    
def predict_character(movie_name, dialogue):
    if check_movie_cache(movie_name):
        movie_model = character_prediction_models[movie_name]
    else:
        print("No movie found")
        return
    
    char_ids = movie_model.named_steps["model"].classes_
    probs = movie_model.predict_proba([dialogue])[0]

    result = {char: round(prob, 3) for char, prob in zip(char_ids, probs)}
    sorted_result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
    return sorted_result


if __name__ == "__main__":
    character_lines = pd.read_csv("character_prediction/character_lines.csv")
    create_models(character_lines)
    while True:
        check_in = input("Predicting Movies. Continue or Start [Y] or quit[q]")
        if check_in == "q":
            break
        movie_name = input("Enter in the Movie Name")
        movie_name = movie_name.strip()
        print(movie_name)
        dialogue = input("Enter in a speech/dialogue. (ex. 'Just take the pencil from him')")
        preds = predict_character(movie_name, dialogue)
        print(preds)
