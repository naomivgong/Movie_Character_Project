from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

testing_rf = False
if testing_rf == True:
    def test(X, y):
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


def train_rf(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(criterion="entropy", max_depth=90, n_estimators=50)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    print("RF accuracy:", acc)
    return rf
