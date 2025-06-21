from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np

def train_xgb(X_train, y_train, X_test, y_test):
    try:
        print(f"Training data shape: {X_train.shape}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        
        # Check for NaN values
        if np.isnan(X_train).any() or np.isnan(y_train).any():
            print("Warning: NaN values found in training data")
            X_train = np.nan_to_num(X_train)
            
        # Initialize model with more conservative parameters
        xgb = XGBClassifier(
            learning_rate=0.3,  
            max_depth=11,        
            n_estimators=170,   
        )
        
        xgb.fit(X_train, y_train)
        
        predictions = xgb.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        return xgb
        
    except Exception as e:
        print(f"Error in XGBoost training: {str(e)}")
        raise
    
testing_xgb = True

def test_xgb(X,y):
    param_grid_xgb = {
        "n_estimators" : [100, 150, 170],
        "max_depth": [11, 12],
        "learning_rate" : [ 0.3, 0.35, 0.35]
    }
    xgb = XGBClassifier()

    grid_search_xgb = GridSearchCV(estimator=xgb, cv = 5, param_grid=param_grid_xgb, scoring="accuracy")
    grid_search_xgb.fit(X, y)
    results = grid_search_xgb.cv_results_

    print("Best Parameters is ", grid_search_xgb.best_params_)
    print("Best Score: ", grid_search_xgb.best_score_)


