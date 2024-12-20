import pickle
import os
from datetime import datetime
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def save_model(model, name=""):
    if name == "":
        name = model.__class__.__name__ + datetime.now().strftime("%Y-%m-%dT%H:%M:%S.f%z")

    with open(f'models/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

def load_model(name):
    with open(f'models/{name}.pkl', 'rb') as f:
        return pickle.load(f)
    
def cross_val(model, X, y, n_splits=5):
    return cross_val_score(model, X, y, cv=n_splits)