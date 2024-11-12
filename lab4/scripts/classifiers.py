import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# KNN Class
# KNN Class
class KNNModel:
    def __init__(self):
        # Hyperparameters definition
        self.params = {
            "n_neighbors": [3, 5, 7, 10],  # Добавлены значения соседей
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree"],
            "metric": ["euclidean", "manhattan", "minkowski"]  # Добавлен метрика "minkowski"
        }

    def train(self, x_train, y_train):
        # Define the model
        model = KNeighborsClassifier()
        
        # Hyperparameter optimization
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=self.params,
                                   scoring="f1_weighted",  # Изменено на F1-score для учёта дисбаланса
                                   cv=5,
                                   n_jobs=-1)
        grid_search.fit(x_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        print("Best KNN Parameters:", grid_search.best_params_)
        
        return best_model


# SVM Class
class SVMModel:
    def __init__(self):
        # Hyperparameters definition
        self.params = {
            "C": [0.1, 1, 10, 100],  # Расширенный диапазон значений C
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"],
            "degree": [2, 3, 4],  # Добавлен полином 4-й степени
            "class_weight": ["balanced"]  # Учитывает дисбаланс классов
        }

    def train(self, x_train, y_train):
        # Define the model
        model = SVC()
        
        # Hyperparameter optimization
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=self.params,
                                   scoring="f1_weighted",  # Изменено на F1-score для учёта дисбаланса
                                   cv=5,
                                   n_jobs=-1)
        grid_search.fit(x_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        print("Best SVM Parameters:", grid_search.best_params_)
        
        return best_model


# Random Forest Class
class RandomForestModel:
    def __init__(self):
        # Hyperparameters definition 
        self.params = {
            "n_estimators": [10, 20, 40],  
            "max_depth": [10, 20, 30],  
            "min_samples_split": [5, 10, 20],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"]
        }

    def train(self, x_train, y_train):
        # Define the model with balanced class weights
        model = RandomForestClassifier(class_weight="balanced")
        
        # Hyperparameter optimization with F1-weighted score
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=self.params,
                                   scoring="f1_weighted",  # Изменено на F1-score для учёта дисбаланса
                                   cv=5,
                                   n_jobs=-1)
        grid_search.fit(x_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        print("Best Random Forest Parameters:", grid_search.best_params_)
        
        return best_model
