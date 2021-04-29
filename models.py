import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import pickle

from utils import Utils

class Models:

    def tree_training(self, X,y):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)
        t = tree.DecisionTreeClassifier(criterion='entropy', max_depth=7)
        model = t.fit(X,y)
        score_entrenamiento = model.score(x_train, y_train)
        score_prueba = model.score(x_test, y_test)

        return score_entrenamiento, score_prueba
        
