import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.model_selection import train_test_split
import seaborn as sns


class Utils:

    def load_from_csv(self, path): #Debe de llamarse a si mismo
        return pd.read_csv(path)

    def load_from_mysql(self):
        pass

    def features_target(self, dataset, drop_cols, cols_wanted, y):
        w = dataset.drop(drop_cols, axis=1)
        X = pd.get_dummies(w, cols_wanted)
        X = X.drop(y, axis=1)
        y = dataset[y]
        return X, y

    def grafica_barras(self, dataset, columna):
        fig = sns.displot(dataset, x= columna)
        fig.savefig('./out/grafica_1.png')

