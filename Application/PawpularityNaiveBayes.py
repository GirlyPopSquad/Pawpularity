import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score
)

from CsvManager import get_train_dataframe

df = get_train_dataframe()
X = df[['Eyes', 'Face', 'Occlusion']]
y = df['Pawpularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


def train_pawpularity_bayes ():
    df['Pawpularity'] = df['Pawpularity'].apply(lambda x: 0 if x >= 75 else 1)
    
    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy