import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

def train_occlusion_bayes ():
    df = pd.read_csv('Application/Data/train.csv')
    X = df.drop(columns=['Id', 'Pawpularity', 'Occlusion'])
    y = df['Occlusion']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    model = GaussianNB()
    model.fit(X_train, y_train)
    
    return model
    