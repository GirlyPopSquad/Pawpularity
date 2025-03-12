from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
import pandas as pd

def load_pawpularity_data():
    df = pd.read_csv('Application/Data/train.csv')
    X = df.drop(columns=['Id', 'Pawpularity'])
    y = df['Pawpularity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    return X_train, X_test, y_train, y_test

def load_human_data():
    df = pd.read_csv('Application/Data/train.csv')
    X = df.drop(columns=['Id', 'Human', 'Pawpularity'])
    y = df['Human']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    return X_train, X_test, y_train, y_test

def train_pawpularity_model():
    X_train, X_test, y_train, y_test = load_pawpularity_data()
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2

def train_human_model():
    X_train, X_test, y_train, y_test = load_human_data()
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_preb_prob = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_preb_prob)
    
    return model, accuracy, loss


