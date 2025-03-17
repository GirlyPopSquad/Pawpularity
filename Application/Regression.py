from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, log_loss
from sklearn.model_selection import train_test_split, GridSearchCV
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
    
    param_grid = {
        'fit_intercept': [True, False]
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    
    grid_search.fit(X_train, y_train)

    return model, mse, r2, grid_search.best_params_, grid_search.best_score_

def train_human_model():
    X_train, X_test, y_train, y_test = load_human_data()
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_preb_prob = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_preb_prob)
    
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],          
        'penalty': ['l2', 'l1'],               
        'solver': ['liblinear', 'saga']        
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    
    grid_search.fit(X_train, y_train)
    
    return model, accuracy, loss, grid_search.best_params_, grid_search.best_score_


