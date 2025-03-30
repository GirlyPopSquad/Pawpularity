import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier

from CsvManager import get_train_dataframe

feature_names = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 
                 'Group', 'Collage', 'Human', 'Info', 'Blur']
df = get_train_dataframe()
X = df[feature_names]
y = df['Pawpularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


def get_bagging_model_accuracy ():
    bagging_classifier = BaggingClassifier(n_estimators=25, random_state=30)
    bagging_classifier.fit(X_train, y_train)

    y_pred = bagging_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def get_boosting_model_accuracy ():
    base = DecisionTreeClassifier(criterion='gini', max_depth=10)
    model_ada = AdaBoostClassifier(n_estimators=10, random_state=42, estimator=base)
    model_ada.fit(X_train, y_train)

    y_pred = model_ada.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

def get_stacking_model_accuracy ():
    models = [('Logistic Regression', LogisticRegression()),
          ('Support Vector Classifier', SVC()),
          ('Decision Tree', DecisionTreeClassifier())]

    stacking_model = StackingClassifier(estimators=models,
    final_estimator=LogisticRegression(), cv=5)
    stacking_model.fit(X_train, y_train)

    y_pred = stacking_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy