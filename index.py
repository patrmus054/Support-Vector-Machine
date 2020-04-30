import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=Warning)
data = np.loadtxt('Data_User_Modeling_Dataset.csv', delimiter=',', skiprows=1)
X = data[:, :5]
y = data[:, 5]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,  random_state=2)
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Dokładność:", metrics.accuracy_score(y_test, y_pred))
parameters = {'kernel': ('linear', 'rbf', 'poly'),
              'C': [2**-5, 10],
              'gamma': [2**-5, 5],
              'degree': [2, 3, 4, 5]}
svc = svm.SVC()
top_classifier_gs = None
best_cv_gs = 2
best_score_gs = 0
# 4-krotna walidacji krzyzow
for i in range(2, 20):
    setGS = GridSearchCV(svc, parameters, cv=i)
    setGS.fit(X_train, y_train)

    if(setGS.best_score_ > best_score_gs):
        best_cv_gs = i
        top_classifier_gs = setGS
        best_score_gs = setGS.best_score_
print("Najlepszy klasyfikator: {}".format(str(top_classifier_gs.best_params_)))
print("Wynik: {}".format(best_cv_gs))
print("najlepsze cv {}".format(best_score_gs))
print("Dokladność {}".format(top_classifier_gs.score(X_test, y_test)))

top_classifier_rs = None
best_cv_rs = 2
best_score_rs = 0
for i in range(2, 20):
    setRS = RandomizedSearchCV(svc, parameters, cv=i, random_state=2)
    setRS.fit(X, y)

    if(setRS.best_score_ > best_score_rs):
        best_cv_rs = i
        top_classifier_rs = setRS
        best_score_rs = setRS.best_score_
print("Najlepszy klasyfikator : {}".format(str(top_classifier_rs.best_params_)))
print("Wynik {}".format(best_cv_rs))
print("najlepsze cv {}".format(best_score_rs))
print("Dokladność {}".format(top_classifier_rs.score(X_test, y_test)))
