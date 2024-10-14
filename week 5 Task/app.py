# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from hyperopt import fmin, tpe, hp, Trials
import warnings
import shap
from sklearn.inspection import PartialDependenceDisplay

warnings.filterwarnings('ignore')

# Load the dataset
school_df = pd.read_csv('../week 4 Task/updated_data.csv', index_col=0)
X = school_df.drop('Target_encoded', axis=1)
y = school_df['Target_encoded']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define grid search function
def gridsearch(model, params):
    gridsearch = GridSearchCV(estimator=model, param_grid=params, cv=10, n_jobs=-1)
    gridsearch.fit(X_train, y_train)
    train_pred = gridsearch.predict(X_train)
    test_pred = gridsearch.predict(X_test)
    model_pred_proba = gridsearch.predict_proba(X_test)[:, 1]
    
    print('Training accuracy:', accuracy_score(train_pred, y_train))
    print('Testing accuracy:', accuracy_score(test_pred, y_test))
    scores = cross_val_score(gridsearch, X_train, y_train, cv=10)
    print("Mean cross-validation score:", np.mean(scores))
    print('ROC AUC:', roc_auc_score(y_test, model_pred_proba))
    print("Confusion matrix:\n", confusion_matrix(y_test, test_pred))
    print("Classification report:\n", classification_report(y_test, test_pred))

# Define randomized search function
def randomizedsearch(model, params):
    randomsearch = RandomizedSearchCV(model, params, n_iter=100, scoring='accuracy', cv=10, random_state=42, n_jobs=-1)
    randomsearch.fit(X_train, y_train)
    train_pred = randomsearch.predict(X_train)
    test_pred = randomsearch.predict(X_test)
    model_pred_proba = randomsearch.predict_proba(X_test)[:, 1]
    
    print('Training accuracy:', accuracy_score(train_pred, y_train))
    print('Testing accuracy:', accuracy_score(test_pred, y_test))
    scores = cross_val_score(randomsearch, X_train, y_train, cv=10)
    print("Mean cross-validation score:", np.mean(scores))
    print('ROC AUC:', roc_auc_score(y_test, model_pred_proba))
    print("Confusion matrix:\n", confusion_matrix(y_test, test_pred))
    print("Classification report:\n", classification_report(y_test, test_pred))

# Define hyperopt function
def hyperopt(model, params):
    mdl = model.set_params(**params)
    mdl.fit(X_train, y_train)
    train_pred = mdl.predict(X_train)
    test_pred = mdl.predict(X_test)
    model_pred_proba = mdl.predict_proba(X_test)[:, 1]
    
    print('Training accuracy:', accuracy_score(train_pred, y_train))
    print('Testing accuracy:', accuracy_score(test_pred, y_test))
    scores = cross_val_score(mdl, X_train, y_train, cv=10)
    print("Mean cross-validation score:", np.mean(scores))
    print('ROC AUC:', roc_auc_score(y_test, model_pred_proba))
    print("Confusion matrix:\n", confusion_matrix(y_test, test_pred))
    print("Classification report:\n", classification_report(y_test, test_pred))

# Define learning curve plotting function
def plot_learning_curve(model):
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1, 50))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label="Training scores")
    plt.plot(train_sizes, test_mean, label="Testing scores")
    plt.title(f'{model.__class__.__name__} Learning Curve')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy Score')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

# Logistic Regression
params_logreg = {
    'penalty': ['l1', 'l2', None],
    'multi_class': ['auto', 'ovr', 'multinomial'],
    'C': np.linspace(0.001, 0.1, 100),
    'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
}
logreg = LogisticRegression()
randomizedsearch(logreg, params_logreg)
gridsearch(logreg, params_logreg)
plot_learning_curve(logreg)

# Hyperopt for Logistic Regression
space_logreg = {
    'penalty': hp.choice('penalty', ['l1', 'l2', None]),
    'multi_class': hp.choice('multi_class', ['auto', 'ovr', 'multinomial']),
    'C': hp.uniform('C', 0.001, 0.1),
    'solver': hp.choice('solver', ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'])
}
trials_logreg = Trials()
best_logreg = fmin(fn=lambda params: hyperopt(logreg, params), space=space_logreg, algo=tpe.suggest, max_evals=50)

# Decision Tree
params_dt = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': np.arange(1, 10, 1),
    'class_weight': ['balanced', 'balanced_subsample']
}
dt = DecisionTreeClassifier()
randomizedsearch(dt, params_dt)
gridsearch(dt, params_dt)
plot_learning_curve(dt)

# Hyperopt for Decision Tree
space_dt = {
    'criterion': hp.choice('criterion', ['gini', 'entropy', 'log_loss']),
    'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
    'max_depth': hp.quniform('max_depth', 1, 9, 1),
    'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample'])
}
best_dt = fmin(fn=lambda params: hyperopt(dt, params), space=space_dt, algo=tpe.suggest, max_evals=50)

# SVM
params_svm = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'C': np.arange(1, 10, 1),
    'class_weight': ['dict', 'balanced']
}
svm = SVC(probability=True)
randomizedsearch(svm, params_svm)
gridsearch(svm, params_svm)
plot_learning_curve(svm)

# Hyperopt for SVM
space_svm = {
    'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
    'gamma': hp.choice('gamma', ['scale', 'auto']),
    'C': hp.quniform('C', 1, 10, 1),
    'class_weight': hp.choice('class_weight', ['dict', 'balanced'])
}
best_svm = fmin(fn=lambda params: hyperopt(svm, params), space=space_svm, algo=tpe.suggest, max_evals=50)

# Random Forest
params_rf = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': np.arange(1, 10, 1),
    'n_estimators': np.arange(100, 1000, 100),
    'class_weight': ['balanced', 'balanced_subsample']
}
rf = RandomForestClassifier()
randomizedsearch(rf, params_rf)
gridsearch(rf, params_rf)
plot_learning_curve(rf)

# Hyperopt for Random Forest
space_rf = {
    'criterion': hp.choice('criterion', ['gini', 'entropy', 'log_loss']),
    'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
    'max_depth': hp.quniform('max_depth', 1, 9, 1),
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 100),
    'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample'])
}
best_rf = fmin(fn=lambda params: hyperopt(rf, params), space=space_rf, algo=tpe.suggest, max_evals=50)

# Gradient Boosting
params_gbt = {
    'subsample': np.linspace(0.1, 1, 10),
    'max_features': ['sqrt', 'log2', None],
    'max_depth': np.arange(1, 10, 1),
    'n_estimators': np.arange(100, 1000, 100)
}
gbt = GradientBoostingClassifier()
randomizedsearch(gbt, params_gbt)
gridsearch(gbt, params_gbt)
plot_learning_curve(gbt)

# Stacking Classifier
base_models = [
    ('dt', DecisionTreeClassifier(max_depth=4, criterion='log_loss', class_weight='balanced')),
    ('rf', RandomForestClassifier(n_estimators=300, max_features='sqrt', max_depth=9, criterion='gini', class_weight='balanced')),
    ('svm', SVC(kernel="linear", gamma='auto', C=2, probability=True)),
    ('gb', GradientBoostingClassifier(subsample=0.1, n_estimators=700, max_features='sqrt', max_depth=1))
]

# Define meta-model
meta_model = LogisticRegression(C=0.001, max_iter=300, multi_class='auto', penalty=None, solver='newton-cg')

# Create the stacking classifier
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# Fit the stacking model
stacking_model.fit(X_train, y_train)
train_pred = stacking_model.predict(X_train)
test_pred = stacking_model.predict(X_test)
model_pred_proba = stacking_model.predict_proba(X_test)[:, 1]

print('Training accuracy:', accuracy_score(train_pred, y_train))
print('Testing accuracy:', accuracy_score(test_pred, y_test))
scores = cross_val_score(stacking_model, X_train, y_train, cv=10)
print("Mean cross-validation score:", np.mean(scores))
print('ROC AUC:', roc_auc_score(y_test, model_pred_proba))
print("Confusion matrix:\n", confusion_matrix(y_test, test_pred))
print("Classification report:\n", classification_report(y_test, test_pred))

# Feature importances from Random Forest
importances = rf.feature_importances_
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Plot feature importances from Random Forest
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances from Random Forest')
plt.grid(axis='x')
plt.show()

# SHAP Values for Decision Tree
explainer = shap.TreeExplainer(dt)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)

# Force plot for the first instance
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_test.iloc[0])

# Partial Dependence Plots
disp = PartialDependenceDisplay.from_estimator(rf, X_train, features=X.columns, grid_resolution=50)
plt.suptitle('Partial Dependence Plots')
plt.show()