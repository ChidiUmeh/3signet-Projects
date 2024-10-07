# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import pickle
import warnings

warnings.filterwarnings('ignore')

class ModelDevelopmentPipeline:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, index_col=0)
        self.X, self.y = self.preprocess_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, stratify=self.y, random_state=42)

    def preprocess_data(self):
        self.data['Target_encoded'] = np.where(self.data['Target_encoded'] >= 1, 0, 1)
        X = self.data.drop('Target_encoded', axis=1).round(2)
        y = self.data['Target_encoded']
        return X, y

    def train_and_evaluate_model(self, model, params, model_name):
        # Hyperparameter tuning
        grid_search = RandomizedSearchCV(model, params, n_iter=100, scoring='accuracy', cv=10, random_state=42, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_

        # Cross-validation
        scores = cross_val_score(best_model, self.X_train, self.y_train, cv=10)
        print(f"{model_name} cross-validation scores: {scores}")
        print(f"Mean cross-validation score: {np.mean(scores)}")

        # Fitting the model
        best_model.fit(self.X_train, self.y_train)
        train_pred = best_model.predict(self.X_train)
        test_pred = best_model.predict(self.X_test)
        test_pred_proba = best_model.predict_proba(self.X_test)[:, 1]

        # Evaluation metrics
        print(f"{model_name} training accuracy: {accuracy_score(train_pred, self.y_train)}")
        print(f"{model_name} testing accuracy: {accuracy_score(test_pred, self.y_test)}")
        self.plot_roc_curve(self.y_test, test_pred_proba, model_name)
        self.print_classification_report(test_pred)

        return best_model

    def plot_roc_curve(self, y_test, pred_proba, model_name):
        fpr, tpr, _ = roc_curve(y_test, pred_proba)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC curve')
        plt.show()
        print(f"ROC AUC score: {roc_auc_score(y_test, pred_proba)}")

    def print_classification_report(self, predictions):
        print("Confusion matrix:\n", confusion_matrix(self.y_test, predictions))
        print("Classification report:\n", classification_report(self.y_test, predictions))

    def plot_learning_curve(self, model):
        train_sizes, train_scores, test_scores = learning_curve(model, self.X_train, self.y_train, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1, 50), verbose=1)
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label="Training scores")
        plt.plot(train_sizes, test_mean, label="Testing scores")
        plt.title(f'Learning curve for {model}')
        plt.xlabel('Training size')
        plt.ylabel('Accuracy score')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

    def save_model(self, model, model_name):
        with open(f'{model_name}_model.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)

def main():
    pipeline = ModelDevelopmentPipeline('updated_data.csv')

    # Logistic Regression
    logreg_params = {
        'penalty': ['l1', 'l2', None],
        'multi_class': ['auto', 'ovr', 'multinomial'],
        'C': np.arange(0.001, 0.1),
        'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
    }
    logreg_model = pipeline.train_and_evaluate_model(LogisticRegression(random_state=42), logreg_params, 'Logistic Regression')
    pipeline.save_model(logreg_model, 'LogisticRegression')

    # Decision Tree
    dt_params = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': np.arange(1, 10, 1),
        'class_weight': ['balanced', 'balanced_subsample']
    }
    dt_model = pipeline.train_and_evaluate_model(DecisionTreeClassifier(), dt_params, 'Decision Tree')
    pipeline.save_model(dt_model, 'DecisionTree')

    # Support Vector Machine
    svm_params = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        'gamma': ['scale', 'auto'],
        'C': np.arange(1, 10, 1),
        'class_weight': ['dict', 'balanced']
    }
    svm_model = pipeline.train_and_evaluate_model(SVC(probability=True), svm_params, 'SVM')
    pipeline.save_model(svm_model, 'SVM')

    # Random Forest
    rf_params = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': np.arange(1, 10, 1),
        'n_estimators': np.arange(100, 1000, 100),
        'class_weight': ['balanced', 'balanced_subsample']
    }
    rf_model = pipeline.train_and_evaluate_model(RandomForestClassifier(), rf_params, 'Random Forest')
    pipeline.save_model(rf_model, 'RandomForest')

    # Gradient Boosting
    gbt_params = {
        'subsample': np.arange(0.1, 1, 0.1),
        'max_features': ['sqrt', 'log2', None],
        'max_depth': np.arange(1, 10, 1),
        'n_estimators': np.arange(100, 1000, 100)
    }
    gbt_model = pipeline.train_and_evaluate_model(GradientBoostingClassifier(), gbt_params, 'Gradient Boosting')
    pipeline.save_model(gbt_model, 'GradientBoosting')

if __name__ == '__main__':
    main()





