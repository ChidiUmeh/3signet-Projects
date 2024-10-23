# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import  GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV,GridSearchCV,cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import shap
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

class ModelTrainer:
    def __init__(self, model, params=None):
        self.model = model
        self.params = params
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, filepath):
        df = pd.read_csv(filepath, index_col=0)
        X = df.drop('Target', axis=1)
        y = df['Target']
        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    def grid_search(self):
        gridsearch = GridSearchCV(estimator=self.model, param_grid=self.params, cv=10, n_jobs=-1)
        gridsearch.fit(self.X_train, self.y_train)
        self._evaluate_model(gridsearch)

    def randomized_search(self):
        randomsearch = RandomizedSearchCV(self.model, self.params, n_iter=100, scoring='accuracy', cv=10, random_state=42, n_jobs=-1)
        randomsearch.fit(self.X_train, self.y_train)
        self._evaluate_model(randomsearch)

    def _evaluate_model(self, model):
        train_pred = model.predict(self.X_train)
        test_pred = model.predict(self.X_test)
        model_pred_proba = model.predict_proba(self.X_test)[:, 1]

        print('Training accuracy:', accuracy_score(self.y_train, train_pred))
        print('Testing accuracy:', accuracy_score(self.y_test, test_pred))
        print("Mean cross-validation score:", np.mean(cross_val_score(model, self.X_train, self.y_train, cv=10)))
        print('ROC AUC:', roc_auc_score(self.y_test, model_pred_proba))
        print("Confusion matrix:\n", confusion_matrix(self.y_test, test_pred))
        print("Classification report:\n", classification_report(self.y_test, test_pred))
        print("Best parameters:", model.best_params_)

    def plot_learning_curve(self):
        train_sizes, train_scores, test_scores = learning_curve(self.model, self.X_train, self.y_train, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1, 50), verbose=1)
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label="Training scores")
        plt.plot(train_sizes, test_mean, label="Testing scores")
        plt.title(f'{self.model.__class__.__name__} Learning Curve')
        plt.xlabel('Training size')
        plt.ylabel('Accuracy score')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

    def hyperparameter_tuning(self, space, max_evals=80):
        def objective(space):
            model = self.model.set_params(**space)
            model.fit(self.X_train, self.y_train)
            score = np.mean(cross_val_score(model, self.X_train, self.y_train, cv=10))
            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        print("Best hyperparameters:", best)

    def explain_model(self):
        # SHAP analysis
        explainer = shap.Explainer(self.model)
        shap_values = explainer(self.X_train)
        shap.summary_plot(shap_values, self.X_train)

# Random Forest Classifier
params_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
}

space_rf = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 100),
    'max_depth': hp.choice('max_depth', [None, 10, 20, 30]),
    'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4]),
    'min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
    'max_features': hp.choice('max_features', ['sqrt', 'log2'])
}

rf_model = ModelTrainer(RandomForestClassifier())
X, y = rf_model.load_data('your_data.csv')  # Update with your data file path
rf_model.split_data(X, y)
rf_model.randomized_search()
rf_model.grid_search()
rf_model.plot_learning_curve()
rf_model.hyperparameter_tuning(space_rf)

# Gradient Boosting Classifier
params_gbt = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.5, 0.7, 1.0],
    'max_features': ['auto', 'sqrt', 'log2'],
}

space_gbt = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 100),
    'learning_rate': hp.choice('learning_rate', [0.01, 0.1, 0.2]),
    'max_depth': hp.choice('max_depth', [3, 5, 7]),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 4, 1),
    'subsample': hp.choice('subsample', [0.5, 0.7, 1.0]),
    'max_features': hp.choice('max_features', ['sqrt', 'log2'])
}

gbt_model = ModelTrainer(GradientBoostingClassifier())
gbt_model.split_data(X, y)
gbt_model.randomized_search()
gbt_model.grid_search()
gbt_model.plot_learning_curve()
gbt_model.hyperparameter_tuning(space_gbt)

# XGBoost Classifier
params_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 2, 3, 4],
    'subsample': [0.5, 0.6, 0.7, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
}

space_xgb = {
    'n_estimators': hp.choice('n_estimators', [100, 200, 300]),
    'learning_rate': hp.choice('learning_rate', [0.01, 0.1, 0.2]),
    'max_depth': hp.choice('max_depth', [3, 5, 7, 9]),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'subsample': hp.choice('subsample', [0.5, 0.6, 0.7, 0.8, 1.0]),
    'colsample_bytree': hp.choice('colsample_bytree', [0.5, 0.6, 0.7, 0.8, 1.0]),
    'gamma': hp.choice('gamma', [0, 0.1, 0.2, 0.3])
}

xgb_model = ModelTrainer(XGBClassifier())
xgb_model.split_data(X, y)
xgb_model.randomized_search()
xgb_model.grid_search()
xgb_model.plot_learning_curve()
xgb_model.hyperparameter_tuning(space_xgb)

# Explanation for the XGBoost model
xgb_model.explain_model()

# Create a model class that inherits the nn.Module
class Model(nn.Module):
    def __init__(self, in_features=74, h1=50, h2=30, h3=20, h4=10, output_features=2):
        super().__init__()  # Instantiate our nn.Module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.out = nn.Linear(h4, output_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.out(x)
        return x

# Pick a random seed for randomization
torch.manual_seed(41)

# Create an instance of model
model = Model()

# Convert X features to float tensors
X_train = torch.FloatTensor(X_train.values)
X_test = torch.FloatTensor(X_test.values)

# Convert y labels to tensors long(int)
y_train = torch.LongTensor(y_train.values)
y_test = torch.LongTensor(y_test.values)

# Set the criterion of model to measure the error
criterion = nn.CrossEntropyLoss()

# Set the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Get unique values and their counts
unique_values, counts = torch.unique(y_test, return_counts=True)

# Display results
for value, count in zip(unique_values, counts):
    print(f'Value: {value.item()}, Count: {count.item()}')

# Train our model
epochs = 50
losses = []

for i in range(epochs):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.detach().numpy())

    if i % 10 == 0:
        print(f'Epoch: {i} Loss: {loss}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Graph
plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('Epochs')

# Evaluate Model on Test Data Set
model.eval()
with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)

print(f'Loss: {loss}')

correct = 0
num = []
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        num.append(y_val.argmax().item())

        print(f'{i + 1}. {str(y_val)} \t {y_test[i]}')
        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(f'\n{correct} out of {len(y_test)} = {100 * correct / len(y_test)}% correct')

print("Confusion matrix:\n", confusion_matrix(y_test, num))
print("Classification report:\n", classification_report(y_test, num))

torch.save(model.state_dict(), 'my_pytorch_model.pt')

# Stacking and blending models
base_models = [
    ('dt', DecisionTreeClassifier(criterion='entropy', max_depth=7, max_features='sqrt', min_samples_leaf=8, min_samples_split=8, splitter='best')),
    ('rf', RandomForestClassifier(max_depth=None, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, n_estimators=600, random_state=42)),
    ('svm', SVC(probability=True, C=20, gamma=0.001, random_state=42)),
    ('gbt', GradientBoostingClassifier(n_estimators=500, min_samples_split=6, min_samples_leaf=5, learning_rate=0.1, max_depth=5, max_features='log2', subsample=1, random_state=42)),
    ('xgb', XGBClassifier(colsample_bytree=1.0, gamma=0.1, learning_rate=0.1, max_depth=9, min_child_weight=1.0, n_estimators=200, subsample=0.6, random_state=42))
]

# Define meta-model
meta_model = LogisticRegression(C=0.001, class_weight=None, max_iter=800, multi_class='auto', penalty=None, solver='saga', random_state=42)

# Create the stacking classifier
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

# Fit the stacking model
stacking_model.fit(X_train, y_train)
train_pred = stacking_model.predict(X_train)
test_pred = stacking_model.predict(X_test)
model_pred_proba = stacking_model.predict_proba(X_test)[:, 1]

print('training accuracy: ', accuracy_score(train_pred, y_train))
print('testing accuracy: ', accuracy_score(test_pred, y_test))
scores = cross_val_score(stacking_model, X_train, y_train, cv=10)  # 10-fold cross-validation
print("Mean cross-validation score:", np.mean(scores))
print('roc_auc: ', roc_auc_score(y_test, model_pred_proba))
print("Confusion matrix:\n", confusion_matrix(y_test, test_pred))
print("Classification report:\n", classification_report(y_test, test_pred))

# Blending predictions
preds_model_1 = base_models[0][1].predict(X_test)
preds_model_2 = base_models[1][1].predict(X_test)
preds_model_3 = base_models[2][1].predict(X_test)
preds_model_4 = base_models[3][1].predict(X_test)
preds_model_5 = base_models[4][1].predict(X_test)

# Create a DataFrame to hold the predictions
preds_df = pd.DataFrame({
    'model_1': preds_model_1,
    'model_2': preds_model_2,
    'model_3': preds_model_3,
    'model_4': preds_model_4,
    'model_5': preds_model_5,
})

# Use a logistic regression model to blend the predictions
blender = LogisticRegression(C=0.001, class_weight=None, max_iter=800, multi_class='auto', penalty=None, solver='saga', random_state=42)
blender.fit(preds_df, y_test)

# Final prediction on the holdout set
final_preds = blender.predict(preds_df)

# Evaluate the final model
accuracy = accuracy_score(y_test, final_preds)
print(f"Blended model accuracy: {accuracy:.4f}")
scores = cross_val_score(blender, preds_df, y_test, cv=10)  # 10-fold cross-validation
print("Mean cross-validation score:", np.mean(scores))
print("Confusion matrix:\n", confusion_matrix(y_test, final_preds))
print("Classification report:\n", classification_report(y_test, final_preds))

# Create a voting classifier with the optimized base models
voting_clf = VotingClassifier(estimators=[
    ('rf', RandomForestClassifier(max_depth=None, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, n_estimators=600, random_state=42)),
    ('svc', SVC(probability=True, C=20, gamma=0.001, random_state=42)),
    ('logreg', LogisticRegression(C=0.001, class_weight=None, max_iter=800, multi_class='auto', penalty=None, solver='saga', random_state=42)),
    ('dt', DecisionTreeClassifier(criterion='entropy', max_depth=7, max_features='sqrt', min_samples_leaf=8, min_samples_split=8, splitter='best', random_state=42)),
    ('xgb', XGBClassifier(colsample_bytree=1.0, gamma=0.1, learning_rate=0.1, max_depth=9, min_child_weight=1.0, n_estimators=200, subsample=0.6, random_state=42)),
    ('gbt', GradientBoostingClassifier(n_estimators=500, min_samples_split=6, min_samples_leaf=5, learning_rate=0.1, max_depth=5, max_features='log2', subsample=1, random_state=42))
], voting='soft')  # Use soft voting for probability-based predictions

# Train the voting classifier
voting_clf.fit(X_train, y_train)

train_pred = voting_clf.predict(X_train)
test_pred = voting_clf.predict(X_test)
model_pred_proba = voting_clf.predict_proba(X_test)[:, 1]
print('training accuracy: ', accuracy_score(train_pred, y_train))
print('testing accuracy: ', accuracy_score(test_pred, y_test))
scores = cross_val_score(voting_clf, X_train, y_train, cv=10)  # 10-fold cross-validation
print("Mean cross-validation score:", np.mean(scores))
print('roc_auc: ', roc_auc_score(y_test, model_pred_proba))
print("Confusion matrix:\n", confusion_matrix(y_test, test_pred))
print("Classification report:\n", classification_report(y_test, test_pred))

# Save the models
with open('votingclassifier_model.pkl', 'wb') as f:
    pickle.dump(voting_clf, f)

with open('blender_model.pkl', 'wb') as f:
    pickle.dump(blender, f)

with open('stacking_model.pkl', 'wb') as f:
    pickle.dump(stacking_model, f)

# Train a Random Forest model
rf_model = RandomForestClassifier(max_depth=None, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, n_estimators=600, random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importances
importances = rf_model.feature_importances_
rf_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
rf_importances = rf_importances.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(20, 20))
plt.barh(rf_importances['Feature'], rf_importances['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.grid(axis='x')
plt.show()

rf_col = rf_importances[rf_importances['Importance'] >= 0.01]['Feature'].to_list()

dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=7, max_features='sqrt', min_samples_leaf=8, min_samples_split=8, splitter='best', random_state=42)
dt_model.fit(X_train, y_train)

# Get feature importances
importances = dt_model.feature_importances_
dt_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
dt_importances = dt_importances.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(20, 20))
plt.barh(dt_importances['Feature'], dt_importances['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances from Decision Tree')
plt.grid(axis='x')
plt.show()

dt_col = dt_importances[dt_importances['Importance'] >= 0.01]['Feature'].to_list()

xgb = XGBClassifier(colsample_bytree=1.0, gamma=0.1, learning_rate=0.1, max_depth=9, min_child_weight=1.0, n_estimators=200, subsample=0.6, random_state=42)
xgb.fit(X_train, y_train)

# Get feature importances
importances = xgb.feature_importances_
xgb_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
xgb_importances = xgb_importances.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(20, 20))
plt.barh(xgb_importances['Feature'], xgb_importances['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances from XGBoost')
plt.grid(axis='x')
plt.show()

xgb_col = xgb_importances[xgb_importances['Importance'] >= 0.01]['Feature'].to_list()

gbt = GradientBoostingClassifier(n_estimators=500, min_samples_split=6, min_samples_leaf=5, learning_rate=0.1, max_depth=5, max_features='log2', subsample=1, random_state=42)
gbt.fit(X_train, y_train)

# Get feature importances
importances = gbt.feature_importances_
gbt_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
gbt_importances = gbt_importances.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(20, 20))
plt.barh(gbt_importances['Feature'], gbt_importances['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances from Gradient Boosting')
plt.grid(axis='x')
plt.show()
