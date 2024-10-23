import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import torch.optim as optim

warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, data_path, target_column='Target'):
        self.data = pd.read_csv(data_path, index_col=0)
        self.X = self.data.drop(target_column, axis=1).values
        self.y = self.data[target_column].values
        self.models = {}
        self.results = {}

    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, stratify=self.y, random_state=random_state
        )

    def train_logistic_regression(self):
        logreg = LogisticRegression()
        scores = cross_val_score(logreg, self.X_train, self.y_train, cv=10)
        print("Logistic Regression Cross-validation scores:", scores)
        print("Mean cross-validation score:", np.mean(scores))
        
        logreg.fit(self.X_train, self.y_train)
        self.evaluate_model(logreg, 'Logistic Regression')
        self.models['Logistic Regression'] = logreg

    def train_decision_tree(self):
        dt = DecisionTreeClassifier()
        scores = cross_val_score(dt, self.X_train, self.y_train, cv=10)
        print("Decision Tree Cross-validation scores:", scores)
        print("Mean cross-validation score:", np.mean(scores))
        
        dt.fit(self.X_train, self.y_train)
        self.evaluate_model(dt, 'Decision Tree')
        self.models['Decision Tree'] = dt

    def train_svc(self):
        svm = SVC(probability=True)
        scores = cross_val_score(svm, self.X_train, self.y_train, cv=10)
        print("SVC Cross-validation scores:", scores)
        print("Mean cross-validation score:", np.mean(scores))
        
        svm.fit(self.X_train, self.y_train)
        self.evaluate_model(svm, 'SVC')
        self.models['SVC'] = svm

    def train_random_forest(self):
        rf = RandomForestClassifier()
        scores = cross_val_score(rf, self.X_train, self.y_train, cv=10)
        print("Random Forest Cross-validation scores:", scores)
        print("Mean cross-validation score:", np.mean(scores))
        
        rf.fit(self.X_train, self.y_train)
        self.evaluate_model(rf, 'Random Forest')
        self.models['Random Forest'] = rf

    def train_gradient_boosting(self):
        gbt = GradientBoostingClassifier()
        scores = cross_val_score(gbt, self.X_train, self.y_train, cv=10)
        print("Gradient Boosting Cross-validation scores:", scores)
        print("Mean cross-validation score:", np.mean(scores))
        
        gbt.fit(self.X_train, self.y_train)
        self.evaluate_model(gbt, 'Gradient Boosting')
        self.models['Gradient Boosting'] = gbt

    def train_xgboost(self):
        xgb = XGBClassifier()
        scores = cross_val_score(xgb, self.X_train, self.y_train, cv=10)
        print("XGBoost Cross-validation scores:", scores)
        print("Mean cross-validation score:", np.mean(scores))
        
        xgb.fit(self.X_train, self.y_train)
        self.evaluate_model(xgb, 'XGBoost')
        self.models['XGBoost'] = xgb

    def evaluate_model(self, model, model_name):
        train_pred = model.predict(self.X_train)
        test_pred = model.predict(self.X_test)
        test_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        print(f'{model_name} Training accuracy: ', accuracy_score(train_pred, self.y_train))
        print(f'{model_name} Testing accuracy: ', accuracy_score(test_pred, self.y_test))
        
        fpr, tpr, _ = roc_curve(self.y_test, test_pred_proba)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC curve')
        plt.show()
        
        print("Confusion matrix:\n", confusion_matrix(self.y_test, test_pred))
        print("Classification report:\n", classification_report(self.y_test, test_pred))
        
        self.results[model_name] = {
            'confusion_matrix': confusion_matrix(self.y_test, test_pred),
            'classification_report': classification_report(self.y_test, test_pred)
        }
        self.plot_learning_curve(model)

    def plot_learning_curve(self, model):
        train_sizes, train_scores, test_scores = learning_curve(
            model, self.X_train, self.y_train, cv=10, scoring='accuracy', n_jobs=-1,
            train_sizes=np.linspace(0.01, 1, 50), verbose=1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label="Training scores")
        plt.plot(train_sizes, test_mean, label="Testing scores")
        plt.title(f'{model.__class__.__name__} Learning curve')
        plt.xlabel('Training size')
        plt.ylabel('Accuracy score')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

    def train_neural_network(self, input_size=29):
        class NeuralNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(input_size, 25)
                self.fc2 = nn.Linear(25, 20)
                self.fc3 = nn.Linear(20, 15)
                self.fc4 = nn.Linear(15, 10)
                self.out = nn.Linear(10, 2)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
                return self.out(x)

        model = NeuralNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Convert data to tensors
        X_train_tensor = torch.FloatTensor(self.X_train)
        y_train_tensor = torch.LongTensor(self.y_train)

        # Train the model
        epochs = 100
        losses = []

        for epoch in range(epochs):
            model.train()
            y_pred = model(X_train_tensor)
            loss = criterion(y_pred, y_train_tensor)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch: {epoch} Loss: {loss.item()}')

        plt.plot(range(epochs), losses)
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.title('Training Loss')
        plt.show()

        # Save the model
        torch.save(model.state_dict(), 'my_pytorch_model.pt')

    def save_models(self):
        for model_name, model in self.models.items():
            with open(f'{model_name.replace(" ", "_").lower()}_model.pkl', 'wb') as file:
                pickle.dump(model, file)
        print("Models saved successfully.")

if __name__ == "__main__":
    trainer = ModelTrainer('../week 3 Task/new_data.csv')
    trainer.split_data()
    trainer.train_logistic_regression()
    trainer.train_decision_tree()
    trainer.train_svc()
    trainer.train_random_forest()
    trainer.train_gradient_boosting()
    trainer.train_xgboost()
    trainer.train_neural_network()
    trainer.save_models()
