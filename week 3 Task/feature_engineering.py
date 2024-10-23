import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class FeatureEngineering:
    def __init__(self, data):
        self.data = data

    def preprocess_data(self):
        # Feature Engineering
        self.data['Curricular units (Average grade)'] = (
            (self.data['Curricular units 1st sem (grade)'] + 
             self.data['Curricular units 2nd sem (grade)']) / 2
        )
        self.data['Total Curricular units (enrolled)'] = (
            self.data['Curricular units 1st sem (enrolled)'] + 
            self.data['Curricular units 2nd sem (enrolled)']
        )
        self.data['Total Curricular units (evaluations)'] = (
            self.data['Curricular units 1st sem (evaluations)'] + 
            self.data['Curricular units 2nd sem (evaluations)']
        )
        self.data['Total Curricular units (approved)'] = (
            self.data['Curricular units 1st sem (approved)'] + 
            self.data['Curricular units 2nd sem (approved)']
        )
        self.data['Completion_Rate_1st'] = (
            self.data['Curricular units 1st sem (approved)'] / 
            self.data['Curricular units 1st sem (enrolled)']
        )
        self.data['Completion_Rate_2nd'] = (
            self.data['Curricular units 2nd sem (approved)'] / 
            self.data['Curricular units 2nd sem (enrolled)']
        )

        # Interaction Features
        self.data['Completion_Grade_Interaction'] = (
            self.data['Completion_Rate_1st'] * self.data['Admission grade']
        )
        self.data['Completion_Age_Interaction'] = (
            self.data['Completion_Rate_1st'] * self.data['Age at enrollment']
        )
        self.data['Grade_Interaction'] = (
            self.data['Admission grade'] * self.data['Previous qualification (grade)']
        )
        self.data['Previous_Age_Interaction'] = (
            self.data['Previous qualification (grade)'] * self.data['Age at enrollment']
        )

        # Grouping Age
        self.data['Age_cut'] = pd.qcut(self.data['Age at enrollment'], 6)
        self.data['Grouped Age at enrollment'] = pd.qcut(
            self.data['Age at enrollment'], q=6, labels=[1, 2, 3, 4, 5, 6]
        )

        # Log transformation for skewed data
        skew_data = [
            'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)',
            'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (enrolled)',
            'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
            'Completion_Rate_2nd', 'Completion_Rate_1st', 
            'Total Curricular units (approved)', 'Total Curricular units (evaluations)',
            'Total Curricular units (enrolled)', 'Curricular units (Average grade)', 
            'Completion_Grade_Interaction', 'Completion_Age_Interaction', 
            'Grade_Interaction', 'Previous_Age_Interaction'
        ]

        for var in skew_data:
            self.data[var] = np.log(self.data[var] + 1)

        # Standardize numeric features
        numeric_features = [
            'Age at enrollment', 'Previous qualification (grade)', 'Admission grade',
            'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)',
            'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
            'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)',
            'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
            'Unemployment rate', 'Inflation rate', 'GDP', 'Completion_Rate_2nd',
            'Completion_Rate_1st', 'Total Curricular units (approved)',
            'Total Curricular units (evaluated)', 'Total Curricular units (enrolled)',
            'Curricular units (Average grade)', 'Completion_Grade_Interaction', 
            'Completion_Age_Interaction', 'Grade_Interaction', 'Previous_Age_Interaction'
        ]

        scaler = StandardScaler()
        self.data[numeric_features] = scaler.fit_transform(self.data[numeric_features])

    def get_processed_data(self):
        return self.data


# Example usage
if __name__ == "__main__":
    # Load new data
    new_data_path = 'path_to_new_data.csv'  # Replace with your actual path
    new_data = pd.read_csv(new_data_path)

    # Initialize feature engineering
    feature_engineering = FeatureEngineering(new_data)
    feature_engineering.preprocess_data()

    # Get processed data
    processed_data = feature_engineering.get_processed_data()
    
    # Save processed data to CSV
    processed_data.to_csv('processed_new_data.csv', index=False)
    print("Feature engineering completed. Processed data saved to 'processed_new_data.csv'.")
