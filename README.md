# Students Dropout Risk EDA Dashboard

## Overview

This Streamlit application provides an exploratory data analysis (EDA) tool for visualizing students' dropout risks. It allows users to upload datasets and create various visualizations to understand the factors influencing student dropouts.

## Features

- **File Upload:** Users can upload CSV, TXT, XLS, or XLSX files to analyze their data.
- **Interactive Visualizations:** Create different types of plots including histograms, box plots, and bar charts based on numerical and categorical data.
- **Dynamic Filtering:** Use the sidebar to filter data by numerical and categorical columns, and choose the type of plots to display.
- **Correlation Heatmap:** Visualize relationships between numeric variables.
- **Scatter Plots:** Explore relationships between specific academic grades and dropout status.

## Installation

To run this app locally, ensure you have Python installed and follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>

2. Install the packages
   pip install streamlit plotly pandas seaborn

3. Run the app with your app's name
   streamlit run app.py

###   Usage
Open your browser and navigate to http://localhost:8501.
Upload a CSV, TXT, XLS, or XLSX file containing student data.
Use the sidebar to select numerical or categorical columns and choose the type of plot you want to visualize.
Explore different relationships between the variables through scatter plots and heatmaps.
Review the statistics of the dropouts and the overall data in tabular format.
Data Format
The dataset should contain the following relevant columns:

Numerical columns: Age at enrollment, Average curricular units grade, Unemployment rate, etc.
Categorical columns: Gender, Scholarship holder, Course, etc.
Target variable: Target (indicating whether a student dropped out or not).
Contributing
Contributions are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Streamlit for an easy-to-use web app framework.
Plotly for powerful data visualization tools.
Pandas and Seaborn for data manipulation and visualization.
css
Copy code

Feel free to adjust any sections to better reflect your project!