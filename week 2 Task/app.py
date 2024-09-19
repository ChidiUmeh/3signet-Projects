import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Students Dropout Risks", page_icon=":bar_chart:",layout="wide")
st.title(" :bar_chart: Students Dropout Risk EDA")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

f1= st.file_uploader(":file_folder: Upload a file",type=(["csv","txt","xlsx","xls"]))
if f1 is not None:
    filename= f1.name
    st.write(filename)
    df=pd.read_csv(filename) 
else:
    os.chdir(r'C:/Users/USER/OneDrive/Desktop/3signet/week 2 Task')
    df = pd.read_csv("cleaned_data.csv")

numeric = ['Age at enrollment', 'Unemployment rate','Inflation rate','GDP','Previous qualification (grade)','Admission grade','Curricular units 1st sem (grade)','Curricular units 2nd sem (grade)']
category =  ['Daytime/evening attendance','Displaced','Educational special needs','Debtor','Tuition fees up to date','Gender','Scholarship holder','International','Marital status','Target','Previous qualification','Nationality','Course','Curricular units 1st sem (credited)','Curricular units 1st sem (enrolled)','Curricular units 1st sem (evaluations)','Curricular units 1st sem (approved)','Curricular units 1st sem (without evaluations)','Curricular units 2nd sem (credited)','Curricular units 2nd sem (enrolled)','Curricular units 2nd sem (evaluations)','Curricular units 2nd sem (approved)','Curricular units 2nd sem (without evaluations)']
for cat in category:
    df[f'{cat}'] = df[f'{cat}'].astype(str)
df.info()

