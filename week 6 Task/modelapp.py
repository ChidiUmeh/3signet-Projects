import torch
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
label = LabelEncoder()
scaler = StandardScaler()
onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore').set_output(transform='pandas')
import streamlit as st
from app import make_prediction
import pandas as pd


st.set_page_config(page_title="Students Dropout Prediction", page_icon=":bar_chart:",layout="wide")
st.title(" :bar_chart: Students Dropout Risk EDA")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)
f1= st.file_uploader(":file_folder: Upload a file",type=(["csv","txt","xlsx","xls"]))
if f1 is not None:
    filename= f1.name
    st.write(filename)
    school=pd.read_csv(filename) 
else: 
    st.write('upload your students file')




def menu():
    school.rename(columns={'Nacionality' :'Nationality', 'Daytime/evening attendance\t':'Daytime/evening attendance'},inplace=True)
    numeric = [
    'Age at enrollment',
    'Previous qualification (grade)',
    'Admission grade',
    'Curricular units 1st sem (enrolled)',
     'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
     'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Unemployment rate',
    'Inflation rate',
    'GDP']


    categorical = ["Mother's occupation","Father's occupation", 'Marital status',
'    Application mode','Course',"Previous qualification",'Application order',"Mother's qualification","Father's qualification",'Nationality']
    oe = ["Mother's occupation","Father's occupation",
    'Application mode','Course',"Mother's qualification","Father's qualification"]
    le = ['Application order',"Previous qualification",'Marital status','Nationality']
    for c in categorical:
      school[c] = school[c].astype(str)





if __name__ =="__main__":
    menu()

