import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import datetime
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
    df = pd.read_csv("updated_data.csv")

numeric = ['Age at enrollment', 'Unemployment rate','Inflation rate','GDP','Previous qualification (grade)','Admission grade','Curricular units 1st sem (grade)','Curricular units 2nd sem (grade)']
category =  ['Daytime/evening attendance','Displaced','Educational special needs','Debtor','Tuition fees up to date','Gender','Scholarship holder','International','Marital status','Target','Previous qualification','Nationality','Course','Curricular units 1st sem (credited)','Curricular units 1st sem (enrolled)','Curricular units 1st sem (evaluations)','Curricular units 1st sem (approved)','Curricular units 1st sem (without evaluations)','Curricular units 2nd sem (credited)','Curricular units 2nd sem (enrolled)','Curricular units 2nd sem (evaluations)','Curricular units 2nd sem (approved)','Curricular units 2nd sem (without evaluations)']
for cat in category:
    df[f'{cat}'] = df[f'{cat}'].astype(str)

avg_num = df.groupby('Target')[numeric].mean().reset_index()


st.sidebar.header('Choose the Education status: ')
# Create for target
target = st.sidebar.multiselect('Pick target status', df['Target'].unique())
if not target:
    df2=df.copy()
else:
    df2 = df[df['Target'].isin(target)]

# Create for scholarship holders
sh= st.sidebar.multiselect('Pick scholarship status', df2['Scholarship holder'].unique())
if not target:
    df3=df2.copy()
else:
    df3 = df2[df2['Scholarship holder'].isin(sh)]

# def create_scatter(x='x_axis', y='y_axis', color_encode=False)
col1, col2, col3 = st.columns([0.1,0.45,0.45])
with col1:
    box_date = str(datetime.datetime.now().strftime('%d %B %Y'))
    st.write(f'Last updated: \n {box_date}')
with col2:
    fig = px.histogram(df, x='Target', title= "Target Distribution",
                       width=600,
                       height=400,hover_data=['Target'], template='gridon')
    st.plotly_chart(fig, use_container_width=True)






