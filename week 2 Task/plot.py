import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
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
# dict = {'Histogram':'histogram','Box plot':'box', 'Bar Chart':'bar', 'Scatter Plot':'scatter'}
plots = ['Histogram','Box plot', 'Bar Chart']
numeric = ['Age at enrollment', 'Average curricular units grade','Unemployment rate','Inflation rate','GDP','Previous qualification (grade)','Admission grade','Curricular units 1st sem (grade)','Curricular units 2nd sem (grade)']
numeric_2 =  ['Average curricular units grade','Admission grade_encoded', 'Curricular units 1st sem (grade)','Curricular units 2nd sem (grade)']
category =  ['Daytime/evening attendance','Displaced','Educational special needs','Debtor','Tuition fees up to date','Gender','Scholarship holder','International','Marital status','Target','Previous qualification','Nationality','Course','Curricular units 1st sem (credited)','Curricular units 1st sem (enrolled)','Curricular units 1st sem (evaluations)','Curricular units 1st sem (approved)','Curricular units 1st sem (without evaluations)','Curricular units 2nd sem (credited)','Curricular units 2nd sem (enrolled)','Curricular units 2nd sem (evaluations)','Curricular units 2nd sem (approved)','Curricular units 2nd sem (without evaluations)']
for cat in category:
    df[f'{cat}'] = df[f'{cat}'].astype(str)
df['Tuition fees up to date'] = np.where(df['Tuition fees up to date']=='1', 'Yes','No')
df['Scholarship holder'] = np.where(df['Scholarship holder']=='1', 'Yes','No')

col1, col2, col3= st.columns(3)

st.sidebar.header('Select your Filter (You can only select Numerical column with Histogram or Boxplot OR Categorical column with Bar chart ): ')
# Create for target
num = st.sidebar.multiselect('Select the Numerical column to view: ', numeric)
if not num:
    df2=df.copy()
else:
    df2 = df[num]
# num_2 = st.sidebar.multiselect('Select for Grades Relationship:', ['Average curricular units grade vs Admission grade', 'Curricular units 1st sem (grade) vs Curricular units 2nd sem (grade)'])
# if not num_2:
#     df3=df2.copy()
# else:
#     df3 = df2[num]
cat = st.sidebar.multiselect('Select the Categorical column to view: ', category)
if not cat:
    df3=df2.copy()
else:
    df3 = df2[cat]
plot_type = st.sidebar.multiselect('Select Plot to view: ', plots)




with col1:
    box_date = str(datetime.datetime.now().strftime('%d %B %Y'))
    st.write(f'Last updated: \n {box_date}')
if not num and not plot_type and not cat:
    a=df
    # with col2:
    #     st.subheader("Target Distribution")
    #     fig = px.histogram(df, x='Target',    width=600,
    #                height=400,hover_data=['Target'], template='gridon')
    #     st.plotly_chart(fig,use_container_width=True, height=200)
elif not plot_type and not cat:
    a = num
elif not plot_type and not num:
    a = cat
elif plot_type and cat:
    def create_bar(column):
        st.subheader(f"{c} Distribution")
        fig = px.bar(df, x=column, width=600,
                  height=400,hover_data=[f'{c}'], template='gridon')
        st.plotly_chart(fig,use_container_width=True, height=200)
    for c in cat:
        for p in plot_type:
            if p=='Bar Chart':
                create_bar(column=c)
elif plot_type and num:
    def create_box(num_var):
        st.subheader(f"{num_var} Distribution")
        fig = px.box(df, x=num_var, width=600,
                  height=400,hover_data=[num_var], template='gridon')
        st.plotly_chart(fig,use_container_width=True, height=200)
    for n in num:
        for p in plot_type:
            if p=='Box plot':
                create_box(num_var=n)
            elif p =="Histogram":
                st.subheader(f"{n} Distribution")
                fig = px.histogram(df, x=n, width=600,
                    height=400,hover_data=[n], template='gridon')
                st.plotly_chart(fig,use_container_width=True, height=200)








# Create a tree based on Target, 



data1 = px.scatter(df, x='Average curricular units grade',y='Admission grade_encoded')
data1['layout'].update(title='Relationship between Admission grade and Average curricular units grade',
                        xaxis=dict(title='Average curricular units grade',titlefont=dict(size=20)),yaxis=dict(title='Admission grade',
                                                            titlefont=dict(size=19)))
st.plotly_chart(data1, use_container_width=True)

# Create heatmap using Plotly Express
st.subheader(f"Relationship between Numeric Variables")
fig = px.imshow(
  df[numeric].corr(),
  color_continuous_scale="Inferno_r",
)
st.plotly_chart(fig,use_container_width=True, height=200)


for c in category:
    for n in numeric:
        st.subheader(f"{n} vs {c} Distribution")
        fig = px.box(x=df[n], color=df[c],
                         width=600,
                     height=400,)
        st.plotly_chart(fig,use_container_width=True, height=200)



