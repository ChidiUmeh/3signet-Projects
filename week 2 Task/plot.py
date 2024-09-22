import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Students Dropout Risks", page_icon=":bar_chart:",layout="wide")
st.title(" :bar_chart: Students Dropout Risk EDA")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)

f1= st.file_uploader(":file_folder: Upload a file",type=(["csv","txt","xlsx","xls"]))
if f1 is not None:
    filename= f1.name
    st.write(filename)
    df=pd.read_csv(filename) 
else:
    # os.chdir(r"C:/Users/USER/OneDrive/Desktop/3signet/week 2 Task")
    
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

c1,c2,c3=st.columns(3)
col1, col2, col3,col4= st.columns(4)



st.sidebar.header('Select your Filter: ')
# Create for target
num = st.sidebar.multiselect('Select the Numerical column to view: ', numeric)
if not num:
    df2=df.copy()
else:
    df2 = df[df.columns].isin(num)
# num_2 = st.sidebar.multiselect('Select for Grades Relationship:', ['Average curricular units grade vs Admission grade', 'Curricular units 1st sem (grade) vs Curricular units 2nd sem (grade)'])
# if not num_2:
#     df3=df2.copy()
# else:
#     df3 = df2[num]
cat = st.sidebar.multiselect('Select the Categorical column to view: ', category)
if not cat:
    df3=df2.copy()
else:
    df3 = df2[df2.columns].isin(cat)
plot_type = st.sidebar.multiselect('Select Plot to view: ', plots)
# rel = st.sidebar.multiselect('Select to view Relationship: ', ['Numeric vs Dropouts', "Between Numerics"])
if df.empty:
    st.warning("Not available")



with col1:
    if not num and not plot_type and not cat:
        filtered_df=df
    # with col2:
    #     st.subheader("Target Distribution")
    #     fig = px.histogram(df, x='Target',    width=600,
    #                height=400,hover_data=['Target'], template='gridon')
    #     st.plotly_chart(fig,use_container_width=True, height=200)
    elif not plot_type and not cat:
        flitered_df= df[df.columns].isin(num)
    elif not num and not plot_type:
        filtered_df = df[df.columns].isin(cat)
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
                else:
                    st.warning("You can only select bar chart for categorical column")
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
                else:
                    st.warning("You can't select bar chart for Numeric columns")
                
    elif plot_type and num and cat:
        def create_box2(var_1,var_2):
            st.subheader(f"{var_1} vs {var_2} Distribution")
            fig = px.box(df, x=var_1, color=var_2,
                         width=600,
                     height=400,)
            st.plotly_chart(fig,use_container_width=True, height=200)
        for c in cat:
           for n in num:
                for p in plot_type:
                    if p=='Box plot':
                        create_box2(var_1=n,var_2=c)

# Target proportions
# proportions = df['Target'].value_counts().reset_index()
all = round((df['Target']=='Dropout').mean()*100,2)
# with col2:
#     st.subheader('Proportions of the Students')
#     for i in proportions:
#         st.subheader(f'{proportions[i]}')
with col4:
    box_date = str(datetime.datetime.now().strftime('%d %B %Y'))
    st.write(f'Last updated: \n {box_date}')
with col3:
    st.subheader('Statistics of the Dropouts')
    fig = px.pie(df,'Target',)

    st.plotly_chart(fig,use_container_width=True, height=200)
with col2:
    st.subheader(f"Admission grades Distribution")
    fig = px.box(df,x='Average curricular units grade',color='Target',
                         width=600,
                     height=400,)
    st.plotly_chart(fig,use_container_width=True, height=200)


with col1:
        st.subheader(f"Distribution Average curricular units grades")
        fig = px.bar(df,x='Target', y='Admission grade',
                         width=600,
                     height=400,)
        st.plotly_chart(fig,use_container_width=True, height=200)    




# chart1,chart2,chart3=st.columns(3)
# with chart1:




    # data1 = px.scatter(df, x='Average curricular units grade',y='Admission grade_encoded')
    # data1['layout'].update(title='Relationship between Admission grade and Average curricular units grade',
    #                     xaxis=dict(title='Average curricular units grade',titlefont=dict(size=20)),yaxis=dict(title='Admission grade',
    #                                                         titlefont=dict(size=19)))

    # for c in category:
#     for n in numeric:
#         st.subheader(f"{n} vs {c} Distribution")
#         fig = px.box(x=df[n], color=df[c],
#                          width=600,
#                      height=400,)
#         st.plotly_chart(fig,use_container_width=True, height=200)



data1 = px.scatter(df, x='Average curricular units grade',y='Admission grade_encoded', color='Target')
data1['layout'].update(title='Relationship between Admission grade and Average curricular units grade',
                        xaxis=dict(title='Average curricular units grade',titlefont=dict(size=20)),yaxis=dict(title='Admission grade',
                                                            titlefont=dict(size=19)))
st.plotly_chart(data1, use_container_width=True)

data1 = px.scatter(df, x='Curricular units 1st sem (grade)',y='Curricular units 2nd sem (grade)', color='Target')
data1['layout'].update(title='Relationship between Curricular units 1st sem (grade) and Curricular units 2nd sem (grade)',
                        xaxis=dict(title='Average curricular units grade',titlefont=dict(size=20)),yaxis=dict(title='Admission grade',
                                                            titlefont=dict(size=19)))
st.plotly_chart(data1, use_container_width=True)







# Create a tree based on Target, 




# Create heatmap using Plotly Express
st.subheader(f"Relationship between Numeric Variables")
fig = px.imshow(
  df[numeric].corr(),
  color_continuous_scale="Inferno_r",
)
st.plotly_chart(fig,use_container_width=True, height=200)


# for c in category:
#     for n in numeric:
#         st.subheader(f"{n} vs {c} Distribution")
#         fig = px.box(x=df[n], color=df[c],
#                          width=600,
#                      height=400,)
#         st.plotly_chart(fig,use_container_width=True, height=200)

 
st.dataframe(df.iloc[:,:37])
