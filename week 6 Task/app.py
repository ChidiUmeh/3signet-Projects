import streamlit as st
import pandas as pd
import numpy as np
from predict_page import show_predict_page, label, onehot, new_model

show_predict_page()

# categorical = ["Mother's occupation","Father's occupation", 'Marital status',
# 'Application mode','Course',"Previous qualification",'Application order',"Mother's qualification","Father's qualification",'Nationality']
# oe = ["Mother's occupation","Father's occupation",
# 'Application mode','Course',"Mother's qualification","Father's qualification"]
# le = ['Application order',"Previous qualification",'Marital status','Nationality']

# moth_occupation =(0,1,2,3,4,5,6,7,9,10,90,'others')
# fath_occupation =(0,1,2,3,4,5,6,7,8,9,10,90,'others')
# mode= [1,7,17,18,39,42,43,44,52,'others']
# Course =(33,8014,9003, 9070,  9085, 9119,9130, 9147, 9238,9254, 9500, 9556, 9670, 9773,9853,9991)
# moth_qualification= (1,2,3,12,19,34,37,38,'others')
# fath_qualification =(1,2,3,19,34,37,38,'others')
# order =(0, 1 ,2, 3, 4, 5, 6 )
# qualification =( 1, 'others')
# status = (1, 2 , 3,4, 5, 6)
# Nationality = ( 1 , 'others')

# col1,col2,col3 =st.columns(3)

# with col1:
#     moth_occupation = st.selectbox("Mother's occupation",moth_occupation)
#     fath_occupation =st.selectbox("Father's occupation",fath_occupation)
#     mode = st.selectbox("Application mode",mode)
#     Course = st.selectbox("Course",Course)
# with col2:
#     moth_qualification = st.selectbox("Mother's qualification",moth_qualification)
#     fath_qualification = st.selectbox("Father's qualification",fath_qualification)
#     qualification =st.selectbox('Previous qualification',qualification)
# with col3:
#     status = st.selectbox('Marital status',status)
#     Nationality =st.selectbox('Nationality',Nationality)
# order = st.slider('Application order', 0,6,1)

# ok = st.button('Predict')
# if ok:
#     lenc = np.array(le)
#     oenc = np.array(oe)
#     label = label.fit_transform(lenc)
#     onehot = onehot.fit_transform(oenc)

#     st.subheader(f'The predicted Dropout is ')


import streamlit as st
import pandas as pd

df = pd.read_csv("new_data.csv",index_col=0)
df.drop('Target',axis=1, inplace=True)
x= df.columns
# Title of the app
st.title("Student Dropout Risk Prediction")

# Sidebar for user input
st.sidebar.header("User Input Parameters")

# Create a function to collect input for all 37 features
def user_input_features():
    # Create a dictionary to hold the input values for each feature
    feature_inputs = {}
    
    # Dynamically generate number inputs for 37 features
    for i in range(1, 74):  # Loop from 1 to 37 (inclusive)
        feature_inputs[x[i]] = st.sidebar.number_input(x[i], min_value=0.0, max_value=70.0, value=5.0)
    
    # Convert the dictionary to a DataFrame
    return pd.DataFrame(feature_inputs, index=[0])

# Store the user input in a DataFrame
input_df = user_input_features()

# Display the user inputs in the main app
st.subheader("User Input Parameters")
st.write(input_df)

# Make predictions using the loaded model
prediction = new_model.forward(input_df)
# prediction_proba = new_model.predict_proba(input_df)

# # Display prediction and prediction probabilities
# st.subheader("Prediction")
# st.write(f"Predicted class: {prediction[0]}")

# st.subheader("Prediction Probabilities")
# st.write(prediction_proba)


