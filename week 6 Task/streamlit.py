import torch
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import streamlit as st
from app import make_prediction
import pandas as pd
school_df= pd.read_csv('../week 1 task/cleaned_data.csv', index_col=0)


Fath_qual =  [3,4,37,1,2,38,'others']
Moth_qual = [1,2,3,12,19,37,34,'others' ]
Moth_occ= [4,'others']
Fath_occ=  [ 9,0,90, "others"]
Application_mode = [7,51,17,39,'others' ]
Course =[9130,9119, 9003, 33,9085, 9070,9556, 9500, 9670,
       9773, 9853,9238,  9254, 9147,8014, 9991]
Marital_status = ['Yes','No']
International = ['Yes','No']
Gender = ['Yes','No']
Scholarship_holder = ['Yes','No']
Debtor = ['Yes','No']
Tuition_fees_up_to_date = ['Yes','No']

def main():
    st.title('Students Dropout Prediction')
    admission_grade = st.text_input('Admission grade')
    enrollment_age=st.text_input('Age at enrollment')
    enrolled_first=st.text_input('Curricular units 1st sem (enrolled)')
    approved_first=st.text_input('Curricular units 1st sem (approved)')
    grade_first=st.text_input('Curricular units 1st sem (grade)')
    evaluations_first=st.text_input('Curricular units 1st sem (evaluations)')
    enrolled_second=st.text_input('Curricular units 2nd sem (enrolled)')
    approved_second=st.text_input('Curricular units 2nd sem (approved)')
    grade_second=st.text_input('Curricular units 2nd sem (grade)')
    evaluations_second=st.text_input('Curricular units 2nd sem (evaluations)')
    prev_qual_grade=st.text_input('Previous qualification (grade)')
    inflation_rate=st.text_input('Inflation rate') 
    unemployment_rate=st.text_input('Unemployment rate')
    gender = st.sidebar.multiselect('Gender', Gender)
    scholarship_holder=st.sidebar.multiselect('Scholarship holder',Scholarship_holder)
    debtor=st.sidebar.multiselect('Debtor',Debtor)
    tuition_fees_up_to_date =st.sidebar.multiselect('Tuition fees up to date',Tuition_fees_up_to_date )
    marital_status=st.sidebar.multiselect('Marital status',Marital_status)
    international=st.sidebar.multiselect('International',International)
    app_mode = st.sidebar.multiselect("Application mode",Application_mode)
    course=st.sidebar.multiselect('Course', Course)
    fath_qual = st.sidebar.multiselect("Father's qualification",Fath_qual)
    moth_qual=st.sidebar.multiselect("Mother's qualification",Moth_qual)
    fath_occ=st.sidebar.multiselect("Father's occupation",Fath_occ)
    moth_occ = st.sidebar.multiselect("Mother's occupation",Moth_occ)







    # Completion_Age_Interaction =Completion_Rate_1st*enrollment_age
    # Completion_Grade_Interaction=Completion_Rate_1st*admission_grade
    # Completion_Rate_1st = approved_first/enrolled_first
    # Completion_Rate_2nd = approved_second/enrolled_second
    # Curricular_units_Average_grade =(grade_first + grade_second)/2
    # Grade_Interaction =admission_grade*prev_qual_grade
    # Previous_Age_Interaction=prev_qual_grade*enrollment_age
    # Total_Curricular_units_approved= approved_first+approved_second
    # Total_Curricular_units_enrolled= enrolled_first+enrolled_second
    # Total_Curricular_units_evaluations= evaluations_first+evaluations_second

    # if enrollment_age ==18:
    #     AdmGrades_category = 1
    # elif enrollment_age >18 and enrollment_age <= 19:
    #     AdmGrades_category = 2
    # elif enrollment_age >19 and enrollment_age <= 20:
    #     AdmGrades_category = 3
    # elif enrollment_age >20 and enrollment_age <= 22:
    #     AdmGrades_category = 4
    # elif enrollment_age >22 and enrollment_age <= 30:
    #     AdmGrades_category = 5
    # elif enrollment_age >30 and enrollment_age <= 70:
    #     AdmGrades_category = 6

    if st.button('Predict'):
        y_pred = make_prediction([grade_first,grade_second])
        st.success('Good!!!')
        return y_pred

if __name__ =="__main__":
    main()


            # Grouped_Admission_grade=



    # Grouped_Age_at_enrollment=


    # Grouped_Curricular_units_1st_sem_grade=


    # Grouped_Curricular_units_2nd_sem_grade=

    
    # Grouped_Previous_qualification_grade=




















       
       

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# school_df[numeric] = scaler.fit_transform(school_df[numeric])







