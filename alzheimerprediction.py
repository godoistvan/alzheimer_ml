# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:55:51 2024

@author: User
"""

import numpy as np
import pickle 
import streamlit as st

loaded_model=pickle.load(open('C:/Users/User/Downloads/catboost_model.sav'))

def diabetes_prediction(input_data):
    

    input_data_as_numpy_array=np.asarray(input_data)

    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==0):
       return 'The person has alzheimer'
    else:
        return 'The person doesnt have alzheimer'

def main():
    
    st.title('Alzheimer prediction webapp')
    
Age = st.number_input('Age', min_value=60, max_value=90, value=65)
BMI = st.number_input('BMI', min_value=15.0, max_value=40.0, value=25.0)
AlcoholConsumption = st.number_input('AlcoholConsumption (units/week)', min_value=0, max_value=20, value=0)
PhysicalActivity = st.number_input('PhysicalActivity (hours/week)', min_value=0, max_value=10, value=0)
DietQuality = st.number_input('DietQuality (0-10)', min_value=0, max_value=10, value=5)
SleepQuality = st.number_input('SleepQuality (4-10)', min_value=4, max_value=10, value=7)
SystolicBP = st.number_input('SystolicBP (mmHg)', min_value=90, max_value=180, value=120)
DiastolicBP = st.number_input('DiastolicBP (mmHg)', min_value=60, max_value=120, value=80)
CholesterolTotal = st.number_input('CholesterolTotal (mg/dL)', min_value=150, max_value=300, value=200)
CholesterolLDL = st.number_input('CholesterolLDL (mg/dL)', min_value=50, max_value=200, value=100)
CholesterolHDL = st.number_input('CholesterolHDL (mg/dL)', min_value=20, max_value=100, value=50)
CholesterolTriglycerides = st.number_input('CholesterolTriglycerides (mg/dL)', min_value=50, max_value=400, value=150)
MMSE = st.number_input('MMSE (0-30)', min_value=0, max_value=30, value=25)
FunctionalAssessment = st.number_input('FunctionalAssessment (0-10)', min_value=0, max_value=10, value=10)
ADL = st.number_input('ADL (0-10)', min_value=0, max_value=10, value=10)

# Categorical inputs (using st.selectbox where appropriate)
Gender = st.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Male' if x == 0 else 'Female')
Ethnicity = st.selectbox('Ethnicity', options=[0,1,2,3], format_func=lambda x: ['Caucasian','African American','Asian','Other'][x])
EducationLevel = st.selectbox('EducationLevel', options=[0,1,2,3], format_func=lambda x: ['None','High School','Bachelor\'s','Higher'][x])
Smoking = st.selectbox('Smoking', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
FamilyHistoryAlzheimers = st.selectbox('FamilyHistoryAlzheimers', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
CardiovascularDisease = st.selectbox('CardiovascularDisease', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
Diabetes = st.selectbox('Diabetes', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
Depression = st.selectbox('Depression', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
HeadInjury = st.selectbox('HeadInjury', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
Hypertension = st.selectbox('Hypertension', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
MemoryComplaints = st.selectbox('MemoryComplaints', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
BehavioralProblems = st.selectbox('BehavioralProblems', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
Confusion = st.selectbox('Confusion', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
Disorientation = st.selectbox('Disorientation', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
PersonalityChanges = st.selectbox('PersonalityChanges', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
DifficultyCompletingTasks = st.selectbox('DifficultyCompletingTasks', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
Forgetfulness = st.selectbox('Forgetfulness', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')

# Additional Variables (depending on how these are defined)
Diagnosis = st.text_input('Diagnosis')
AgeGroup = st.text_input('AgeGroup')
BMICategory = st.text_input('BMICategory')
AlcoholConsumptionCategory = st.text_input('AlcoholConsumptionCategory')
PhysicalActivityCategory = st.text_input('PhysicalActivityCategory')
ComorbidityScore = st.number_input('ComorbidityScore', min_value=0, value=0)
CognitiveBehavioralIssuesCount = st.number_input('CognitiveBehavioralIssuesCount', min_value=0, value=0)
MemoryAttentionCluster = st.text_input('MemoryAttentionCluster')
MMSESeverity = st.text_input('MMSESeverity')