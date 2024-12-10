# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:55:51 2024
@author: User
"""

import numpy as np
import pickle
import streamlit as st
import pandas as pd
import joblib
# Load the trained model once at the start of the script
loaded_model = pickle.load(open('catboost_model.sav', 'rb'))
minmax = joblib.load('minmax.pkl')
standard = joblib.load('standard.pkl')
encoders = joblib.load('encoders.pkl')
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=object)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person has Alzheimer\'s'
    else:
        return 'The person does not have Alzheimer\'s'

def main():
    st.title('Alzheimer Prediction Web App')

    # Numeric Inputs
    Age = st.number_input('Age', min_value=60, max_value=90, value=65)
    BMI = st.number_input('BMI', min_value=15.0, max_value=40.0, value=25.0)
    AlcoholConsumption = st.number_input('Alcohol Consumption (units/week)', min_value=0, max_value=20, value=0)
    PhysicalActivity = st.number_input('Physical Activity (hours/week)', min_value=0, max_value=10, value=0)
    DietQuality = st.number_input('Diet Quality (0-10)', min_value=0, max_value=10, value=5)
    SleepQuality = st.number_input('Sleep Quality (4-10)', min_value=4, max_value=10, value=7)
    SystolicBP = st.number_input('Systolic BP (mmHg)', min_value=90, max_value=180, value=120)
    DiastolicBP = st.number_input('Diastolic BP (mmHg)', min_value=60, max_value=120, value=80)
    CholesterolTotal = st.number_input('Cholesterol Total (mg/dL)', min_value=150, max_value=300, value=200)
    CholesterolLDL = st.number_input('Cholesterol LDL (mg/dL)', min_value=50, max_value=200, value=100)
    CholesterolHDL = st.number_input('Cholesterol HDL (mg/dL)', min_value=20, max_value=100, value=50)
    CholesterolTriglycerides = st.number_input('Cholesterol Triglycerides (mg/dL)', min_value=50, max_value=400, value=150)
    MMSE = st.number_input('MMSE (0-30)', min_value=0, max_value=30, value=25)
    FunctionalAssessment = st.number_input('Functional Assessment (0-10)', min_value=0, max_value=10, value=10)
    ADL = st.number_input('ADL (0-10)', min_value=0, max_value=10, value=10)

    # Categorical Inputs as numeric codes
    Gender = st.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Male' if x == 0 else 'Female')
    Ethnicity = st.selectbox('Ethnicity', options=[0,1,2,3], format_func=lambda x: ['Caucasian','African American','Asian','Other'][x])
    EducationLevel = st.selectbox('Education Level', options=[0,1,2,3], format_func=lambda x: ['None','High School','Bachelor\'s','Higher'][x])
    Smoking = st.selectbox('Smoking', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    FamilyHistoryAlzheimers = st.selectbox('Family History of Alzheimer\'s', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    CardiovascularDisease = st.selectbox('Cardiovascular Disease', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    Diabetes = st.selectbox('Diabetes', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    Depression = st.selectbox('Depression', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    HeadInjury = st.selectbox('Head Injury', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    Hypertension = st.selectbox('Hypertension', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    MemoryComplaints = st.selectbox('Memory Complaints', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    BehavioralProblems = st.selectbox('Behavioral Problems', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    Confusion = st.selectbox('Confusion', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    Disorientation = st.selectbox('Disorientation', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    PersonalityChanges = st.selectbox('Personality Changes', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    DifficultyCompletingTasks = st.selectbox('Difficulty Completing Tasks', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    Forgetfulness = st.selectbox('Forgetfulness', options=[0,1], format_func=lambda x: 'No' if x == 0 else 'Yes')

    # Additional fields: If the model expects encoded values, you must encode them here.
    # For demonstration, let's assume these are either already encoded or not used.
    AgeGroup = st.text_input('Age Group', 'Elderly')
    BMICategory = st.text_input('BMI Category', 'Normal')
    AlcoholConsumptionCategory = st.text_input('Alcohol Consumption Category', 'Moderate')
    PhysicalActivityCategory = st.text_input('Physical Activity Category', 'Moderate')
    ComorbidityScore = st.number_input('Comorbidity Score', min_value=0, value=0)
    CognitiveBehavioralIssuesCount = st.number_input('Cognitive Behavioral Issues Count', min_value=0, value=0)
    MemoryAttentionCluster = st.text_input('Memory Attention Cluster', '2')
    MMSESeverity = st.text_input('MMSE Severity', 'Normal')

    # Prepare the input features
    # If the model expects numeric encodings for the textual categories (like AgeGroup),
    # you should convert them before calling the prediction function.
    # Here, we are passing them as-is. Adjust as needed.
    input_features = [
        Age,
        Gender,
        Ethnicity,
        EducationLevel,
        BMI,
        Smoking,
        AlcoholConsumption,
        PhysicalActivity,
        DietQuality,
        SleepQuality,
        FamilyHistoryAlzheimers,
        CardiovascularDisease,
        Diabetes,
        Depression,
        HeadInjury,
        Hypertension,
        SystolicBP,
        DiastolicBP,
        CholesterolTotal,
        CholesterolLDL,
        CholesterolHDL,
        CholesterolTriglycerides,
        MMSE,
        FunctionalAssessment,
        MemoryComplaints,
        BehavioralProblems,
        ADL,
        Confusion,
        Disorientation,
        PersonalityChanges,
        DifficultyCompletingTasks,
        Forgetfulness,
        AgeGroup,
        BMICategory,
        AlcoholConsumptionCategory,
        PhysicalActivityCategory,
        ComorbidityScore,
        CognitiveBehavioralIssuesCount,
        MemoryAttentionCluster,
        MMSESeverity
    ]

    Diagnosis = ''
    if st.button('Get Alzheimer Test Result'):
        columns = [
            'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking', 'AlcoholConsumption',
            'PhysicalActivity', 'DietQuality', 'SleepQuality', 'FamilyHistoryAlzheimers',
            'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury', 'Hypertension',
            'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
            'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 'MemoryComplaints',
            'BehavioralProblems', 'ADL', 'Confusion', 'Disorientation', 'PersonalityChanges',
            'DifficultyCompletingTasks', 'Forgetfulness', 'AgeGroup', 'BMICategory',
            'AlcoholConsumptionCategory', 'PhysicalActivityCategory', 'ComorbidityScore',
            'CognitiveBehavioralIssuesCount', 'MemoryAttentionCluster', 'MMSESeverity'
        ]

        input_df = pd.DataFrame([input_features], columns=columns)

    # Identify numerical and categorical columns
        numerical_columns = [
        'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL',
        'CholesterolHDL', 'CholesterolTriglycerides'
        ]

        categorical_columns = [
        'BMICategory', 'AlcoholConsumptionCategory', 'AgeGroup',
        'PhysicalActivityCategory', 'MMSESeverity'
        ]

    # Apply transformations
        input_df[numerical_columns] = minmax.transform(input_df[numerical_columns])
        input_df[numerical_columns] = standard.transform(input_df[numerical_columns])

        for col in categorical_columns:
            input_df[col] = encoders[col].transform(input_df[col].astype(str))

        final_input = input_df.values.tolist()[0]

        Diagnosis = diabetes_prediction(final_input)
        st.success(Diagnosis)


if __name__ == '__main__':
    main()
