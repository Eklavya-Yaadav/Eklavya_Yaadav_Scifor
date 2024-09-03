# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 08:21:34 2024

@author: eklavya yaadav
"""

import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

# loading the saved model
model = pickle.load(open('E:\Model Apps\Loan Application Prediction\Eklavya_loan','rb'))


# creating a function for Prediction

def loan_prediction(input_data):
    
    new_data_df = pd.DataFrame([input_data])

    scaler = StandardScaler()
    new_data_scaled = scaler.fit_transform(new_data_df)

    predicted_status = model.predict(new_data_scaled)

    print("Predicted Loan Status:", "Approved" if predicted_status[0] == 1 else "Rejected")
  
    
  
def main():
    
    
    # giving a title
    st.title('Loan Prediction Web App')
    
    
    # getting the input data from the user
    
    
    Gender = st.text_input('Number of Pregnancies')
    Married = st.text_input('Glucose Level')
    Dependents = st.text_input('Blood Pressure value')
    Education = st.text_input('Skin Thickness value')
    Self_Employed = st.text_input('Insulin Level')
    Total_Income = st.text_input('BMI value')
    Loan_Amount = st.text_input('Diabetes Pedigree Function value')
    Loan_Amount_Term = st.text_input('Age of the Person')
    Credit_History = st.text_input('Age of the Person')
    Property_Area = st.text_input('Age of the Person')
    
    
    # code for Prediction
    Loan_Status = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        Loan_Status = loan_prediction([Gender, Married, Dependents, Education, Self_Employed, Total_Income, Loan_Amount, Loan_Amount_Term, Credit_History, Property_Area ])
        
        
    st.success(Loan_Status)
    
    
    
    
    
if __name__ == '__main__':
    main()
    

























