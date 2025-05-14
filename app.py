import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder 
import pandas as pd
import pickle

#Load the trained model
model = load_model('model.h5')

#Open the preprocessor
with open('preprocessor.pkl','rb') as file:
    preprocessor = pickle.load(file)

##Streamlit app

st.title("Customer Churn Prediction")

#User input
geography = st.selectbox('Geography',['France','Spain','Germany'])
gender = st.selectbox('Gender',['Male','Female'])
age = st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

#Prepare the input data
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Geography':[geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary':[estimated_salary]
})

input_transformed = preprocessor.transform(input_data)
prediction = model.predict(input_transformed)
prediction_proba = prediction[0][0]
st.write(f"Churn Probability: {prediction_proba:.2f}")
if prediction_proba>0.5:
       st.error("Prediction: Customer is likely to leave 😢")
else:
    st.success("Prediction: Customer is likely to stay 😊")