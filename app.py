import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler , LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

##load the model
model = load_model("model.h5")

#load the encoder and scaler
with open("label_encoder_gender.pkl" , "rb") as file:
    label_encoder_gender = pickle.load(file)
with open("onehot_encoder_geo.pkl" , "rb") as file:
    onehot_encoder_geo = pickle.load(file)
with open("scaler.pkl" , "rb") as file:
    scaler = pickle.load(file)       

## streamlit app
st.title("Customer Churn Prediction")

#User input
geography = st.selectbox("Geography" , onehot_encoder_geo.categories_[0]) #with ever is the category of 0, by default it will get loaded. It can be [0] , [1] , [2] , any name that we give over here.

gender = st.selectbox('Gender' , label_encoder_gender.classes_) #whatever values we have wrt to classes , we'll get over here.
age = st.slider('Age' , 18 , 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

#prepare the user input by taking it in some kind of dictionary
#prepare the input data
input_data = pd.DataFrame ({
    "CreditScore":[credit_score],
    "Gender":[label_encoder_gender.transform([gender])[0]], #getting the 0th value
    "Age":[age],
    "Tenure":[tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]

})
#one hot encode "geography"

geo_encoded= onehot_encoder_geo.transform([[geography]]).toarray()#taken in form of list and converted into array
geo_encoded_df = pd.DataFrame(geo_encoded, columns= onehot_encoder_geo.get_feature_names_out(["Geography"]))# converting it into a dataframe

#combine one hot encoded with the input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df] , axis =1)

#scale the input data
input_data_scaled = scaler.transform(input_data)

#predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability : {prediction_proba: .2f}')

if prediction_proba>0.5:
    st.write("the customer is likely to churn")
else :
    st.write("The customer is not likely to churn")    