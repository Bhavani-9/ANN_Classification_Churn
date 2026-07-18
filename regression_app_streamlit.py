import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle 
import streamlit as st

# load the trained model
model = tf.keras.models.load_model('regressionmodel.h5')

# load the pickle files
with open('le.pkl','rb') as file:
    labelencoder_gender=pickle.load(file)

with open('ohe_geo.pkl','rb') as file:
    ohe_geography=pickle.load(file)
    
with open('sscaler.pkl','rb') as file:
    standard_scaler=pickle.load(file)

# Streamlit 
st.title("Churn Prediction")

# inputs

CreditScore=st.number_input('CreditScore')
Geography = st.selectbox("Geography",ohe_geography.categories_[0])
Gender = st.selectbox('Gender',labelencoder_gender.classes_)
Age = st.slider('age',18,90)
Tenure=st.slider('Tenure',0,10)
Balance = st.number_input('Balance')
NumOfProducts = st.slider('NumOfProducts',1,4)
HasCrditCard = st.selectbox('HasCrCard',[0,1])
IsActiveMember = st.selectbox('IsActiveMember',[0,1])
EstimatedSalary = st.number_input('EstimatedSalary')


data=pd.DataFrame({
    'CreditScore' : [CreditScore],
    'Gender' : [labelencoder_gender.transform([Gender])[0]],
    'Age' : [Age],
    'Tenure' : [Tenure],
    'Balance' : [Balance],
    'NumOfProducts' : [NumOfProducts],
    'HasCrCard' : [HasCrditCard],
    'IsActiveMember' : [IsActiveMember],
    'EstimatedSalary' : [EstimatedSalary]
})

geo_encoded=ohe_geography.transform([[Geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=ohe_geography.get_feature_names_out(['Geography']))


data_df=pd.concat([data.reset_index(drop=True),geo_encoded_df],axis=1)
data_df

# Scale the data
data_df_scaled=standard_scaler.transform(data_df)

# prediction churn
prediction=model.predict(data_df_scaled)
prediction_probability=prediction[0][0]

st.write(f"Churn Probability : {prediction_probability:.2f}")

if prediction_probability > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")







