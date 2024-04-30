import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('XGB_churn.pkl')
gender_encode = joblib.load('gender_encode.pkl')
oneHot_encode_geo = joblib.load('oneHot_encode_geo.pkl')

def main():
    st.title('Churn Model Deployment')

    CreditScore = st.number_input("Credit Score", 0, 850) # Done
    Geography = st.selectbox("Geography", ["France", "Spain", "Germany"]) #Done
    Gender = st.radio("Gender", ["Male", "Female"]) # Done
    Age = st.number_input("Age", 0, 100) # Done
    Tenure = st.number_input("Tenure (years with the bank)", 0, 10) # Done
    Balance = st.number_input("Balance", 0, 1000000000) # Done
    NumOfProducts = st.selectbox("Number of Products", [1, 2, 3, 4]) # Done
    HasCrCard = st.radio("Has Credit Card", ["Yes", "No"]) # Done
    IsActiveMember = st.radio("Is Active Member", ["Yes", "No"]) #Done
    EstimatedSalary = st.number_input("Estimated Salary", 0, 1000000000) # Done
    
    HasCrCard = 1 if HasCrCard == 'Yes' else 0
    IsActiveMember = 1 if IsActiveMember == 'Yes' else 0

    data = {'CreditScore':int(CreditScore), 'Geography': Geography, 'Gender': Gender, 'Age': int(Age),
            'Tenure': int(Tenure), 'Balance': int(Balance), 'NumOfProducts': NumOfProducts, 'HasCrCard':int(HasCrCard), 
            'IsActiveMember': int(IsActiveMember),'EstimatedSalary': int(EstimatedSalary)}
    
    
    df = pd.DataFrame([list(data.values())], columns=['CreditScore','Geography','Gender','Age','Tenure','Balance',
                                                      'NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary'])

    df = df.replace(gender_encode)
    cat_geo = df[['Geography']]
    cat_enc_geo=pd.DataFrame(oneHot_encode_geo.transform(cat_geo).toarray(),columns=oneHot_encode_geo.get_feature_names_out())
    df = pd.concat([df,cat_enc_geo], axis=1)
    df = df.drop(['Geography'],axis=1)
    
    if st.button('Make Prediction'):
        features=df      
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()