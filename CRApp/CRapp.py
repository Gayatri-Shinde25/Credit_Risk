# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5dc;  
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the model from disk
model = joblib.load(r"C:/Users/a/Downloads/Credit Risk/CRApp/XGBpdModel.pkl")

# Import the preprocessing function
from AutomatePreprocessing import ModelPreProcessing

def main():
    
    # Setting Application title
    st.title('Credit Risk Prediction App')

    # Setting Application description
    st.markdown("""
    :dart:  This Streamlit app is designed to predict loan default risk based on customer information.
    The application supports both online prediction and batch data prediction.
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    # Setting Application sidebar default
    image = Image.open('App.jpg')
    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Loan Default Risk')
    st.sidebar.image(image)

    if add_selectbox == "Online":
        st.info("Input data below")
        # Collect input data
        st.subheader("Personal Information")
        person_age = st.number_input('Person Age:', min_value=18, max_value=75, value=30)
        person_income = st.number_input('Person Income:', min_value=0, max_value=500000, value=50000)
        person_home_ownership = st.selectbox('Home Ownership:', ('Rent', 'Own', 'Mortgage','Other'))
        person_emp_length = st.slider('Employment Length (years):', min_value=0, max_value=47, value=5)

        st.subheader("Loan Information")
        loan_intent = st.selectbox('Loan Intent:', ('Personal', 'Business', 'Education', 'Debt Consolidation', 'Venture', 'Home Improvement','Medical'))
        loan_grade = st.selectbox('Loan Grade:', ('A', 'B', 'C', 'D', 'E', 'F', 'G'))
        loan_amnt = st.number_input('Loan Amount:', min_value=0, max_value=50000, value=1000)
        loan_int_rate = st.number_input('Loan Interest Rate (%):', min_value=0.0, max_value=30.0, value=10.0)
        loan_percent_income = st.number_input('Loan Amount as Percentage of Income:', min_value=0.0, max_value=100.0, value=0.0)
        cb_person_default_on_file = st.selectbox('Default on File:', ('Yes', 'No'))
        cb_person_cred_hist_length = st.number_input('Credit History Length (Years):', min_value=0, max_value=80, value=12)

        data = {
            'person_age': person_age,
            'person_income': person_income,
            'person_home_ownership': person_home_ownership,
            'person_emp_length': person_emp_length,
            'loan_intent': loan_intent,
            'loan_grade': loan_grade,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income,
            'cb_person_default_on_file': cb_person_default_on_file,
            'cb_person_cred_hist_length': cb_person_cred_hist_length
        }
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)

        # Preprocess inputs
        preprocess_df = ModelPreProcessing(features_df)

        if st.button('Predict'):
            prediction = model.predict(preprocess_df)
            if prediction == 0:
                st.success('No, the person is unlikely to default on the loan.')
            else:
                st.warning('Yes, the person is likely to default on the loan.')
    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            # Get overview of data
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            # Preprocess inputs
            preprocess_df = ModelPreProcessing(data)
            if st.button('Predict'):
                # Get batch prediction
                prediction = model.predict(preprocess_df)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1: 'Yes, the person is likely to default on the loan.',
                                                       0: 'No, the person is unlikely to default on the loan.'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)

if __name__ == '__main__':
    main()
