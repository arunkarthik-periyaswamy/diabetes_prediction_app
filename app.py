import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Loading the pickle model
model = joblib.load('Diabetes_Prediction_Model.pkl')

def diabetes_app():
    st.set_page_config(page_title="Diabetes Prediction Tool", page_icon="🩺", layout="wide")

    st.markdown("""
        <style>
        .stApp {
            background-color: #f0f8ff;
            color: #333333;
        }
        .st-bw {
            background-color: #ffffff;
        }
        .stButton>button {
            background-color: #1e90ff;
            color: white;
            font-weight: bold;
        }
        .sidebar .sidebar-content {
            color: #1e90ff;
        }
        </style>
    """, unsafe_allow_html=True)

    # Ttitle to show on the webpage
    st.title('🩺 Diabetes Prediction Tool')

    # Loading the image  
    st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://www.onlymyhealth.com/immersive/addressing-india-diabetes-dilemma/images/OMH-diabets.gif" alt="Alt Text" width="300">
    </div>
    """,
    unsafe_allow_html=True
    )

    # Sidebar for user input
    st.sidebar.markdown(
    """
    <h2 style="color: #1e90ff;">User Input Parameters</h2>
    """,
    unsafe_allow_html=True
    )

    age = st.sidebar.slider('Age', 1, 80, 30)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    hypertension = st.sidebar.selectbox('Hypertension', ['No', 'Yes'])
    heart_disease = st.sidebar.selectbox('Heart Disease', ['No', 'Yes'])
    smoking_history = st.sidebar.selectbox('Smoking History', ['never', 'former', 'current', 'not current'])
    bmi = st.sidebar.slider('BMI', 10.0, 50.0, 25.0, 0.1)
    hba1c_level = st.sidebar.slider('HbA1c Level', 3.5, 9.0, 5.5, 0.1)
    blood_glucose_level = st.sidebar.slider('Blood Glucose Level', 70, 300, 120)
    health_risk_score = st.sidebar.slider('Health Risk Score', 0, 100, 50)
    high_glucose = st.sidebar.selectbox('High Glucose', ['No', 'Yes'])

    # Map smoking history and high glucose to numerical values
    smoking_history_map = {
        'never': 0,
        'former': 1,
        'current': 2,
        'not current': 3
    }
    high_glucose_map = {
        'No': 0,
        'Yes': 1
    }

    # Create a dataframe from user input
    data = {
        'gender': [1 if gender == 'Male' else 0],
        'age': [age],
        'hypertension': [1 if hypertension == 'Yes' else 0],
        'heart_disease': [1 if heart_disease == 'Yes' else 0],
        'smoking_history': [smoking_history_map[smoking_history]],
        'bmi': [bmi],
        'HbA1c_level': [hba1c_level],
        'blood_glucose_level': [blood_glucose_level],
        'health_risk_score': [health_risk_score],
        'high_glucose': [high_glucose_map[high_glucose]]
    }
    input_df = pd.DataFrame(data)

    # # Display the user input values in the form
    # st.sidebar.subheader('User Input Values')
    # st.sidebar.write(f"Age: {age}")
    # st.sidebar.write(f"Gender: {gender}")
    # st.sidebar.write(f"Hypertension: {hypertension}")
    # st.sidebar.write(f"Heart Disease: {heart_disease}")
    # st.sidebar.write(f"Smoking History: {smoking_history}")
    # st.sidebar.write(f"BMI: {bmi}")
    # st.sidebar.write(f"HbA1c Level: {hba1c_level}")
    # st.sidebar.write(f"Blood Glucose Level: {blood_glucose_level}")
    # st.sidebar.write(f"Health Risk Score: {health_risk_score}")
    # st.sidebar.write(f"High Glucose: {high_glucose}")

    # Predict the outcome
    if st.button('Predict'):
        prediction = model.predict(input_df)
        st.subheader('Prediction Result')
        if prediction[0] == 1:
            st.markdown(
                """
                <div style="color: red; font-size: 24px;">
                    The Diabetes Prediction model predicts that the patient has Diabetes.
                </div>
                <div style="text-align: center;">
                    <img src="https://static.wixstatic.com/media/4ae295_a6c79e5afd2e4654b50fd5f45233148d~mv2.gif" alt="Diabetes GIF" width="300">
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="color: green; font-size: 24px;">
                    The Diabetes Prediction model predicts that the patient does not have Diabetes.
                </div>
                <div style="text-align: center;">
                    <img src="https://media1.giphy.com/media/3ornk8qaF9ytl9zmY8/giphy.gif?cid=6c09b952fcpqdm7reanstlwot8ws6vobwugymwdjp5ika63p&ep=v1_gifs_search&rid=giphy.gif&ct=g" alt="No Diabetes GIF" width="300">
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == '__main__':
    diabetes_app()