import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_diabetes_data():
    return pd.read_csv("diabetes_prediction_dataset.csv")

def preprocess_data(df):
    df.drop_duplicates(inplace=True)
    df = df.dropna()
    df['gender'] = df['gender'].str.strip()
    df['smoking_history'] = df['smoking_history'].str.strip()
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100], labels=['Child', 'Young Adult', 'Adult', 'Middle-aged', 'Senior'])
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal weight', 'Overweight', 'Obese'])
    df['health_risk_score'] = df['hypertension'] + df['heart_disease'] + (df['age'] > 50).astype(int)
    df['high_glucose'] = (df['blood_glucose_level'] > 150).astype(int)
    return df

def normalize_features(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

def main():
    st.title("Diabetes Prediction Dataset Analysis")

    df = load_diabetes_data()
    df = preprocess_data(df)

    st.sidebar.header("Filters")
    age_range = st.sidebar.slider("Age Range", int(df['age'].min()), int(df['age'].max()), (int(df['age'].min()), int(df['age'].max())))
    gender = st.sidebar.multiselect("Gender", df['gender'].unique(), df['gender'].unique())
    bmi_range = st.sidebar.slider("BMI Range", float(df['bmi'].min()), float(df['bmi'].max()), (float(df['bmi'].min()), float(df['bmi'].max())))
    smoking_history = st.sidebar.multiselect("Smoking History", df['smoking_history'].unique(), df['smoking_history'].unique())
    HbA1c_range = st.sidebar.slider("HbA1c Level Range", float(df['HbA1c_level'].min()), float(df['HbA1c_level'].max()), 
                                     (float(df['HbA1c_level'].min()), float(df['HbA1c_level'].max())))
    glucose_range = st.sidebar.slider("Blood Glucose Level Range", float(df['blood_glucose_level'].min()), 
                                       float(df['blood_glucose_level'].max()), 
                                       (float(df['blood_glucose_level'].min()), float(df['blood_glucose_level'].max())))

    filtered_df = df[
        (df['age'] >= age_range[0]) & (df['age'] <= age_range[1]) &
        (df['gender'].isin(gender)) &
        (df['bmi'] >= bmi_range[0]) & (df['bmi'] <= bmi_range[1]) &
        (df['smoking_history'].isin(smoking_history)) &
        (df['HbA1c_level'] >= HbA1c_range[0]) & (df['HbA1c_level'] <= HbA1c_range[1]) &
        (df['blood_glucose_level'] >= glucose_range[0]) & (df['blood_glucose_level'] <= glucose_range[1])
    ]

    st.subheader("Dataset Overview")
    st.write(filtered_df.head())
    st.write(f"Shape of the dataset: {filtered_df.shape}")

    st.subheader("Data Distribution")
    numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns
    selected_col = st.selectbox("Select a column for distribution plot", numeric_cols)
    fig, ax = plt.subplots()
    sns.histplot(data=filtered_df, x=selected_col, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    corr_matrix = filtered_df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Scatter Plot")
    x_axis = st.selectbox("Select X-axis", numeric_cols)
    y_axis = st.selectbox("Select Y-axis", numeric_cols)
    fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color='diabetes')
    st.plotly_chart(fig)

    st.subheader("Box Plot")
    box_col = st.selectbox("Select a column for box plot", numeric_cols)
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_df, y=box_col, x='diabetes', ax=ax)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
