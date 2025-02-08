import streamlit as st  
import tensorflow as tf
from tensorflow import keras
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

def predict_diabetes(input_data,model_path):
     model = keras.models.load_model(model_path)
     scaler=joblib.load('./model/scaler.pkl')
     input_data=np.array(input_data).reshape(1,-1)
     input_data=scaler.transform(input_data)
     prediction = model.predict(input_data)[0][0]
     return "Diabetic" if prediction > 0.5 else "Non-Diabetic"
 
def main():
    st.title("Diabetes Prediction App")
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    
    model_choice = st.radio("Choose Model", ["Model 1", "Model 2"])
    model_path = "./model/diabetes_model1.h5" if model_choice == "Model 1" else "./model/diabetes_model2.h5"
    
    
    if st.button("Predict"):
        input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        result = predict_diabetes(input_data, model_path)
        st.write("Prediction:", result)


if __name__ =="__main__":
    main()
      