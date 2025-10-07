import streamlit as st
import joblib
import numpy as np

# Set the title
st.title("💼 Salary Prediction App")

st.divider()

st.write("This app predicts a person's salary based on their years of experience and job rate.")

# Input fields
years = st.number_input("Enter the years at company:", value=1, step=1, min_value=0)
jobrate = st.number_input("Enter the job rate:", value=3.5, step=0.5, min_value=0.0)

# Combine inputs into a single array
x = np.array([[years, jobrate]])

# Load the trained model
try:
    model = joblib.load("Linearmodel_Prediction_LR.pkl")  # ✅ Corrected filename format
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.divider()

# Predict button
if st.button("🔮 Predict Salary"):
    st.balloons()
    try:
        prediction = model.predict(x)
        st.success(f"The predicted salary is **${prediction[0]:,.2f}** 💰")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.info("Press the button above to get the predicted salary.")

    

