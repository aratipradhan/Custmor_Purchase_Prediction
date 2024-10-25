import streamlit as st
import numpy as np
import pickle

# Load the model and scaler
with open('model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit App Layout
st.title("Customer Purchase Prediction App")

st.write("""
### Will the customer make a purchase?
Enter the customer's details to check the likelihood of making a purchase.
""")

# Input fields for features
feature_1 = st.number_input("Customer Age", min_value=0.0, step=0.1)
feature_2 = st.number_input("Income Level", min_value=0.0, step=0.1)

# Prediction Button
if st.button("Predict Purchase"):
    # Prepare the input for prediction
    input_data = np.array([[feature_1, feature_2]])
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    purchase_prediction = classifier.predict(input_data_scaled)
    purchase_probability = classifier.predict_proba(input_data_scaled)

    # Display prediction results
    if purchase_prediction[0] == 1:
        st.success("The customer is likely to make a purchase.")
    else:
        st.warning("The customer is not likely to make a purchase.")

    # Display the probability
    st.write(f"**Probability of No Purchase:** {purchase_probability[0][0]:.2f}")
    st.write(f"**Probability of Purchase:** {purchase_probability[0][1]:.2f}")
