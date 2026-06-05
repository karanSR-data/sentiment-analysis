import streamlit as st
import joblib
import re
import numpy as np


# Load Model & Vectorizer

vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")
model = joblib.load("../models/Logistic Regression_model.pkl")


# Text Cleaning Function

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)  # remove HTML tags
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)  # keep only alphabets
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

# StreamLit UI

st.set_page_config(page_title = "Sentiment Analysis App", layout = "centered")

st.title("📝 Sentiment Analysis App")
st.write("Enter a review below and let the ML model classify it as **Positive** or **Negative**.")

user_input = st.text_area("Enter review:", height=150)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first!")
    else:
        #clean text
        cleaned = clean_text(user_input)

        #Vecctorize
        vector = vectorizer.transform([cleaned])

        #Predict
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0][1]
        # probability of positive

        #output
        if prediction == 1:
            st.success(f"😀 **Positive Review**")
        else:
            st.error(f"😞 **Negative Review**")

        st.write(f"### Confidence Score: {probability:.2f}")

st.markdown("---")
st.write("Built with ❤️ using Streamlit & Machine Learning ")