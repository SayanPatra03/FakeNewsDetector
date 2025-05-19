import streamlit as st
import pickle

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.title("📰 Fake News Detector")

user_input = st.text_area("Enter the news article text here:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize input
        input_vector = vectorizer.transform([user_input])
        result = model.predict(input_vector)[0]

        if result == 1:
            st.success("✅ This is Real News.")
        else:
            st.error("🚨 This is Fake News.")
