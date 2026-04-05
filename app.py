# app.py - Streamlit web application for spam detection
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data (only needed once)
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('punkt')

download_nltk_data()

# Load the saved model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

# Text preprocessing function (must match training preprocessing)
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = nltk.word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

# Main app
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")

st.title("📩 Adaptive Spam Email Detection System")
st.markdown("""
This system uses **ensemble machine learning** to detect spam messages in real-time.
Enter a message below to check if it's spam or legitimate.
""")

# Load model
model, vectorizer = load_model()

# User input
user_input = st.text_area("Enter your message:", height=150, 
                          placeholder="Type or paste a message here...")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Analyze Message", type="primary"):
        if user_input:
            # Preprocess and predict
            cleaned = preprocess_text(user_input)
            features = vectorizer.transform([cleaned])
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            # Display result
            if prediction == 'spam':
                st.error(f"🚫 **SPAM DETECTED**")
                st.markdown(f"Confidence: {probabilities[1]*100:.2f}%")
                st.markdown("⚠️ This message appears to be unwanted spam.")
            else:
                st.success(f"✅ **NOT SPAM**")
                st.markdown(f"Confidence: {probabilities[0]*100:.2f}%")
                st.markdown("This message appears to be legitimate.")
        else:
            st.warning("Please enter a message to analyze.")

with col2:
    if st.button("🧹 Clear"):
        st.rerun()

# Add example messages for testing
with st.expander("Try these example messages"):
    st.markdown("**Spam example:**")
    st.code("CONGRATULATIONS! You've won a FREE ticket to Bahamas. Call now to claim your prize!")
    
    st.markdown("**Ham (legitimate) example:**")
    st.code("Hey, are we still meeting for coffee tomorrow at 3pm?")

# Footer
st.markdown("---")
st.caption("Powered by Ensemble Learning (Naive Bayes + Logistic Regression + Random Forest) | TF-IDF Feature Extraction")