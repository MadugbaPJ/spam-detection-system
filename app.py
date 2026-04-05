# app.py - Streamlit web application for spam detection
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ============================================
# DOWNLOAD NLTK DATA - FIX FOR STREAMLIT CLOUD
# ============================================
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data - must run before using tokenizers"""
    try:
        # Try to download with quiet mode first
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        # Verify punkt_tab is available by testing tokenization
        test_text = "test"
        nltk.word_tokenize(test_text)
        
        return True
    except LookupError:
        # If still missing, try with verbose output
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        return True
    except Exception as e:
        st.error(f"NLTK download error: {str(e)}")
        return False

# Download NLTK data at startup
with st.spinner("Loading language models..."):
    nltk_ready = download_nltk_data()

# ============================================
# Load Model and Vectorizer
# ============================================
@st.cache_resource
def load_model():
    """Load the saved model and vectorizer"""
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

# ============================================
# Text Preprocessing
# ============================================
stop_words = set(stopwords.words('english')) if nltk_ready else set()
stemmer = PorterStemmer()

def preprocess_text(text):
    """Clean and preprocess message text"""
    if not nltk_ready:
        return text  # Fallback
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    try:
        words = nltk.word_tokenize(text)
    except LookupError:
        # Fallback: simple split if NLTK fails
        words = text.split()
    
    # Remove stopwords and stem
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

# ============================================
# Main App
# ============================================
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")

st.title("📩 Adaptive Spam Email Detection System")
st.markdown("""
This system uses **ensemble machine learning** to detect spam messages in real-time.
Enter a message below to check if it's spam or legitimate.
""")

# Load model
if nltk_ready:
    try:
        model, vectorizer = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
else:
    st.error("❌ Failed to load language models. Please refresh the page.")
    st.stop()

# User input
user_input = st.text_area("Enter your message:", height=150, 
                          placeholder="Type or paste a message here...")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Analyze Message", type="primary"):
        if user_input:
            # Preprocess and predict
            with st.spinner("Analyzing message..."):
                cleaned = preprocess_text(user_input)
                features = vectorizer.transform([cleaned])
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                
                # Get spam probability
                spam_prob = probabilities[1] if len(probabilities) > 1 else 0
                
                # Display result
                st.markdown("---")
                if prediction == 'spam':
                    st.error(f"🚫 **SPAM DETECTED**")
                    st.metric("Confidence", f"{spam_prob*100:.2f}%")
                    st.warning("⚠️ This message appears to be unwanted spam. Do not click any links.")
                else:
                    st.success(f"✅ **NOT SPAM**")
                    st.metric("Confidence", f"{(1-spam_prob)*100:.2f}%")
                    st.info("This message appears to be legitimate.")
        else:
            st.warning("Please enter a message to analyze.")

with col2:
    if st.button("🧹 Clear"):
        st.rerun()

# Example messages
with st.expander("📋 Try these example messages"):
    st.markdown("**Spam example:**")
    st.code("CONGRATULATIONS! You've won a FREE ticket to Bahamas. Call now to claim your prize!")
    
    st.markdown("**Ham (legitimate) example:**")
    st.code("Hey, are we still meeting for coffee tomorrow at 3pm?")
    
    if st.button("📋 Copy Spam Example"):
        st.session_state.example = "CONGRATULATIONS! You've won a FREE ticket to Bahamas. Call now to claim your prize!"

# Footer
st.markdown("---")
st.caption("Powered by Ensemble Learning (Naive Bayes + Logistic Regression + Random Forest) | TF-IDF Feature Extraction")
