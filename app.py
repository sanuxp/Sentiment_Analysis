import streamlit as st
import joblib
import neattext.functions as nfx

# Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

# Preprocess text input
def preprocess_text(text):
    text = nfx.remove_stopwords(text)
    text = nfx.remove_punctuations(text)
    text = nfx.remove_special_characters(text)
    return text

# Main Streamlit app
def main():
    st.title("Emotion Detection from Text")
    st.write("Enter a sentence to predict its emotion (e.g., happy, sad, angry, love, fear).")

    # Text input
    user_input = st.text_area("Enter your text here:", height=100)

    # Load model
    model = load_model()

    # Predict button
    if st.button("Predict Emotion"):
        if user_input.strip():
            # Preprocess input
            clean_text = preprocess_text(user_input)
            
            # Make prediction
            prediction = model.predict([clean_text])[0]
            
            # Display result
            st.success(f"Predicted Emotion: **{prediction}**")
        else:
            st.error("Please enter some text to analyze.")

if __name__ == "__main__":
    main()