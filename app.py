import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the model sand tokenizer
tokenizer = AutoTokenizer.from_pretrained("nmcahill/mbti-classifier")
model = AutoModelForSequenceClassification.from_pretrained("nmcahill/mbti-classifier")

# Streamlit UI
st.title("Prediksi Tipe Kepribadian MBTI")

# Input text box
input_text = st.text_area("Masukkan teks untuk diprediksi:", "")

if input_text:
    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Predict MBTI type with probabilities
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)

    # List of MBTI personality types
    mbti_types = ["INFJ", "INFP", "INTJ", "INTP", "ISFJ", "ISFP", "ISTJ", "ISTP", "ENFJ", "ENFP", "ENTJ", "ENTP", "ESFJ", "ESFP", "ESTJ", "ESTP"]

    # Get predicted MBTI type and its probability
    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_mbti_type = mbti_types[predicted_class]
    predicted_probability = probabilities[0][predicted_class].item()

    # Display the predicted MBTI type and its probability
    st.write(f"Tipe kepribadian MBTI yang diprediksi: {predicted_mbti_type}")
    st.write(f"Probabilitas prediksi: {predicted_probability:.2%}")
