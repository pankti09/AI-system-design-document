import re
import emoji
import torch
import streamlit as st
import torch.nn.functional as F
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load three separate models
sentiment_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)  # Negative, Neutral, Positive
response_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)  # No Reply, Reply Needed
crisis_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)  # No Crisis, Possible Crisis, Crisis

# Preprocessing functions
def clean_text(text):
    """Perform text cleaning: lowercasing, emoji conversion, punctuation removal."""
    text = text.lower()
    text = emoji.demojize(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(messages):
    """Tokenize and preprocess SMS messages."""
    messages = [clean_text(msg) for msg in messages]
    return tokenizer(messages, padding=True, truncation=True, return_tensors="pt")

def classify_message(model, messages):
    """Runs classification on a specific model."""
    inputs = preprocess_text(messages)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = F.softmax(outputs.logits, dim=-1)
    classifications = torch.argmax(probs, dim=1).tolist()
    return classifications, probs.tolist()

def classify_sms(sms_messages):
    """Classifies SMS messages and returns a DataFrame."""
    sentiment_predictions, sentiment_probs = classify_message(sentiment_model, sms_messages)
    response_predictions, response_probs = classify_message(response_model, sms_messages)
    crisis_predictions, crisis_probs = classify_message(crisis_model, sms_messages)
    
    sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    response_labels = {0: "No Reply Needed", 1: "Reply Needed"}
    crisis_labels = {0: "No Crisis", 1: "Possible Crisis", 2: "Crisis Detected"}
    
    results = []
    for i, msg in enumerate(sms_messages):
        
        flag_for_review = sentiment_probs[i][1] < 0.6 or crisis_probs[i][2] > 0.8
        results.append({
            "Message": msg,
            "Sentiment": sentiment_labels[sentiment_predictions[i]],
            "Response Needed": response_labels[response_predictions[i]],
            "Crisis Level": crisis_labels[crisis_predictions[i]],
            "Sentiment Confidence": sentiment_probs[i],
            "Response Confidence": response_probs[i],
            "Crisis Confidence": crisis_probs[i],
            "Flag for Review": flag_for_review
        })
    
    return pd.DataFrame(results)


# Streamlit UI
st.title("SMS Grief Support Classification")
user_input = st.text_area("Enter SMS messages (one per line)")
if st.button("Classify Messages"):
    messages = user_input.split("\n")
    messages = [msg.strip() for msg in messages if msg.strip()]
    if messages:
        df = classify_sms(messages)
        st.dataframe(df.T.style.set_properties(**{'text-align': 'center'}).set_table_styles(
            [{'selector': 'th', 'props': [('background-color', '#f4f4f4'), ('text-align', 'center')]}])).set_table_styles(
            [{'selector': 'th', 'props': [('background-color', '#f4f4f4'), ('text-align', 'center')]}])
