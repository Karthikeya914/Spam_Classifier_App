import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data (only needs to run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
# Initialize Porter Stemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    stop_words = set(stopwords.words('english'))
    for i in text:
        if i not in stop_words:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return y

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Streamlit app UI
st.title("📩 Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message to classify.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = vectorizer.transform([" ".join(transformed_sms)]).toarray()
        result = model.predict(vector_input)[0]
        prob = model.predict_proba(vector_input)[0]

        if prob[1]>0.3:
            st.error(f"🚨 Spam (Probability: {round(prob[1]*100, 2)}%)")
        else:
            st.success(f"✅ Ham (Probability: {round(prob[0]*100, 2)}%)")