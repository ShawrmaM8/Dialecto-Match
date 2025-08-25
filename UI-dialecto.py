import streamlit as st
import pandas as pd
import random
import pickle

# Load model/vectorizer
clf = pickle.load(open(r"C:\Users\muzam\OneDrive\Desktop\PROJECTS\Dialecto Match\d-model\d_model.pkl", "rb"))
vectorizer = pickle.load(open(r"C:\Users\muzam\OneDrive\Desktop\PROJECTS\Dialecto Match\d-model\d_tfidf.pkl", "rb"))

# Load data
df = pd.read_json(r"C:\Users\muzam\OneDrive\Desktop\PROJECTS\Dialecto Match\d-data\dialectals.json")

# App
st.title("🗣 DialectoMatch: Guess the Dialect!")

sample = df.sample(1).iloc[0]
text = sample['text']
true_dialect = sample['dialect']

st.markdown(f"### What dialect is this?\n> {text}")

options = ['Levantine', 'Gulf', 'Maghrebi', 'Egyptian']
guess = st.radio("Choose one:", options)

if st.button("Submit"):
    pred = clf.predict(vectorizer.transform([text]))[0]

    if guess == true_dialect:
        st.success(f"✅ Correct! It *is* {true_dialect}.")
    else:
        st.error(f"❌ Nope! It’s actually {true_dialect}.")

    st.markdown("### 🔍 Features of this dialect:")
    st.markdown(get_dialect_features(true_dialect))

    # Optional:
    # st.markdown("### 🧠 How would this sound in another dialect?")

