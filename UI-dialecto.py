import streamlit as st
import pandas as pd
import random
import pickle

# Load model/vectorizer
clf = pickle.load(open(r"C:\Users\muzam\OneDrive\Desktop\PROJECTS\Dialecto Match\d-model\d_model.pkl", "rb"))
vectorizer = pickle.load(open(r"C:\Users\muzam\OneDrive\Desktop\PROJECTS\Dialecto Match\d-model\d_tfidf.pkl", "rb"))

# Load data
df = pd.read_json(r"C:\Users\muzam\OneDrive\Desktop\PROJECTS\Dialecto Match\d-data\dialectals.json")

# Define get_dialect_features
def get_dialect_features(dialect: str, text: str, df: pd.DataFrame) -> str:
    """
    Fetch features & MSA translation from dataset for a given dialect + text
    """
    row = df(df['dialect'] == dialect) & (df['text'] == text)]

    if not row.empty:
        features = row.iloc[0]['features']
        msa_translation = row.iloc[0]['msa_translation']
        return f""
        **Dialect markers: ** {features}
        **MSA (فصحى): ** {msa_translation}
        ""
    else:
        return "No features found for this example."
        return "لم يتم العثور على أي ميزات لهذا المثال"
# App
st.title("🗣 DialectoMatch: Guess the Arabic Dialect!")

sample = df.sample(1).iloc[0]
text = sample['text']
true_dialect = sample['dialect']

st.markdown(f"### What dialect is this? ما هي لهجة هذه الجملة؟ \n> {text}")

options = ['Levantine شامي', 'Gulf خليجي', 'Maghrebi مغربي', 'Egyptian مصري', 'Iraqi عراقي']
guess = st.radio("Choose one اختر واحد :", options)

if st.button("Submit"):
    pred = clf.predict(vectorizer.transform([text]))[0]

    if guess == true_dialect:
        st.success(f"✅ Correct! It *is* {true_dialect}.")
    else:
        st.error(f"❌ Nope! It’s actually {true_dialect}.")

    st.markdown("### 🔍 Features of this dialect:")
    st.markdown(get_dialect_features(true_dialect, text, df))

    # Optional:
    # st.markdown("### 🧠 How would this sound in another dialect?")



