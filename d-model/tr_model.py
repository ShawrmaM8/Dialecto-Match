import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_json(r"C:\Users\muzam\OneDrive\Desktop\PROJECTS\Dialecto Match\d-data\dialectals.json")
df = pd.DataFrame(df)

X_tr, X_te, y_tr, y_te = train_test_split(df['text'], df['dialect'], test_size=0.2)

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=10000)
X_tr_vec = vectorizer.fit_transform(X_tr)

model = LogisticRegression(max_iter=1000)
model.fit(X_tr_vec, y_tr)

# Save model and vectorizer
with open(r"C:\Users\muzam\OneDrive\Desktop\PROJECTS\Dialecto Match\d-model\d_model.pkl", "wb") as m:
    pickle.dump(model, m)

with open(r"C:\Users\muzam\OneDrive\Desktop\PROJECTS\Dialecto Match\d-model\d_tfidf.pkl", "wb") as v:
    pickle.dump(vectorizer, v)
