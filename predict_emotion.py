import pickle
from src.preprocess import clean_text

logistic = pickle.load(open("models/logistic_model.pkl", "rb"))
nb = pickle.load(open("models/nb_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

def predict_emotion(text, model_type="logistic"):
    text = clean_text(text)
    vec = vectorizer.transform([text])

    if model_type == "nb":
        return nb.predict(vec)[0]
    return logistic.predict(vec)[0]
