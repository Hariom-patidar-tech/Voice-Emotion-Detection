import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocess import clean_text

# Load dataset
df = pd.read_csv("data/emotion_dataset.csv")

# Preprocess
df["clean_text"] = df["text"].apply(clean_text)

X = df["clean_text"]
y = df["emotion"]

# Vectorization
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_df=0.9,
    min_df=1
)


X_vec = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train Logistic Regression
model = LogisticRegression(
    max_iter=2000,
    C=2.0,
    solver='lbfgs',
    
)


model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, pred))

# Save model
pickle.dump(model, open("models/logistic_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/tfidf_vectorizer.pkl", "wb"))


