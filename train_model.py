import pandas as pd
import pickle
import re
import os
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ===============================
# Download NLTK resources
# ===============================
nltk.download("stopwords")
nltk.download("wordnet")


# ===============================
# Create NLP tools
# ===============================
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ===============================
# Text Cleaning Function
# ===============================
def clean_text(text):

    # remove links
    text = re.sub(r"http\S+", "", text)

    # remove special characters
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # lowercase
    text = text.lower()

    words = text.split()

    # remove stopwords and apply lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ===============================
# Load Dataset
# ===============================
print("Loading dataset...")

data = pd.read_csv("dataset/UpdatedResumeDataSet.csv")

print("Dataset loaded successfully")
print("Total samples:", len(data))


# ===============================
# Clean Resume Text
# ===============================
print("Cleaning text...")

data["Resume"] = data["Resume"].apply(clean_text)


# ===============================
# Input and Target
# ===============================
X = data["Resume"]
y = data["Category"]


# ===============================
# TF-IDF Vectorization
# ===============================
print("Vectorizing text...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1,2)
)

X_vectorized = vectorizer.fit_transform(X)


# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)


# ===============================
# Model (Logistic Regression)
# ===============================
print("Training model...")

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


# ===============================
# Prediction
# ===============================
y_pred = model.predict(X_test)


# ===============================
# Evaluation
# ===============================
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ===============================
# Save Model
# ===============================
if not os.path.exists("model"):
    os.makedirs("model")

pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved successfully!")