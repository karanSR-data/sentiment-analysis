import joblib
import re

# text cleaning

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|https\S+","", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/Logistic Regression_model.pkl")

def predict_sentiment(review):
    cleaned  = clean_text(review)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

     # Convert to meaningful label
    return "Positive 😀" if prediction == 1 else "Negative 😞"

if __name__ == "__main__":
    user_input = input("Enter your review: ")
    result = predict_sentiment(user_input)
    print("\nSentiment:", result)
