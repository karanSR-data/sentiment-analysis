import joblib

# save vectorizer
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

# save the model
joblib.dump(results[best_model_name]["model"], f"models/{best_model_name}_model.p")

print("Model + vectorizer saved successfully!")