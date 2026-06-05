📊 Sentiment Analysis Project
🧠 Overview

This project is a Sentiment Analysis system that classifies movie reviews as Positive or Negative using Machine Learning techniques.
It uses NLP preprocessing and a Logistic Regression model trained on the IMDB dataset.

🚀 Features
Text preprocessing (cleaning, stopword removal, etc.)
TF-IDF vectorization for feature extraction
Logistic Regression model for classification
Trained on IMDB movie review dataset
Simple prediction pipeline for new text input
Web app interface using Flask/Streamlit (if applicable)
📁 Project Structure
Sentiment_Analysis/
│
├── app/
│   └── app.py                  # Web app for prediction
│
├── data/
│   └── IMDB Dataset.csv        # Dataset used for training
│
├── models/
│   ├── Logistic Regression_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── models.py               # Model training script
│
├── notebooks/
│   └── sentiment_analysis.ipynb # Jupyter notebook (EDA + training)
│
├── src/
│   └── predict.py             # Prediction script
│
└── README.md
⚙️ Tech Stack
Python 🐍
Pandas & NumPy
Scikit-learn
NLP (TF-IDF Vectorizer)
Logistic Regression
Flask / Streamlit (for UI if used)
📊 Dataset
IMDB Movie Reviews Dataset
Contains labeled reviews:
Positive
Negative

🧠 Model Workflows

Load dataset
Clean and preprocess text
Convert text → numerical features using TF-IDF
Train Logistic Regression model
Evaluate performance
Save model & vectorizer using pickle
Use saved model for predictions

🔮 How to Run the Project

1️⃣ Clone repository
git clone https://github.com/karanSR-data/sentiment-analysis.git
cd sentiment-analysis
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run Jupyter Notebook (optional)
jupyter notebook notebooks/sentiment_analysis.ipynb
4️⃣ Run Web App
python app/app.py
5️⃣ Make Predictions (CLI)
python src/predict.py
📈 Example Output
Input: "This movie was amazing and full of emotions"
Prediction: Positive 😊

Input: "Worst movie I have ever seen"
Prediction: Negative 😡
⚠️ Notes
Dataset file is large (63MB), may trigger GitHub warnings.
Model files are saved using pickle.
Ensure all dependencies are installed before running.
📌 Future Improvements
Use Deep Learning (LSTM / BERT)
Improve accuracy with hyperparameter tuning
Deploy on HuggingFace / Streamlit Cloud
Add real-time sentiment dashboard
👨‍💻 Author

Karan Singh Rajput
GitHub: karanSR-data
