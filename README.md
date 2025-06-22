# 📰 News Article Classification (Fake/Real)

## 🔍 About the Project

This project presents an intelligent system designed to classify news articles as either **Fake** or **Real** using Natural Language Processing (NLP) and Machine Learning techniques. With the proliferation of misinformation across digital platforms, especially on social media, the need for automated and accurate fake news detection tools has become more pressing than ever. This project not only addresses that need but also offers a user-friendly and visually engaging interface.

## 🎯 Objective

To develop a robust machine learning pipeline that can accurately detect fake news articles and present the results through an interactive and visually appealing GUI, providing not only a verdict but also meaningful insight into the prediction.

---

## 🧠 Problem Statement

Misinformation and fake news have become global concerns due to their potential to manipulate public opinion and influence socio-political landscapes. Manual verification is time-consuming and often inefficient. The goal is to automate this process through AI and provide transparent and explainable predictions.

---

## 💡 Proposed Solution

Our solution involves building a supervised learning model trained on labeled news articles. We use Natural Language Processing for text cleaning and preprocessing, transform the content into numerical vectors using TF-IDF, and then classify the news using a Logistic Regression model. An engaging Streamlit web interface allows users to test the classifier live.

---

## 🛠️ Tools & Technologies Used

* **Programming Language**: Python 3
* **Natural Language Processing**: NLTK
* **Machine Learning**: Scikit-learn
* **Data Manipulation**: Pandas, NumPy
* **Model**: Logistic Regression
* **Vectorizer**: TF-IDF
* **User Interface**: Streamlit
* **Model Serialization**: Joblib

---

## 📋 Project Features

* **Single News Article Classification**: Users can paste a news article to receive an instant prediction.
* **CSV Upload for Bulk Classification**: Upload a `.csv` file containing multiple news articles for batch prediction.
* **Keyword-Based Explanation**: Top contributing words are highlighted with their relative weights, offering transparency in classification.
* **Result Report Generation**: Users can download a full report of their prediction for future reference.
* **Visually Enhanced GUI**: Gradient galaxy-inspired theme, animated elements, and responsive layout for a modern user experience.

---

## 🧪 Model Training Workflow

1. **Dataset Acquisition**: A labeled dataset containing real and fake news articles is sourced from Kaggle.
2. **Text Preprocessing**: Cleaning the news content by removing punctuation, converting to lowercase, tokenizing, removing stopwords, and lemmatizing.
3. **Feature Extraction**: Using TF-IDF to convert text into numerical feature vectors.
4. **Model Training**: Training a Logistic Regression model on the processed dataset.
5. **Model Evaluation**: Using metrics like Accuracy, Precision, Recall, and F1-Score.
6. **Model Deployment**: Integration into a Streamlit-based user interface for live testing and demonstration.

---

## 📁 Project Components

* `app.py` – Streamlit app that serves as the GUI.
* `fake_news_model.pkl` – Trained Logistic Regression model saved using joblib.
* `tfidf_vectorizer.pkl` – TF-IDF vectorizer fitted on the dataset.
* `News_Article_Classification_Report.docx` – Final documentation summarizing the project.
* `sample_news.csv` – Sample input for testing bulk classification.
* `Fake_News_Demo_Recording.mp4` – Screen recording of the application in action.

---

## 🖼️ GUI Preview

(images/"C:\Users\NISHA KUMARI\Pictures\Screenshots\Screenshot 2025-06-22 155106.png")

![Fake News Classifier GUI]("C:\Users\NISHA KUMARI\Pictures\Screenshots\Screenshot 2025-06-22 155106.png")


## 📌 Conclusion

The News Article Classification (Fake/Real) project successfully demonstrates how AI and NLP can be harnessed to address one of the modern era's most critical challenges: misinformation. The combination of a well-performing ML model, a clear explanation mechanism, and an attractive user interface makes it both impactful and user-centric. This project stands as a testament to how technology can empower individuals and organizations to make informed decisions based on verified information.


