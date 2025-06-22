# 📰 News Article Classification (Fake/Real)

## 🔍 About the Project

This project aims to classify news articles as **Fake** or **Real** using Natural Language Processing (NLP) techniques and a Machine Learning model. In an age where misinformation spreads rapidly, this tool can help verify the authenticity of news content with ease and clarity.

## 🎯 Objective

To build an intelligent and interactive fake news detection system that:

* Accurately classifies news articles
* Provides a clean explanation of predictions
* Offers a live demo interface for real-time use

---

## 🛠️ Tools & Technologies

* **Language**: Python 3
* **Libraries**: NLTK, Scikit-learn, Pandas, NumPy, Joblib
* **Interface**: Streamlit (for GUI)
* **Model**: Logistic Regression
* **Vectorization**: TF-IDF

---

## ⚙️ Features

* 🧠 **Single Article Prediction** – Paste a news article and get prediction with explanation.
* 📁 **Bulk Prediction via CSV** – Upload a CSV of multiple articles for batch classification.
* 💡 **Explainable Output** – View important words and their weights in the prediction.
* 📥 **Download Report** – Get a classification report in `.txt` or `.csv`.
* 🎨 **Interactive UI** – Gradient background, animated UI, modern typography.

---

## 🚀 Live Demo (Optional)

> 🔗 Coming soon (can be deployed using [Streamlit Cloud](https://share.streamlit.io) or HuggingFace Spaces)

---

## 🧪 How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/your-username/News-Article-Classification-Fake-Real.git
cd News-Article-Classification-Fake-Real
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 📂 Folder Structure

```
├── app.py                         # Main Streamlit app
├── fake_news_model.pkl           # Trained Logistic Regression model
├── tfidf_vectorizer.pkl          # TF-IDF vectorizer
├── requirements.txt              # Required Python packages
├── README.md                     # Project overview
├── News_Article_Classification_Report.docx  # Final report
├── demo/
│   └── Fake_News_Demo_Recording.mp4         # Output video demo
├── data/
│   └── sample_news.csv                     # Sample input for testing
```

