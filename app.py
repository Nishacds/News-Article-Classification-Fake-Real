import streamlit as st
import joblib
import pandas as pd
import datetime
import numpy as np

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page config
st.set_page_config(page_title="Fake News Detector", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');

    html, body, .stApp {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(135deg, #000000, #1a0033, #330033);
        background-size: 300% 300%;
        animation: neonBG 20s ease infinite;
        color: #00ffe7;
    }

    @keyframes neonBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    header, footer {visibility: hidden;}

    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 850px;
        margin: auto;
        background: rgba(255, 255, 255, 0.02);
        border: 2px solid #00f0ff22;
        border-radius: 20px;
        box-shadow: 0 0 20px #00f0ff55;
        backdrop-filter: blur(8px);
    }

    h1 {
        text-align: center;
        font-size: 3rem;
        color: #ff00ff;
        text-shadow: 0 0 15px #ff00ff;
        margin-bottom: 0.2em;
    }

    h4 {
        text-align: center;
        font-weight: 400;
        color: #cccccc;
        margin-bottom: 2rem;
    }

    .stTextArea textarea {
        background-color: #1a1a2e !important;
        color: #00ffe7 !important;
        font-size: 16px !important;
        border-radius: 12px !important;
        border: 1px solid #00ffe766;
        padding: 1rem !important;
        box-shadow: 0 0 12px #00ffe744;
    }

    .stButton>button {
        background: linear-gradient(to right, #ff00cc, #3333ff);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 30px;
        font-weight: bold;
        box-shadow: 0 0 15px #ff00cc88;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background: linear-gradient(to left, #ff00cc, #3333ff);
        transform: scale(1.07);
        box-shadow: 0 0 25px #ff00ccbb;
    }

    .stDownloadButton>button {
        background: linear-gradient(to right, #ff5500, #ff0066);
        color: white;
        font-weight: bold;
        padding: 10px 25px;
        border-radius: 10px;
        box-shadow: 0 0 15px #ff3366aa;
    }

    .stDownloadButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(to left, #ff5500, #ff0066);
        box-shadow: 0 0 25px #ff3366dd;
    }

    .stFileUploader {
        background-color: #111122;
        border: 1px solid #00ffe7;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 0 10px #00ffe766;
    }

    .stDataFrame {
        background-color: #1f1f2f;
        border-radius: 10px;
        padding: 1rem;
        color: #00ffe7;
    }
    </style>
""", unsafe_allow_html=True)

with st.container():
    
    st.markdown("<h1>Fake News Classification</h1>", unsafe_allow_html=True)
    st.markdown("<h4>Detect whether a news article is Real or Fake using AI</h4>", unsafe_allow_html=True)

    user_input = st.text_area("Paste a news article below:", height=200)

    if st.button("Predict Single Article"):
        if user_input.strip() == "":
            st.warning("Please enter news content.")
        else:
            input_vec = vectorizer.transform([user_input])
            result = model.predict(input_vec)[0]

            if result == 1:
                verdict = "This news is most likely REAL."
                st.success(verdict)
            else:
                verdict = "This news is most likely FAKE."
                st.error(verdict)

            st.markdown("### Explanation:")

            feature_names = vectorizer.get_feature_names_out()
            log_probs = model.feature_log_prob_
            log_diff = log_probs[1] - log_probs[0]

            word_scores = []
            for idx in input_vec.nonzero()[1]:
                word = feature_names[idx]
                score = log_diff[idx]
                word_scores.append((word, score))

            top_words = sorted(word_scores, key=lambda x: abs(x[1]), reverse=True)[:5]
            real_words = [w for w, s in top_words if s > 0]
            fake_words = [w for w, s in top_words if s < 0]

            if result == 1:
                if real_words:
                    joined = ", ".join([f"{w}" for w in real_words])
                    st.markdown(
                        f"The model detected trustworthy terms like {joined}, "
                        "which are commonly used in reliable journalism. "
                        "These keywords suggest the content discusses verifiable facts, "
                        "institutional information, or neutral events."
                    )
                else:
                    st.markdown(
                        "The model didn’t detect specific high-confidence words, "
                        "but the overall style and vocabulary suggest the article is factual or informative."
                    )
            else:
                if fake_words:
                    joined = ", ".join([f"{w}" for w in fake_words])
                    st.markdown(
                        f"The model noticed suspicious or sensational words like {joined}, "
                        "which are typical of misleading or clickbait-style articles. "
                        "Such language is often used in fake news to provoke emotional reactions or spread misinformation."
                    )
                else:
                    st.markdown(
                        "The article lacks identifiable trustworthy signals and contains patterns "
                        "similar to texts known to spread misinformation or exaggeration."
                    )

            with st.expander("View contributing words and their influence"):
                for word, score in top_words:
                    direction = "REAL" if score > 0 else "FAKE"
                    st.markdown(f"- {word} → supports {direction} (score: {score:.4f})")

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report = f"""
Fake News Classification Report
-------------------------------
Checked on: {timestamp}

News Content:
{user_input}

Result:
{verdict}

Top Influential Words:
""" + "\n".join([f"- {word} (score: {score:.4f})" for word, score in top_words])

            st.download_button("Download Result Report", report, file_name="fake_news_report.txt")

    st.markdown("---")

    st.subheader("Upload CSV for Bulk News Classification")
    st.caption("Upload a `.csv` file with a column named 'text'.")

    uploaded_file = st.file_uploader("Upload your file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "text" not in df.columns:
                st.error("CSV must contain a column named 'text'.")
            else:
                vectors = vectorizer.transform(df["text"])
                predictions = model.predict(vectors)
                df["Prediction"] = ["REAL" if p == 1 else "FAKE" for p in predictions]
                st.success("Predictions completed.")

                st.dataframe(df[["text", "Prediction"]].head(10))

                csv_result = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results as CSV", csv_result, file_name="predicted_news.csv", mime='text/csv')
        except Exception as e:
            st.error(f"Something went wrong: {e}")
