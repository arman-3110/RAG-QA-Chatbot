# 📊 Loan Approval Q&A Chatbot 🤖

This Streamlit app is an intelligent chatbot that uses **Retrieval-Augmented Generation (RAG)** and **Machine Learning** to answer questions about loan applicants and predict loan approval status based on user inputs.

[![Streamlit App](https://rag-bot-armanjindal.streamlit.app/)

---

## 🧠 Features

- 💬 **Question Answering (RAG Chatbot):**
  - Ask questions like:
    - *"How many applicants got their loan approved?"*
    - *"Who has the highest income?"*
    - *"Tell me about applicants with credit history 0"*
  - Uses a **Hugging Face FLAN-T5 model** + **FAISS** similarity search over tabular data transformed to natural text.

- 📈 **Loan Approval Prediction:**
  - Fill a form with applicant details.
  - Predicts whether the applicant is likely to get a loan.
  - Built using a supervised ML model (e.g., Logistic Regression / Random Forest).

---

## 🛠️ Tech Stack

| Component            | Tool/Library                |
|----------------------|-----------------------------|
| Web App              | Streamlit                   |
| Embedding Model      | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector Store         | FAISS                       |
| Language Model       | Hugging Face (`flan-t5-small`) |
| ML Prediction        | Scikit-learn + joblib       |
| Dataset Source       | [Kaggle - Loan Approval Prediction](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction) |

---


