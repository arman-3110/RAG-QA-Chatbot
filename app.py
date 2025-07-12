import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import joblib


# ------------------ Classification & Filter Logic ------------------

def classify_question(question: str):
    q = question.lower()

    if "how many" in q and "loan approved" in q:
        return "count_approved"
    elif "credit history 0" in q or "credit history = 0" in q:
        return "filter_credit_0"
    elif "average loan amount" in q:
        return "average_loan"
    elif "highest income" in q:
        return "highest_income"
    elif "highest coapplicant income" in q:
        return "highest_coapplicant_income"
    elif "largest loan amount" in q:
        return "largest_loan_amount"
    elif "shortest loan term" in q:
        return "shortest_loan_term"
    elif "lowest income" in q:
        return "lowest_income"
    else:
        return "rag"



def count_approved():
    df = pd.read_csv('Training Dataset.csv')
    return f"Total applicants who got their loan approved: {df[df['Loan_Status'] == 'Y'].shape[0]}"

def filter_credit_0():
    df = pd.read_csv('Training Dataset.csv')
    filtered = df[df['Credit_History'] == 0]
    if filtered.empty:
        return "No applicants found with Credit History 0."
    return "\n\n".join(
        f"{row['Loan_ID']} - {row['Gender']}, {row['Married']}, Income: {row['ApplicantIncome']}, Status: {row['Loan_Status']}"
        for _, row in filtered.iterrows()
    )

def average_loan():
    df = pd.read_csv('Training Dataset.csv')
    avg = df['LoanAmount'].mean()
    return f"The average loan amount is approximately ₹{round(avg, 2)}."


def highest_income():
    df = pd.read_csv('Training Dataset.csv')
    df['ApplicantIncome'] = pd.to_numeric(df['ApplicantIncome'], errors='coerce')
    max_row = df.loc[df['ApplicantIncome'].idxmax()]
    return (
        f"The highest applicant income is ₹{int(max_row['ApplicantIncome'])}, "
        f"by applicant {max_row['Loan_ID']} ({max_row['Gender']}, {max_row['Education']}, "
        f"{'Married' if max_row['Married'] == 'Yes' else 'Unmarried'})."
    )

def highest_coapplicant_income():
    df = pd.read_csv('Training Dataset.csv')
    df['CoapplicantIncome'] = pd.to_numeric(df['CoapplicantIncome'], errors='coerce')
    max_row = df.loc[df['CoapplicantIncome'].idxmax()]
    return (
        f"The highest coapplicant income is ₹{int(max_row['CoapplicantIncome'])}, "
        f"by applicant {max_row['Loan_ID']} ({max_row['Gender']}, {max_row['Education']})."
    )

def largest_loan_amount():
    df = pd.read_csv('Training Dataset.csv')
    df['LoanAmount'] = pd.to_numeric(df['LoanAmount'], errors='coerce')
    max_row = df.loc[df['LoanAmount'].idxmax()]
    return (
        f"The largest loan amount is ₹{int(max_row['LoanAmount'])}, "
        f"taken by applicant {max_row['Loan_ID']} ({max_row['Gender']}, {max_row['Education']})."
    )

def shortest_loan_term():
    df = pd.read_csv('Training Dataset.csv')
    df['Loan_Amount_Term'] = pd.to_numeric(df['Loan_Amount_Term'], errors='coerce')
    min_row = df.loc[df['Loan_Amount_Term'].idxmin()]
    return (
        f"The shortest loan term is {int(min_row['Loan_Amount_Term'])} months, "
        f"for applicant {min_row['Loan_ID']} ({min_row['Gender']}, {min_row['Education']})."
    )

def lowest_income():
    df = pd.read_csv('Training Dataset.csv')
    df['ApplicantIncome'] = pd.to_numeric(df['ApplicantIncome'], errors='coerce')
    min_row = df.loc[df['ApplicantIncome'].idxmin()]
    return (
        f"The lowest applicant income is ₹{int(min_row['ApplicantIncome'])}, "
        f"by applicant {min_row['Loan_ID']} ({min_row['Gender']}, {min_row['Education']})."
    )



# ------------------ Load Data and Preprocess ------------------
@st.cache_resource
def load_data():
    df = pd.read_csv('Training Dataset.csv')
    df.fillna('missing', inplace=True)
    knowledge_base = df.apply(lambda row: (
        f"Applicant {row['Loan_ID']}: {row['Gender']}, "
        f"{'Married' if row['Married'] == 'Yes' else 'Unmarried'}, "
        f"{row['Dependents']} dependents, {row['Education']}, "
        f"{'Self-Employed' if row['Self_Employed'] == 'Yes' else 'Not Self-Employed'}, "
        f"Applicant Income: {row['ApplicantIncome']}, "
        f"Coapplicant Income: {row['CoapplicantIncome']}, "
        f"Loan Amount: {row['LoanAmount']}, Loan Term: {row['Loan_Amount_Term']}, "
        f"Credit History: {row['Credit_History']}, Property Area: {row['Property_Area']}, "
        f"Loan Status: {'Approved' if row['Loan_Status'] == 'Y' else 'Not Approved'}."
    ), axis=1).tolist()
    return knowledge_base

knowledge_base = load_data()

# ------------------ Load Embedding Model and FAISS ------------------
@st.cache_resource
def setup_faiss(knowledge_base):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(knowledge_base, convert_to_numpy=True, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return embedder, embeddings, index

embedder, embeddings, index = setup_faiss(knowledge_base)

# ------------------ Load Hugging Face Model ------------------
@st.cache_resource
def load_hf_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return pipe

qa_pipeline = load_hf_model()

@st.cache_resource
def load_model():
    model = joblib.load('loan_model.pkl')
    encoders = joblib.load('encoders_dict.pkl')  # ✅ now it’s a dictionary
    return model, encoders


loan_model, encoders = load_model()



# ------------------ Streamlit App UI ------------------
st.title("📊 Loan Approval Q&A Chatbot 🤖")

# user_question = st.text_input("Ask a question related to the loan approval dataset:")

# # ------------------ Sample Questions Dropdown ------------------
# sample_questions = [
#     "How many applicants got their loan approved?",
#     "Tell me about applicants with credit history 0",
#     "What is the average loan amount?",
#     "Who has the highest income?",
#     "Who has the highest coapplicant income?",
#     "Who took the largest loan amount?",
#     "Who has the shortest loan term?",
#     "Who has the lowest income?",
# ]

# ------------------ Combined Dropdown + Custom Input ------------------
# ------------------ Unified Suggestion + Text Input ------------------
sample_questions = [
    "How many applicants got their loan approved?",
    "Tell me about applicants with credit history 0",
    "What is the average loan amount?",
    "Who has the highest income?",
    "Who has the highest coapplicant income?",
    "Who took the largest loan amount?",
    "Who has the shortest loan term?",
    "Who has the lowest income?",
]

user_question = st.selectbox(
    "💡 Ask a question related to the loan approval dataset:",
    options=[""] + sample_questions,
    index=0,
    placeholder="Type or select a question...",
)


if st.button("Get Answer"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        rule = classify_question(user_question)

        if rule == "count_approved":
            st.subheader("Answer:")
            st.write(count_approved())

        elif rule == "filter_credit_0":
            st.subheader("Answer:")
            st.write(filter_credit_0())

        elif rule == "average_loan":
            st.subheader("Answer:")
            st.write(average_loan())

        elif rule == "highest_income":
            st.subheader("Answer:")
            st.write(highest_income())



        elif rule == "highest_coapplicant_income":
            st.subheader("Answer:")
            st.write(highest_coapplicant_income())

        elif rule == "largest_loan_amount":
            st.subheader("Answer:")
            st.write(largest_loan_amount())

        elif rule == "shortest_loan_term":
            st.subheader("Answer:")
            st.write(shortest_loan_term())

        elif rule == "lowest_income":
            st.subheader("Answer:")
            st.write(lowest_income())



        else:
            # RAG fallback
            query_embedding = embedder.encode([user_question])
            distances, indices = index.search(query_embedding, 3)
            retrieved_docs = [knowledge_base[i] for i in indices[0]]

            context = " ".join(retrieved_docs)
            prompt = f"Question: {user_question} Context: {context}"

            response = qa_pipeline(prompt, max_length=200, do_sample=False)
            answer = response[0]['generated_text']

            st.subheader("Answer:")
            st.write(answer)

            with st.expander("🔍 Retrieved Context"):
                for doc in retrieved_docs:
                    st.write(doc)


st.header("📈 Loan Approval Prediction")

with st.form("loan_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Term (months)", min_value=0)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    submit = st.form_submit_button("Predict Loan Approval")

if submit:
    # Encode inputs
    input_dict = {
    'Gender': encoders['Gender'].transform([gender])[0],
    'Married': encoders['Married'].transform([married])[0],
    'Dependents': encoders['Dependents'].transform([dependents])[0],
    'Education': encoders['Education'].transform([education])[0],
    'Self_Employed': encoders['Self_Employed'].transform([self_employed])[0],
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_term,
    'Credit_History': credit_history,
    'Property_Area': encoders['Property_Area'].transform([property_area])[0],
}


    

    input_df = pd.DataFrame([input_dict])
    prediction = loan_model.predict(input_df)[0]
    result = "✅ Approved" if prediction == 1 else "❌ Not Approved"

    st.subheader("Prediction Result:")
    st.write(result)
