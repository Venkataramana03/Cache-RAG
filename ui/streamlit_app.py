import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.title("Simple RAG with Groq")

uploaded_file = st.file_uploader("Upload a text file")

if uploaded_file:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "application/pdf",
        )
    }

    res = requests.post(f"{BACKEND_URL}/upload", files=files)

    if res.status_code == 200:
        st.success(res.json()["message"])
    else:
        st.error(f"Error: {res.status_code}")
        st.text(res.text)

question = st.text_input("Ask a question")

if st.button("Ask"):
    response = requests.post(
        f"{BACKEND_URL}/ask",
        json={"question": question}
    )
    st.write(response.json()["answer"])
