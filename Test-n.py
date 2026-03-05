import streamlit as st
import fitz
import pytesseract
import cv2
import numpy as np
import tempfile
import hashlib
import os
import time
import requests
import torch
from pathlib import Path
from PIL import Image

# -------- Surya OCR --------
from surya.models import load_predictors

# -------- LangChain --------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# ---------------- CONFIG ----------------
TESS_LANG = "eng+hin+arb"
OLLAMA_BASE_URL = "http://localhost:8890/v1"
OLLAMA_MODEL = "meta/llama-3.1-8b-instruct"
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

st.set_page_config(page_title="Chat with PDF/Image", layout="wide")
st.title("💬 Chat with PDF or Image (Hindi + English + Arabic)")

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "current_file_hash" not in st.session_state:
    st.session_state.current_file_hash = None

# ---------------- DEVICE ----------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Device: {DEVICE}")

# ---------------- LOAD SURYA ----------------
@st.cache_resource
def load_surya():
    return load_predictors(device=DEVICE)

predictors = load_surya()

# ---------------- ASK LLM ----------------
def ask_llama(prompt: str):

    url = f"{OLLAMA_BASE_URL}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OLLAMA_API_KEY}"
    }

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "Answer ONLY from the given context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()

    return r.json()["choices"][0]["message"]["content"]

# ---------------- PDF OCR ----------------
def extract_pdf_text(pdf_path):

    docs = []
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc):

        text = page.get_text()

        # fallback OCR
        if len(text.strip()) < 50:

            pix = page.get_pixmap()

            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            text = pytesseract.image_to_string(gray, lang=TESS_LANG)

        docs.append(
            Document(
                page_content=text,
                metadata={"page": page_num + 1}
            )
        )

    return docs


# ---------------- SURYA OCR ----------------
def surya_ocr(image):

    rec = predictors["recognition"](
        images=[image],
        task_names=["ocr_with_boxes"],
        det_predictor=predictors["detection"]
    )

    text = " ".join([line.text for line in rec[0].text_lines])

    return [
        Document(
            page_content=text,
            metadata={"page": 1}
        )
    ]

# ---------------- VECTOR STORE ----------------
@st.cache_resource
def build_vectorstore(docs, persist_dir):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base"
    )



    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    return vectorstore


# ---------------- FORMAT DOCS ----------------
def format_docs(docs):

    formatted = []

    for doc in docs:
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Page {page}] {doc.page_content}")

    return "\n\n".join(formatted)


# ---------------- CHAT HISTORY ----------------
def build_history():

    history = ""

    # limit last 6 messages
    for msg in st.session_state.messages[-6:]:

        role = "User" if msg["role"] == "user" else "Assistant"

        history += f"{role}: {msg['content']}\n"

    return history


# ---------------- RAG CHAIN ----------------
def init_rag_chain(retriever):

    prompt = ChatPromptTemplate.from_template(
        """
Answer ONLY from the provided context.

Conversation History:
{history}

Context:
{context}

Question:
{question}

Rules:
- If answer not in context say "Not found in document"
- Cite page numbers when possible

Answer:
"""
    )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
            "history": RunnableLambda(lambda _: build_history())
        }
        | prompt
        | RunnableLambda(lambda x: ask_llama(x.to_string()))
        | StrOutputParser()
    )

    return rag_chain


# ---------------- GENERATE ANSWER ----------------
def generate_answer(question):

    answer = st.session_state.rag_chain.invoke(question)

    return answer


# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload PDF or Image",
    type=["pdf", "jpg", "jpeg", "png"]
)

if uploaded_file:

    file_bytes = uploaded_file.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    uploaded_file.seek(0)

    # avoid rebuilding
    if file_hash != st.session_state.current_file_hash:

        suffix = Path(uploaded_file.name).suffix.lower()

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(file_bytes)
        tmp.close()

        file_path = tmp.name

        with st.spinner("Processing file..."):

            if suffix == ".pdf":

                st.info("PDF detected → Tesseract OCR")

                docs = extract_pdf_text(file_path)

            else:

                st.info("Image detected → Surya OCR")

                image = Image.open(file_path).convert("RGB")

                docs = surya_ocr(image)

        persist_dir = f"./chroma_{file_hash}"

        vectorstore = build_vectorstore(docs, persist_dir)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        st.session_state.rag_chain = init_rag_chain(retriever)

        st.session_state.current_file_hash = file_hash

        os.remove(file_path)

        st.success("Document ready!")

# ---------------- CHAT ----------------
if st.session_state.rag_chain:

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about the document...")

    if user_input:

        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer = generate_answer(user_input)

                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
