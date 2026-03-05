import streamlit as st
import fitz
from PIL import Image

from ocr_engine import extract_text
from hybrid_search import hybrid_search, build_bm25
from vector_store import add_documents
from reranker import rerank
from llm_engine import ask_llm

st.title("Financial PDF Chatbot")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")

    pages_text = []

    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        text = extract_text(img)
        pages_text.append(text)

    chunks = []

    for text in pages_text:
        for chunk in text.split("\n\n"):
            if len(chunk) > 40:
                chunks.append(chunk)

    add_documents(chunks)
    build_bm25(chunks)

    st.success("Document processed")

query = st.text_input("Ask a question")

if query:
    docs = hybrid_search(query)
    docs = rerank(query, docs)

    context = "\n".join(docs)

    answer = ask_llm(context, query)

    st.write(answer)