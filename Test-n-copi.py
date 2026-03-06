import os
import time
import hashlib
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import streamlit as st
import fitz
import pytesseract
import cv2
import numpy as np
import requests
import torch
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
TESS_CONFIG = "--oem 3 --psm 6"  # Good general choice for printed multilingual text

OLLAMA_BASE_URL = "http://localhost:8890/v1"
OLLAMA_MODEL = "meta/llama-3.1-8b-instruct"
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

MAX_IMG_SIDE = 2000  # Cap long side for faster OCR without hurting quality much
PDF_OCR_TEXT_THRESHOLD = 60  # Only OCR pages whose extracted text is shorter than this
CONTEXT_CHAR_LIMIT = 12000    # Protect your LLM context window and latency
HISTORY_LIMIT = 6

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

# ---------------- HELPERS ----------------
def normalize_text(s: str) -> str:
    # Trim & compress whitespace to reduce token count
    return " ".join(s.split())

def limit_chars(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars]

def image_cv_to_pil(img: np.ndarray) -> Image.Image:
    # BGR -> RGB
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def fitz_pixmap_to_pil(pix) -> Image.Image:
    mode = "RGB" if pix.n in [3, 4] else "L"
    if pix.n == 4:
        # Drop alpha for OCR speed/consistency
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 4)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return Image.fromarray(img)
    else:
        return Image.frombytes(mode, (pix.width, pix.height), pix.samples)

def resize_long_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_side:
        return img
    scale = max_side / long_side
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

# ---------------- LOAD SURYA ----------------
@st.cache_resource(show_spinner=False)
def load_surya():
    return load_predictors(device=DEVICE)

predictors = load_surya()

# ---------------- EMBEDDINGS (GPU, cached) ----------------
@st.cache_resource(show_spinner=False)
def get_embeddings(model_name: str = "intfloat/multilingual-e5-base"):
    # Run on GPU if available, normalize vectors for cosine sim stability, batch for speed
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )

embeddings = get_embeddings()

# ---------------- ASK LLM ----------------
def ask_llama(prompt: str):
    url = f"{OLLAMA_BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "Answer ONLY from the given context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "top_p": 0.9,
    }

    # ---- Optional streaming (uncomment to enable) ----
    # payload["stream"] = True
    # with requests.post(url, headers=headers, json=payload, stream=True) as r:
    #     r.raise_for_status()
    #     chunks = []
    #     for line in r.iter_lines():
    #         if not line: continue
    #         # parse Server-Sent Events line(s) here if your server outputs OpenAI-compatible chunks
    #         # chunks.append(...)
    #     return "".join(chunks)

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ---------------- OCR: PYTESSERACT (CPU) ----------------
def pytess_ocr_one(np_img: np.ndarray, lang=TESS_LANG, config=TESS_CONFIG) -> str:
    if np_img.ndim == 3 and np_img.shape[2] == 4:
        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    # Binarize with Otsu (generally good for printed multilingual)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return pytesseract.image_to_string(bin_img, lang=lang, config=config)

# ---------------- PDF OCR ----------------
def extract_pdf_text(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages_needing_ocr = []
    docs = []

    # 1) Try fast text extraction first
    for page_num, page in enumerate(doc):
        text = page.get_text("text") or ""
        if len(normalize_text(text)) < PDF_OCR_TEXT_THRESHOLD:
            pages_needing_ocr.append((page_num, page))
        else:
            docs.append(Document(page_content=normalize_text(text), metadata={"page": page_num + 1}))

    if not pages_needing_ocr:
        return docs

    # 2) OCR only pages that need it
    st.info(f"OCR needed for {len(pages_needing_ocr)} page(s)")

    # Render page images (1x scale is typically sufficient for OCR; avoid 2x to save time)
    pil_images = []
    page_indices = []
    for page_num, page in pages_needing_ocr:
        pix = page.get_pixmap(alpha=False)
        pil_img = fitz_pixmap_to_pil(pix)
        pil_img = resize_long_side(pil_img, MAX_IMG_SIDE)
        pil_images.append(pil_img)
        page_indices.append(page_num)

    if DEVICE.startswith("cuda"):
        # ---- Batch OCR with Surya (fast on GPU) ----
        rec = predictors["recognition"](
            images=pil_images,
            task_names=["ocr_with_boxes"],
            det_predictor=predictors["detection"]
        )
        for idx, page_num in enumerate(page_indices):
            text = " ".join([line.text for line in rec[idx].text_lines])
            docs.append(Document(page_content=normalize_text(text), metadata={"page": page_num + 1}))
    else:
        # ---- Parallel OCR with Tesseract (CPU) ----
        np_images = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in pil_images]
        with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 1)) as ex:
            futures = {ex.submit(pytess_ocr_one, arr): page_num for arr, page_num in zip(np_images, page_indices)}
            for fut in as_completed(futures):
                page_num = futures[fut]
                try:
                    text = fut.result()
                except Exception as e:
                    text = ""
                docs.append(Document(page_content=normalize_text(text), metadata={"page": page_num + 1}))

    # Keep page order
    docs.sort(key=lambda d: d.metadata["page"])
    return docs

# ---------------- SURYA OCR (Images) ----------------
def surya_ocr_image(image: Image.Image):
    image = resize_long_side(image.convert("RGB"), MAX_IMG_SIDE)
    rec = predictors["recognition"](
        images=[image],
        task_names=["ocr_with_boxes"],
        det_predictor=predictors["detection"]
    )
    text = " ".join([line.text for line in rec[0].text_lines])
    return [Document(page_content=normalize_text(text), metadata={"page": 1})]

# ---------------- VECTOR STORE ----------------
@st.cache_resource(show_spinner=False)
def build_or_load_vectorstore(persist_dir: str, collection_name: str, docs: list[Document] | None):
    """
    If Chroma collection exists at persist_dir/collection_name, load it.
    Otherwise, build from `docs`. Pass docs=None to load only.
    """
    if os.path.isdir(persist_dir):
        try:
            vs = Chroma(
                persist_directory=persist_dir,
                collection_name=collection_name,
                embedding_function=embeddings
            )
            # A quick probe to ensure it’s valid
            _ = vs._collection.count()
            return vs
        except Exception:
            pass  # fall through to rebuild

    if not docs:
        raise ValueError("No docs provided to build the vector store.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,    # Smaller chunks can improve retrieval recall
        chunk_overlap=120, # Slightly smaller overlap for fewer tokens
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    return vs

# ---------------- FORMAT DOCS ----------------
def format_docs(docs, char_limit=CONTEXT_CHAR_LIMIT):
    parts = []
    total_chars = 0
    for doc in docs:
        page = doc.metadata.get("page", "?")
        frag = f"[Page {page}] {doc.page_content}\n\n"
        if total_chars + len(frag) > char_limit:
            break
        parts.append(frag)
        total_chars += len(frag)
    return "".join(parts).strip()

# ---------------- CHAT HISTORY ----------------
def build_history():
    # limit last N messages
    last_msgs = st.session_state.messages[-HISTORY_LIMIT:]
    history = []
    for msg in last_msgs:
        role = "User" if msg["role"] == "user" else "Assistant"
        history.append(f"{role}: {normalize_text(msg['content'])}")
    return "\n".join(history)

# ---------------- RAG CHAIN ----------------
def init_rag_chain(retriever):
    # Use MMR retriever via LangChain (diversify to reduce redundant chunks)
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
""".strip()
    )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(lambda docs: format_docs(docs, CONTEXT_CHAR_LIMIT)),
            "question": RunnablePassthrough(),
            "history": RunnableLambda(lambda _: build_history()),
        }
        | prompt
        | RunnableLambda(lambda x: ask_llama(x.to_string()))
        | StrOutputParser()
    )

    return rag_chain

# ---------------- GENERATE ANSWER ----------------
def generate_answer(question: str):
    return st.session_state.rag_chain.invoke(question)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload PDF or Image",
    type=["pdf", "jpg", "jpeg", "png"]
)

if uploaded_file:
    file_bytes = uploaded_file.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    uploaded_file.seek(0)

    # avoid rebuilding for the same file
    if file_hash != st.session_state.current_file_hash:
        suffix = Path(uploaded_file.name).suffix.lower()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(file_bytes)
        tmp.close()
        file_path = tmp.name

        persist_dir = f"./chroma_store/{file_hash}"
        collection_name = f"collection_{file_hash}"

        with st.spinner("Processing file..."):
            if suffix == ".pdf":
                st.info("PDF detected → Extract text; OCR only where needed")
                docs = extract_pdf_text(file_path)
            else:
                st.info("Image detected → Surya OCR")
                image = Image.open(file_path).convert("RGB")
                docs = surya_ocr_image(image)

            # Build or load vector store
            vectorstore = build_or_load_vectorstore(
                persist_dir=persist_dir,
                collection_name=collection_name,
                docs=docs
            )

            # Use MMR for better diversity; lower k (3–4) for speed
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.4}
            )

            st.session_state.rag_chain = init_rag_chain(retriever)
            st.session_state.current_file_hash = file_hash

        try:
            os.remove(file_path)
        except Exception:
            pass

        st.success("Document ready!")

# ---------------- CHAT ----------------
if st.session_state.rag_chain:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about the document...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = generate_answer(user_input)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
