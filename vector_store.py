import chromadb
from embedding_engine import embed_text

client = chromadb.Client()

collection = client.get_or_create_collection(name="financial_docs")

def add_documents(chunks):
    embeddings = embed_text(chunks)
    ids = [str(i) for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist()
    )