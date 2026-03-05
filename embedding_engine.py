from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-base")

def embed_text(texts):
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings