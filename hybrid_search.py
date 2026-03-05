from rank_bm25 import BM25Okapi
from embedding_engine import embed_text
from vector_store import collection

bm25 = None
corpus = []

def build_bm25(chunks):
    global bm25, corpus
    corpus = [c.split() for c in chunks]
    bm25 = BM25Okapi(corpus)

def hybrid_search(query, k=10):
    query_tokens = query.split()
    bm25_scores = bm25.get_scores(query_tokens)

    query_emb = embed_text([query])[0]

    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=k
    )

    vector_docs = results["documents"][0]

    bm25_top = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]

    bm25_docs = [" ".join(corpus[i]) for i in bm25_top]

    return vector_docs + bm25_docs