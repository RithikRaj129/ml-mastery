# vector_db.py - Vector database patterns with FAISS and ChromaDB
# pip install faiss-cpu chromadb sentence-transformers numpy

import numpy as np
import time
from typing import List, Tuple

# ── FAISS: Facebook AI Similarity Search ─────────────────────────
import faiss

def demo_faiss():
    print('=== FAISS Vector Database Demo ===')

    # Simulate document embeddings (e.g. from sentence-transformers)
    np.random.seed(42)
    d = 384           # Embedding dimension (sentence-transformers default)
    n_docs = 100_000  # Number of documents in the database
    n_query = 5       # Number of query vectors
    k = 5             # Top-k results

    # Create embeddings
    print(f'Creating {n_docs:,} embeddings of dimension {d}...')
    embeddings = np.random.randn(n_docs, d).astype(np.float32)

    # Normalise for cosine similarity search
    faiss.normalize_L2(embeddings)

    # ── Flat (exact) index ────────────────────────────────────────
    index_flat = faiss.IndexFlatIP(d)  # Inner product = cosine similarity after L2 normalisation
    index_flat.add(embeddings)
    print(f'Flat index: {index_flat.ntotal:,} vectors')

    # ── IVF (Inverted File) index - much faster for large datasets ─
    n_centroids = 256  # Number of Voronoi cells
    n_probe     = 32   # How many cells to check (accuracy vs speed tradeoff)
    quantiser   = faiss.IndexFlatIP(d)
    index_ivf   = faiss.IndexIVFFlat(quantiser, d, n_centroids, faiss.METRIC_INNER_PRODUCT)
    index_ivf.train(embeddings)   # Must train IVF index
    index_ivf.add(embeddings)
    index_ivf.nprobe = n_probe

    # ── HNSW (Hierarchical Navigable Small World) - fastest ───────
    index_hnsw = faiss.IndexHNSWFlat(d, 32)  # 32 = M parameter
    index_hnsw.hnsw.efConstruction = 200
    index_hnsw.hnsw.efSearch = 64
    index_hnsw.add(embeddings)

    # ── Benchmark all three ───────────────────────────────────────
    queries = np.random.randn(n_query, d).astype(np.float32)
    faiss.normalize_L2(queries)

    print(f'\nSearching {n_query} queries (k={k}) across {n_docs:,} vectors:')
    print(f'{'Index':15s} {'Time (ms)':12s} {'QPS':10s} {'Notes'}')
    print('-'*65)

    for name, index in [('Flat (exact)',index_flat),('IVF256(nprobe=32)',index_ivf),('HNSW(M=32)',index_hnsw)]:
        t0 = time.time()
        n_trials = 100
        for _ in range(n_trials):
            D, I = index.search(queries, k)  # D=distances, I=indices
        elapsed = (time.time()-t0)/n_trials*1000
        qps = n_query/(elapsed/1000)
        print(f'{name:15s} {elapsed:>12.2f} {qps:>10.0f}')

    # ── Semantic search simulation ─────────────────────────────────
    print(f'\nTop-{k} most similar documents to query[0]:')
    D, I = index_hnsw.search(queries[:1], k)
    for rank, (idx, score) in enumerate(zip(I[0], D[0])):
        print(f'  Rank {rank+1}: doc_id={idx:6d}  similarity={score:.4f}')


# ── ChromaDB: modern vector DB with metadata filtering ────────────
def demo_chromadb():
    print('\n=== ChromaDB Demo ===')
    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.Client()  # In-memory for demo

        # Create collection
        collection = client.create_collection(
            name='products',
            metadata={'hnsw:space': 'cosine'}  # Cosine similarity
        )

        # Add documents with embeddings and metadata
        np.random.seed(42)
        n = 1000
        categories = ['electronics','clothing','sports','books','home']

        collection.add(
            embeddings=np.random.randn(n, 128).tolist(),
            documents=[f'Product description #{i}' for i in range(n)],
            metadatas=[{
                'product_id': i,
                'category':   np.random.choice(categories),
                'price':      float(np.random.uniform(10, 500)),
                'in_stock':   bool(np.random.choice([True, False]))
            } for i in range(n)],
            ids=[f'doc_{i}' for i in range(n)]
        )

        print(f'ChromaDB collection: {collection.count()} documents')

        # Vector search with metadata filter
        query_embedding = np.random.randn(1, 128).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=5,
            where={'category': 'electronics', 'in_stock': True},  # Metadata filter
            include=['documents','metadatas','distances']
        )

        print('\nTop 5 electronics in-stock results:')
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            print(f'  {i+1}. id={meta["product_id"]} price=${meta["price"]:.2f} dist={dist:.4f}')

    except ImportError:
        print('ChromaDB not installed: pip install chromadb')

demo_faiss()
demo_chromadb()

# ── RAG Architecture ─────────────────────────────────────────────
print('\n=== RAG (Retrieval-Augmented Generation) Pattern ===')
rag_steps = [
    'Offline: Embed all documents → store in vector DB',
    'User query arrives → embed query with same model',
    'Vector DB: ANN search → retrieve top-k most similar chunks',
    'LLM: prompt = system_prompt + retrieved_chunks + user_query',
    'LLM generates grounded response (not hallucinated)',
    'Optional: re-rank retrieved chunks with cross-encoder before LLM',
]
for i, step in enumerate(rag_steps, 1):
    print(f'  {i}. {step}')
