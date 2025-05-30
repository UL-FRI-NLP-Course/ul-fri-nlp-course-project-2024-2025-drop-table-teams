import os

from langchain.vectorstores import FAISS


def init_retriever(vectorstore: FAISS, k: int = None, th: float = None):
    k_value = k if k is not None else int(os.getenv("vectorstore_k", 5))
    similarity_threshold = th if th is not None else float(os.getenv("similarity_threshold", 0.8))

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 1 - similarity_threshold,
            "k": k_value
        }
    )
    print(f"Retriever initialized with k={k_value} and similarity threshold={similarity_threshold}")
    return retriever

