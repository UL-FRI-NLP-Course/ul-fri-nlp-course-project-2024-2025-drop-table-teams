import os

from langchain_community.vectorstores import FAISS

def init_retriever(vectorstore: FAISS):
    k_value = int(os.getenv("vectorstore_k"))
    retriever = vectorstore.as_retriever(search_type="similarity", k=k_value)
    print("Retriever initialized")
    return retriever
