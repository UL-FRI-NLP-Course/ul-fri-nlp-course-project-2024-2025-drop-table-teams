from langchain_community.vectorstores import FAISS

def init_retriever(vectorstore: FAISS):
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)
    print("Retriever initialized")
    return retriever
