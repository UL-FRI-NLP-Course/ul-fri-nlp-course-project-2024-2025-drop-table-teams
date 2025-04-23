def init_retriever(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", k=10)
    print("Retriever initialized")
    return retriever
