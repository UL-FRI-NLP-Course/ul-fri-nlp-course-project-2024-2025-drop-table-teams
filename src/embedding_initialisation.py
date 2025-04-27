import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.dataset_loading import load_dataset_to_splits


def init_embeddings(cuda_available: bool):
    embed_model = os.getenv("embedding_model")

    faiss_index_path = "faiss_index"
    index_file = os.path.join(faiss_index_path, "index.faiss")
    store_file = os.path.join(faiss_index_path, "index.pkl")

    embedding = HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": "cuda" if cuda_available else "cpu"},
        encode_kwargs={"batch_size": 64 if cuda_available else 16}
    )

    if os.path.exists(index_file) and os.path.exists(store_file):
        print("Loading FAISS from file - skipping embedding build.")
        vectorstore = FAISS.load_local(faiss_index_path, embedding, allow_dangerous_deserialization=True)
        print("Vectorstore loaded successfully.")
        return vectorstore
    else:
        print("FAISS index not found. Building...")
        splits = load_dataset_to_splits()
        vectorstore = FAISS.from_documents(splits, embedding)
        vectorstore.save_local(faiss_index_path)
        print("Vectorstore loaded successfully.")
        return vectorstore
