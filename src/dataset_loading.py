import os
import random

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_dataset_to_splits() -> list[Document]:
    document_limit = os.getenv("vectorstore_document_limit")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    corpus = _find_txt_files_contents(data_dir)
    try:
        document_limit_int = int(document_limit)
        print(f"Using only {document_limit_int} documents.")
        corpus = random.sample(corpus, document_limit_int)
    except ValueError:
        print("Using all available documents.")
    split_documents = text_splitter.split_documents(corpus)
    print(f"Found {len(split_documents)} splits to populate.")
    return split_documents


def _find_txt_files_contents(directory, exclude_names=("README", "language")):
    exclude_names = [name.lower() for name in exclude_names]
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                base_name = os.path.splitext(file)[0]
                if base_name.lower() in exclude_names:
                    continue
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        doc = Document(page_content=content, metadata={"source": file_path})
                        documents.append(doc)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    print(f"Found {len(documents)} documents in directory {directory}")
    return documents
