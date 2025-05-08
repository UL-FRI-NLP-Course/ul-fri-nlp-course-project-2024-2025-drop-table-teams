import requests
import os

from scholarly import scholarly
from keybert import KeyBERT


def download_documents(query: str, data_dir: str, num_docs: int, max_tries: int):
    print(f"Querying Google Scholar for: {query}")
    search = scholarly.search_pubs(query)
    downloads = []
    for i in range(max_tries):
        result = next(search)
        title = result["bib"]["title"]
        print(f"Result {i+1}: {title}")
        if "eprint_url" in result:
            response = requests.get(result["eprint_url"], stream=True, verify=False)

            content_type = response.headers.get("Content-Type", "")
            is_pdf = "application/pdf" in content_type or result["eprint_url"].lower().endswith(".pdf")

            if response.status_code == 200 and is_pdf:
                title = title.replace(" ", "_").replace("/", "_")
                filename = os.path.join(data_dir, f"{title}.pdf")
                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"PDF saved as {filename}")
                downloads.append(filename)
                if len(downloads) >= num_docs:
                    break
            else:
                print("The eprint URL is not a direct link to a PDF.")
        else:
            print("No direct link for downloading this document.")
    return downloads

def initialise_keyword_model():
    return KeyBERT()

def generate_query_from_question(keyword_model, question):
    keywords = keyword_model.extract_keywords(question, keyphrase_ngram_range=(1, 1))
    return " ".join([keyword[0] for keyword in keywords])
