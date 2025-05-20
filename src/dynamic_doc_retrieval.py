import os
import re

import requests
from keybert import KeyBERT
from scholarly import scholarly


def download_documents(query: str, data_dir: str, num_docs: int, max_tries: int):
    print(f"Querying Google Scholar for: {query}")
    search = scholarly.search_pubs(query)
    downloads = []

    for i in range(max_tries):
        try:
            result = next(search)
        except StopIteration:
            print("No more results available.")
            break

        if len(downloads) >= num_docs:
            break

        title = result["bib"].get("title", f"untitled_{i}")
        pub_url = result.get("pub_url", "No publication URL available")
        print(f"\nResult {i + 1}: {title}")
        print(f"Publication URL: {pub_url}")

        if not _download_from_scholarly(data_dir, downloads, result, title):
            _download_from_scihub(data_dir, downloads, pub_url, title)

    return downloads


def _download_from_scholarly(data_dir, downloads, result, title):
    if "eprint_url" in result:
        eprint_url = result["eprint_url"]
        print(f"Trying to download PDF from: {eprint_url}")
        try:
            response = requests.get(eprint_url, stream=True, verify=False)
            content_type = response.headers.get("Content-Type", "")
            is_pdf = "application/pdf" in content_type or eprint_url.lower().endswith(".pdf")

            if response.status_code == 200 and is_pdf:
                os.makedirs(data_dir, exist_ok=True)
                safe_title = safe_filename(title)
                filename = os.path.join(data_dir, f"{safe_title}.pdf")

                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"PDF saved as {filename}")
                downloads.append(filename)
                return True
            else:
                print("The eprint URL is not a direct link to a PDF.")
        except Exception as e:
            print(f"Failed to download PDF: {e}")

    return False


def _download_from_scihub(data_dir, downloads, url, title):
    print(f"Downloading {title} from Sci Hub: {url}")
    safe_title = safe_filename(title)
    filename = os.path.join(data_dir, f"{safe_title}.pdf")

    from scidownl import scihub_download

    paper_type = "doi"
    proxies = {
        'http': 'socks5://127.0.0.1:7890'
    }
    scihub_download(url, paper_type=paper_type, out=filename, proxies=proxies)

    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"Successfully downloaded to {filename}")
        downloads.append(filename)
    else:
        print(f"Failed to download {title} via Sci-Hub.")


def safe_filename(title: str):
    # Replace spaces and slashes with underscores
    title = title.replace(" ", "_").replace("/", "_")
    # Remove all characters that are not alphanumeric, underscore, dash, or dot
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '', title)


def initialise_keyword_model():
    return KeyBERT()


def generate_query_from_question(keyword_model, question):
    keywords = keyword_model.extract_keywords(question, keyphrase_ngram_range=(1, 1))
    return " ".join([keyword[0] for keyword in keywords])

