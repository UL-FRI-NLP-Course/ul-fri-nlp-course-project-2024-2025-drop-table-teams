from scholarly import scholarly
import requests
import os

def download_documents(query: str, data_dir: str, num_docs: int, max_tries: int):

    search = scholarly.search_pubs(query)
    downloads = []
    for i in range(max_tries):
        result = next(search)
        title = result["bib"]["title"]
        print(f"{i}: {title}")
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

    return downloads
