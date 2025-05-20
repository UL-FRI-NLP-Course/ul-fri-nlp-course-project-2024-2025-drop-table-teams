# Natural language processing course: `'; DROP TABLE TEAMS; --`

Henri Sellis, Igor Sitek

**Project 1: Conversational Agent with Retrieval-Augmented Generation**

For our Natural Language Processing course group project, we are developing a conversational agent that retrieves additional information from Google Scholar documents, to increase the quality of answering questions. To accomplish this, first a number of most relevant tags are extracted from the user input, then corresponding queries are made to retrieve documents from Google Scholar, and finally the documents, along with the original user input, are fed to a Large Language Model, using prompt engineering.

[**Report**](/report/report.pdf)

## Local run

1. Copy the env file and fill the HuggingFace token.

    `cp .env-copy .env`
2. Install required packages (preferably in a new env).

    `conda env create my_env`

    `conda activate my_env`

    `pip install -r requirements.txt`
3. Download static dataset.

    `python -m src.dataset_scraping`
4. Run model.

    `python -m src.app`
