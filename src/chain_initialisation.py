from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate


def init_chain(pipeline, retriever):
    llm = HuggingFacePipeline(pipeline=pipeline)
    template = """
    You are a helpful AI QA assistant. When answering questions, use the context enclosed by triple backquotes if 
    it is relevant.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Reply your answer in html format.
    ``
    {context}
    ```
    
    ### Question:
    {question}
    
    ### Answer:
    """
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=template.strip(),
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        verbose=False,
    )
    print("LLM model chain created.")
    return chain
