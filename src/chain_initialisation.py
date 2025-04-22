from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate


def init_chain(pipeline):
    llm = HuggingFacePipeline(pipeline=pipeline)
    template = """
    You are a helpful AI QA assistant. When answering questions, use the context enclosed by triple backquotes if 
    it is relevant.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Reply your answer in markdown format.
    Question: {question}
    Answer:"""
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    return chain
