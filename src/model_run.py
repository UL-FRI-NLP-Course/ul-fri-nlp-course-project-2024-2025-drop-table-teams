from src.chain_initialisation import init_chain
from src.model_initialisation import init_model
from src.pipeline_initialisation import init_pipeline


def run_model():
    model, model_name = init_model("YOUR_TOKEN")
    pipeline = init_pipeline(model, model_name)
    chain = init_chain(pipeline)
    return chain


if __name__ == '__main__':
    chain = run_model()
    question = "What is the scoring criteria of the NLP course?"
    print(chain.invoke({"question": question}))

