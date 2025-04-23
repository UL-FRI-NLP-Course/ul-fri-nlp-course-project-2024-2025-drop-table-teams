import uvicorn
import os
import json

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


# from src.chain_initialisation import init_chain
# from src.model_initialisation import init_model
# from src.pipeline_initialisation import init_pipeline

def init_model(token):
    return "my_model", "my_model_name"


def init_pipeline(model, model_name):
    return lambda x: {"answer": f"Answer to: {x['question']}"}


def init_chain(pipeline):
    return pipeline


model, model_name = init_model("YOUR_TOKEN")
pipeline = init_pipeline(model, model_name)
chain = init_chain(pipeline)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

HISTORY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chat_history"))
os.makedirs(HISTORY_DIR, exist_ok=True)

HISTORY_FILE = os.path.join(HISTORY_DIR, "chat_history.json")

if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        conversation_history = json.load(f)
else:
    conversation_history = []



@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "history": conversation_history})


class Question(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(payload: Question):
    question = payload.question
    answer = chain({"question": question})["answer"]
    conversation_history.append({"question": question, "answer": answer})

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=2)

    return {"history": conversation_history}


if __name__ == '__main__':
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
