import uvicorn
import os
import json
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from src.chain_initialisation import init_chain
from src.embedding_initialisation import init_embeddings
from src.model_initialisation import init_model
from src.models import ChatName, Question
from src.pipeline_initialisation import init_pipeline
from src.retriever_initialisation import init_retriever

load_dotenv()
HISTORY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chat_history"))
os.makedirs(HISTORY_DIR, exist_ok=True)
cuda_enabled = os.getenv("CUDA_ENABLED", "Y").lower() in ['true', 'yes', 'y']

SESSION_FILE = os.path.join(HISTORY_DIR, "current_chat.txt")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    global model, model_name, pipeline, chain, vectorstore, retriever
    print("Initializing model...")
    token = os.getenv("hf_token")
    if not token:
        raise RuntimeError("hf_token not found in .env")
    model, model_name = init_model(token, cuda_enabled)
    pipeline = init_pipeline(model=model, model_name=model_name)
    vectorstore = init_embeddings(cuda_enabled)
    retriever = init_retriever(vectorstore)
    chain = init_chain(pipeline=pipeline, retriever=retriever)
    print("Start-up complete.")


@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    current_chat_id = f"chat-{uuid.uuid4().hex[:6]}"
    _set_current_chat_id(current_chat_id)
    current_chat_history = []

    return templates.TemplateResponse("chat.html",
                                      {"request": request, "history": current_chat_history, "chat_id": current_chat_id})


@app.get("/chat-{chat_id}", response_class=HTMLResponse)
async def get_chat(request: Request, chat_id: str):
    chat_history = _load_chat_history(chat_id)
    if not chat_history:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return templates.TemplateResponse("chat.html", {"request": request, "history": chat_history, "chat_id": chat_id})


@app.post("/ask")
async def ask_question(payload: Question):
    chat_id = _get_current_chat_id()
    history = _load_chat_history(chat_id)
    formatted_history = [(entry["question"], entry["answer"]) for entry in history]

    question = payload.question
    answer = chain.invoke(
        {"question": question,
         "chat_history": formatted_history}
    )["answer"].split("### Answer:")[-1].strip()

    history.append({"question": question, "answer": answer})
    _save_chat_history(chat_id, history)
    return {"history": history}


@app.get("/chats")
async def list_chats():
    try:
        files = os.listdir(HISTORY_DIR)
        chats = [os.path.splitext(f)[0] for f in files if f.endswith(".json")]
        return {"chats": chats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/change_chat")
async def change_chat(new_chat: ChatName):
    chat_name = new_chat.chat_name
    if not chat_name:
        raise HTTPException(status_code=422, detail="Chat name cannot be empty.")

    _set_current_chat_id(chat_name)
    return {"new_chat_name": chat_name, "history": _load_chat_history(chat_name)}


@app.post("/change_chat_name")
async def change_chat_name(new_chat: ChatName):
    chat_name = new_chat.chat_name.strip()
    current_chat_id = _get_current_chat_id()
    if not chat_name:
        raise HTTPException(status_code=422, detail="Chat name cannot be empty.")

    old_file_path = os.path.join(HISTORY_DIR, f"{current_chat_id}.json")
    new_file_path = os.path.join(HISTORY_DIR, f"{chat_name}.json")

    try:
        if os.path.exists(old_file_path):
            os.rename(old_file_path, new_file_path)
        else:
            raise HTTPException(status_code=404, detail="Chat history file not found.")
        _set_current_chat_id(chat_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rename chat: {str(e)}")
    return {"new_chat_name": chat_name, "history": _load_chat_history(chat_name)}


def _get_current_chat_id():
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r") as f:
            return f.read().strip()
    new_chat_id = f"chat-{uuid.uuid4().hex[:6]}"
    with open(SESSION_FILE, "w") as f:
        f.write(new_chat_id)
    return new_chat_id


def _set_current_chat_id(chat_id):
    with open(SESSION_FILE, "w") as f:
        f.write(chat_id)


def _load_chat_history(chat_id):
    try:
        with open(os.path.join(HISTORY_DIR, f"{chat_id}.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def _save_chat_history(chat_id, history):
    with open(os.path.join(HISTORY_DIR, f"{chat_id}.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=False)
