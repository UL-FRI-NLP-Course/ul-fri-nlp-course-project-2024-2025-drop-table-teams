from pydantic import BaseModel


class Question(BaseModel):
    question: str


class ChatName(BaseModel):
    chat_name: str
