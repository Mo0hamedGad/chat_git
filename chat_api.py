from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app2 import ask_bot, make_database

app = FastAPI()
chat_history = []

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        make_database()
        response = ask_bot(req.message, chat_history)
        return {"reply": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping")
async def ping():
    return {"status": "ok"}
