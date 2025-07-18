from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uuid import uuid4
from typing import Dict, List
from app2 import ask_bot, make_database

app = FastAPI()

# Store sessions in-memory
sessions: Dict[str, List[dict]] = {}

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    new_session: bool = False  # Optional flag to force new session

class ChatResponse(BaseModel):
    reply: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        # Start a new session if requested or no session_id provided
        if req.new_session or not req.session_id:
            session_id = str(uuid4())
            chat_history = []
        else:
            session_id = req.session_id
            chat_history = sessions.get(session_id, [])

        make_database()  # Optional if already loaded at app startup

        response = ask_bot(req.message, chat_history)

        # Update chat history
        chat_history.append({"role": "user", "message": req.message})
        chat_history.append({"role": "bot", "message": response})
        sessions[session_id] = chat_history

        return ChatResponse(reply=response, session_id=session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "Session ended", "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/ping")
async def ping():
    return {"status": "ok"}
