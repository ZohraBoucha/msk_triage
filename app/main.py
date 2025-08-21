from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# Import the agent that contains all our logic
from .triage_agent import TriageAgent

# --- Data Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class PromptRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "llama3.1:8b"

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, SWLEOC Triage Tool!"}

@app.post("/ask")
async def ask_llm(request: PromptRequest):
    """
    This endpoint now simply acts as a gateway to the TriageAgent.
    """
    # Create an instance of our agent
    agent = TriageAgent(model=request.model)
    
    # Convert Pydantic models to simple dicts for the agent
    message_dicts = [msg.dict() for msg in request.messages]
    
    try:
        # Get the next response from the agent
        response_text = await agent.get_next_response(message_dicts)
        return {"response": response_text}
    except Exception as e:
        # The agent handles its own internal errors, but we catch any others
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

