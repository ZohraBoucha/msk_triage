from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# Import the existing agent
from .triage_agent import TriageAgent
# Import the NEW agent
from .summarization_agent import SummarizationAgent


# --- Data Models (No changes here) ---
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

# --- This is your existing endpoint (No changes here) ---
@app.post("/ask")
async def ask_llm(request: PromptRequest):
    """
    This endpoint now simply acts as a gateway to the TriageAgent.
    """
    agent = TriageAgent(model=request.model)
    message_dicts = [msg.dict() for msg in request.messages]
    
    try:
        response_text = await agent.get_next_response(message_dicts)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

# --- ADD THIS NEW ENDPOINT ---
@app.post("/summarize")
async def summarize_conversation(request: PromptRequest):
    """
    This endpoint takes the final conversation and generates the summary.
    """
    agent = SummarizationAgent(model=request.model)
    message_dicts = [msg.dict() for msg in request.messages]
    
    try:
        summary_text = await agent.summarize_and_triage(message_dicts)
        return {"response": summary_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")