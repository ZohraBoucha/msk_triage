import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Define the data model for the request body
class PromptRequest(BaseModel):
    prompt: str
    model: str = "llama3.1:8b"  # Default model

app = FastAPI()

# URL for the Ollama server running in the other container
OLLAMA_API_URL = "http://llm_server:11434/api/generate"

@app.get("/")
def read_root():
    return {"message": "Hello, SWLEOC Triage Tool!"}

@app.post("/ask")
async def ask_llm(request: PromptRequest):
    """
    Receives a prompt, sends it to the Ollama server,
    and returns the model's response.
    """
    try:
        # Use httpx for async requests
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                OLLAMA_API_URL,
                json={
                    "model": request.model,
                    "prompt": request.prompt,
                    "stream": False  # We want the full response at once
                },
            )
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
            # The response from Ollama is JSON
            ollama_response = response.json()
            
            return {"response": ollama_response.get("response", "No response content.")}

    except httpx.RequestError as exc:
        # Handle network-related errors
        raise HTTPException(status_code=503, detail=f"Error connecting to Ollama: {exc}")
    except Exception as e:
        # Handle other potential errors
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
