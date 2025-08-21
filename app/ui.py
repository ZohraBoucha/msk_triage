import streamlit as st
import requests
import json

# Set the title for the Streamlit app
st.title("SWLEOC MSK Triage Chatbot")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    # Start with the initial assistant message
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Leo, an AI assistant from SWLEOC. To start, could you please describe your main symptom?"}
    ]

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input from the chat input box
if prompt := st.chat_input("What are your symptoms?"):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare and display the assistant's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Call your FastAPI backend
        try:
            # The URL for the FastAPI service inside Docker
            fastapi_url = "http://triage_app:8000/ask"
            
            # Send the entire conversation history
            payload = {
                "messages": st.session_state.messages,
                "model": "llama3.1:8b" 
            }
            
            response = requests.post(fastapi_url, json=payload)
            response.raise_for_status() # Raise an exception for bad status codes
            
            assistant_response = response.json().get("response", "Sorry, I encountered an error.")
            message_placeholder.markdown(assistant_response)
            
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the backend: {e}")
            assistant_response = "Error: Could not connect to the backend."

        # Add assistant's full response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

