import streamlit as st
import requests
import json

# Set the title for the Streamlit app
st.title("SWLEOC MSK Triage Chatbot")
st.markdown("**AI-Powered Musculoskeletal Assessment with Questionnaire-Based Triage**")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm Leo, an AI assistant from SWLEOC. I'll help you with a structured musculoskeletal assessment using specialized questionnaires. To start, could you please describe your main musculoskeletal problem or concern?"}]

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input from the chat input box
if prompt := st.chat_input("Describe your musculoskeletal problem..."):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- This is the main logic block ---
    with st.chat_message("assistant"):
        # 1. Get the next question from the conversational agent
        with st.spinner("Analyzing your response..."):
            try:
                ask_url = "http://triage_app:8000/ask"
                payload = {"messages": st.session_state.messages, "model": "llama3.1:8b"}
                response = requests.post(ask_url, json=payload)
                response.raise_for_status()
                
                assistant_response = response.json().get("response", "Sorry, I encountered an error.")
                st.markdown(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the backend: {e}")
                assistant_response = "Error: Could not connect to the backend."
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        # 2. Check if the conversation is now complete
        if "clinical summary with differential diagnosis will be prepared" in assistant_response:
            # 3. If so, automatically call the summarization agent
            with st.spinner("Assessment complete. Generating SBAR clinical summary and differential diagnosis..."):
                try:
                    summarize_url = "http://triage_app:8000/summarize"
                    payload = {"messages": st.session_state.messages, "model": "llama3.1:8b"}
                    summary_response = requests.post(summarize_url, json=payload)
                    summary_response.raise_for_status()

                    summary_text = summary_response.json().get("response", "Could not generate summary.")
                    
                    # Display the final summary with better formatting
                    st.markdown("---") # Add a separator for clarity
                    st.markdown("## 📋 Clinical Assessment Summary")
                    st.markdown(summary_text)
                    st.session_state.messages.append({"role": "assistant", "content": summary_text})

                except requests.exceptions.RequestException as e:
                    st.error(f"Could not connect to the summarization service: {e}")

# Add sidebar with information about the system
with st.sidebar:
    st.markdown("## 🏥 SWLEOC MSK Triage System")
    st.markdown("""
    This AI-powered system uses structured questionnaires to assess musculoskeletal conditions and provide:
    
    - **SBAR Clinical Summaries** (Situation, Background, Assessment, Recommendation)
    - **Top 3 Differential Diagnoses** with confidence scores
    - **Evidence-based Triage Recommendations**
    - **Red Flag Screening** for urgent conditions
    
    ### Supported Assessments:
    - Knee Osteoarthritis
    - Knee Injuries
    - More assessments coming soon...
    
    ### Triage Pathways:
    - 🚨 **Urgent/A&E** - Fractures, dislocations, red flags
    - 🏥 **GP/Primary Care** - Red flags, unclear diagnosis
    - 🏃 **MSK Physiotherapy** - Mechanical pain, strains
    - 🔧 **Orthopaedic Surgery** - Failed conservative treatment
    """)
    
    if st.button("🔄 Start New Assessment"):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm Leo, an AI assistant from SWLEOC. I'll help you with a structured musculoskeletal assessment using specialized questionnaires. To start, could you please describe your main musculoskeletal problem or concern?"}]
        st.rerun()