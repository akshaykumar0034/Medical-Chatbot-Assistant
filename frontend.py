import streamlit as st
import requests
from datetime import datetime
import time

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Page Configuration
st.set_page_config(
    page_title="AI Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visibility and Claude-like UI
st.markdown("""
<style>
    /* Dark theme for main area */
    .main {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #2d2d30;
        padding: 1rem 0.5rem;
    }
    
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    /* Chat message containers */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        line-height: 1.6;
        font-size: 1rem;
    }
    
    .user-message {
        background-color: #2d2d30;
        border-left: 4px solid #0078d4;
        color: #ffffff;
    }
    
    .assistant-message {
        background-color: #1a1a1c;
        border-left: 4px solid #10a37f;
        color: #e0e0e0;
    }
    
    .message-role {
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .user-role {
        color: #0078d4;
    }
    
    .assistant-role {
        color: #10a37f;
    }
    
    .message-content {
        color: #e0e0e0;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    /* Header */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
        padding: 1rem 0;
    }
    
    /* Session item styling */
    .stButton button {
        text-align: left;
        font-size: 0.9rem;
        background-color: #383838 !important;
        color: #ffffff !important;
        border: 1px solid #404040;
    }
    
    .stButton button:hover {
        background-color: #454545 !important;
        border-color: #0078d4;
    }
    
    /* Warning box */
    .warning-box {
        background-color: #3d3a2e;
        color: #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        font-size: 0.85rem;
    }
    
    .warning-box strong {
        color: #ffc107 !important;
    }
    
    /* Info box */
    .info-box {
        background-color: #2d3e50;
        color: #5bc0de;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    /* Chat input area */
    .stChatInput {
        border-radius: 0.5rem;
        background-color: #2d2d30;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e1e1e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #404040;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        color: #888;
    }
    
    .empty-state h2 {
        color: #ffffff;
        margin-bottom: 1rem;
    }
    
    /* Quick start buttons */
    .quick-start-btn {
        background-color: #2d2d30;
        color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #404040;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .quick-start-btn:hover {
        background-color: #383838;
        border-color: #0078d4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# API Functions
def create_new_session(session_name="New Chat"):
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/sessions/create",
            json={"session_name": session_name},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error creating session: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Make sure the FastAPI server is running on port 8000.")
    except Exception as e:
        st.error(f"Error creating session: {e}")
    return None

def get_all_sessions():
    try:
        response = requests.get(f"{API_BASE_URL}/api/sessions", timeout=10)
        if response.status_code == 200:
            return response.json()["sessions"]
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Please start the backend server.")
    except Exception as e:
        st.error(f"Error fetching sessions: {e}")
    return []

def send_message(session_id, message):
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/chat",
            json={"session_id": session_id, "message": message},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Error sending message: {e}")
    return None

def get_session_history(session_id):
    try:
        response = requests.get(f"{API_BASE_URL}/api/sessions/{session_id}/history", timeout=10)
        if response.status_code == 200:
            return response.json()["history"]
    except Exception as e:
        st.error(f"Error fetching history: {e}")
    return []

def delete_session(session_id):
    try:
        response = requests.delete(f"{API_BASE_URL}/api/sessions/{session_id}", timeout=10)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error deleting session: {e}")
    return False

# Sidebar - Chat History
with st.sidebar:
    st.markdown("### 🏥 Medical Assistant")
    st.markdown("---")
    
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        timestamp = datetime.now().strftime("%I:%M %p")
        new_session = create_new_session(f"Chat {timestamp}")
        if new_session:
            st.session_state.current_session_id = new_session["session_id"]
            st.session_state.messages = []
            st.session_state.input_key += 1
            st.rerun()
    
    st.markdown("### 📝 Chat History")
    
    # Load sessions
    st.session_state.sessions = get_all_sessions()
    
    # Display sessions
    if st.session_state.sessions:
        for session in reversed(st.session_state.sessions):
            session_id = session["session_id"]
            session_name = session["session_name"]
            
            col1, col2 = st.columns([4, 1])
            
            with col1:
                button_type = "secondary" if st.session_state.current_session_id == session_id else "tertiary"
                if st.button(
                    f"💬 {session_name}",
                    key=f"session_{session_id}",
                    use_container_width=True,
                    type=button_type
                ):
                    st.session_state.current_session_id = session_id
                    st.session_state.messages = get_session_history(session_id)
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"delete_{session_id}", help="Delete chat"):
                    if delete_session(session_id):
                        if st.session_state.current_session_id == session_id:
                            st.session_state.current_session_id = None
                            st.session_state.messages = []
                        st.rerun()
    else:
        st.info("No chats yet. Create one to start!")
    
    st.markdown("---")
    
    # Example prompts
    with st.expander("💡 Example Prompts"):
        st.markdown("""
        **Emergency:**
        - I have severe chest pain!
        - Difficulty breathing
        
        **Symptoms:**
        - I have a fever of 101°F
        - Persistent headache
        
        **Questions:**
        - What is a fever?
        - When to see a doctor?
        """)
    
    # Warning
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Important:</strong><br/>
        This is NOT a substitute for professional medical advice. 
        Always consult a healthcare provider.
    </div>
    """, unsafe_allow_html=True)

# Main Chat Area
st.markdown('<div class="main-header">🏥 AI Medical Assistant</div>', unsafe_allow_html=True)

# Check if session exists
if not st.session_state.current_session_id:
    st.markdown("""
    <div class="empty-state">
        <h2>Welcome to AI Medical Assistant</h2>
        <p>Create a new chat or select an existing one to begin</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick start section
    st.markdown("### Quick Start")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🆘 Emergency Help", use_container_width=True, key="qs1"):
            new_session = create_new_session("Emergency")
            if new_session:
                st.session_state.current_session_id = new_session["session_id"]
                st.session_state.messages = []
                st.rerun()
    
    with col2:
        if st.button("🤒 Symptom Check", use_container_width=True, key="qs2"):
            new_session = create_new_session("Symptom Analysis")
            if new_session:
                st.session_state.current_session_id = new_session["session_id"]
                st.session_state.messages = []
                st.rerun()
    
    with col3:
        if st.button("❓ Ask Question", use_container_width=True, key="qs3"):
            new_session = create_new_session("General Question")
            if new_session:
                st.session_state.current_session_id = new_session["session_id"]
                st.session_state.messages = []
                st.rerun()

else:
    # Display chat messages
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-role user-role">👤 You</div>
                <div class="message-content">{content}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div class="message-role assistant-role">🏥 Medical Assistant</div>
                <div class="message-content">{content}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input(
        "Describe your symptoms or ask a medical question...",
        key=f"chat_input_{st.session_state.input_key}"
    )
    
    if user_input:
        # Add user message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-role user-role">👤 You</div>
            <div class="message-content">{user_input}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show loading
        with st.spinner("🤔 Analyzing your message..."):
            response = send_message(st.session_state.current_session_id, user_input)
        
        if response:
            # Add assistant message
            assistant_message = response["response"]
            st.session_state.messages.append({"role": "assistant", "content": assistant_message})
            
            # Rerun to display
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.875rem; padding: 1rem;'>
    Made with ❤️ using Streamlit | Powered by Google Gemini & LangGraph
</div>
""", unsafe_allow_html=True)