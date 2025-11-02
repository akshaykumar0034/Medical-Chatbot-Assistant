"""
Medical Assistant Backend API
FastAPI server that integrates with Streamlit frontend
Run with: uvicorn backend:app --reload --host 0.0.0.0 --port 8000
"""

import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Literal, Dict, Optional
import operator
import faiss # Make sure you have 'pip install faiss-cpu'

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticBaseModel, Field

# LangChain Core Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel as LangChainBaseModel

# Model & Embeddings Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

# LangGraph Imports
from langgraph.graph import StateGraph, END
from typing_extensions import Annotated, TypedDict

# ============================================================================
# CONFIGURATION
# ============================================================================

print("=" * 60)
print("🏥 Medical Assistant Backend API")
print("=" * 60)

# Load environment variables
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    raise EnvironmentError("❌ GOOGLE_API_KEY not found in .env file. Please add it.")

print("✅ Environment variables loaded")

# ============================================================================
# SYSTEM PROMPT & GUARDRAILS
# ============================================================================

SYSTEM_PROMPT = """
You are an AI medical assistant. You are **NOT** a doctor or a substitute for one.
Your purpose is to provide medical information and preliminary triage, **NOT** to diagnose, prescribe, or give medical advice.

**Your Core Rules:**
1.  **Safety First:** If a user describes symptoms of a potential medical emergency (e.g., "chest pain," "difficulty breathing," "severe bleeding," "suicidal thoughts"), you MUST immediately stop all other analysis and output the emergency response.
2.  **No Diagnosis:** You must NEVER provide a diagnosis. You can state what symptoms are "consistent with" but must always follow up with a disclaimer.
3.  **Always Disclaimer:** Every response that provides any medical information or triage MUST end with the disclaimer: "I am an AI assistant, not a medical professional. Please consult a qualified healthcare provider for any medical advice or diagnosis."
4.  **Use History:** Use the chat history to understand the context of the user's follow-up questions.
"""

# Pydantic Models for Structured Output (LLM Guardrails)
class EmergencyCheck(LangChainBaseModel):
    """Detects medical emergencies"""
    is_emergency: bool = Field(..., description="True if a medical emergency is detected.")
    reasoning: str = Field(..., description="Brief explanation for the classification.")

class IntentClassifier(LangChainBaseModel):
    """Classifies user intent"""
    intent: Literal["symptom_analysis", "general_question"] = Field(
        ..., description="The user's primary intent."
    )

class SymptomTriage(LangChainBaseModel):
    """Provides symptom triage"""
    assessment: str = Field(..., description="Summary of what symptoms might be consistent with.")
    triage_level: Literal["Self-Care", "See a Doctor", "Urgent Care"] = Field(
        ..., description="Recommended level of care."
    )
    recommendation: str = Field(..., description="Suggested next steps.")
    follow_up_question: str = Field(..., description="Follow-up question or 'None'.")
    disclaimer: str = Field(..., description="Mandatory medical disclaimer.")

class GeneralResponse(LangChainBaseModel):
    """Provides general medical information"""
    response: str = Field(..., description="Answer based on provided context.")
    disclaimer: str = Field(..., description="Mandatory medical disclaimer.")

# ============================================================================
# RAG SETUP (Knowledge Base)
# ============================================================================

# --- MODIFICATION 1: This whole function is changed ---
def setup_retriever():
    """Initialize the RAG retriever with medical knowledge"""
    print("📚 Setting up RAG retriever...")
    
    DEMO_MEDICAL_TEXT = """
    A fever is a temporary increase in your body temperature, often due to an illness.
    For an adult, a fever may be uncomfortable, but it usually isn't a cause for concern unless it reaches 103 F (39.4 C) or higher.
    
    Common cold symptoms include runny nose, sore throat, cough, congestion, slight body aches, mild headache, sneezing, and low-grade fever.
    
    Headaches can be caused by tension, dehydration, stress, lack of sleep, or underlying medical conditions.
    """
    
    docs = [Document(page_content=DEMO_MEDICAL_TEXT, metadata={"source": "medical_knowledge_base"})]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    print("🔄 Loading HuggingFace embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("💾 Creating vector store...")
    # We are now using COSINE SIMILARITY (IP, inner product, on normalized vectors)
    # This means scores will be from -1 to 1 (or 0 to 1). HIGHER is better.
    vectorstore = FAISS.from_documents(
        splits, 
        embeddings,
        distance_strategy=faiss.METRIC_INNER_PRODUCT # Use Inner Product (Cosine)
    )
    
    print("✅ RAG retriever ready")
    
    # We return the whole vectorstore to use its similarity_search_with_score method
    return vectorstore

# ============================================================================
# LANGGRAPH STATE
# ============================================================================

class GraphState(TypedDict):
    """State passed through the LangGraph workflow"""
    chat_history: Annotated[list, operator.add]
    current_input: str
    triage_level: str

# ============================================================================
# MEDICAL ASSISTANT GRAPH
# ============================================================================

class MedicalGraph:
    """Main LangGraph workflow for medical assistant"""
    
    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
        self.setup_chains()
        print("✅ Medical Graph initialized")

    def setup_chains(self):
        """Setup LLM chains with structured output"""
        
        # Emergency detection chain
        self.emergency_check_chain = (
            ChatPromptTemplate.from_messages([
                ("system", f"{SYSTEM_PROMPT}\n\nYou must strictly use the EmergencyCheck tool. Classify if this is a medical emergency."),
                ("human", "User input: {input}")
            ]) | self.llm.with_structured_output(EmergencyCheck)
        )
        
        # Intent classification chain
        self.intent_classifier_chain = (
            ChatPromptTemplate.from_messages([
                ("system", f"{SYSTEM_PROMPT}\n\nYou must strictly use the IntentClassifier tool. Classify the user's intent."),
                ("user", "Chat History:\n{history}\n\nUser input: {input}")
            ]) | self.llm.with_structured_output(IntentClassifier)
        )

        # Symptom triage chain
        self.triage_chain = (
            ChatPromptTemplate.from_messages([
                ("system", f"{SYSTEM_PROMPT}\n\nYou must strictly use the SymptomTriage tool. Analyze the user's symptoms."),
                ("user", "Chat History:\n{history}\n\nUser input: {input}")
            ]) | self.llm.with_structured_output(SymptomTriage)
        )

        # RAG chain (Strict)
        self.rag_chain = (
            ChatPromptTemplate.from_messages([
                ("system", f"{SYSTEM_PROMPT}\n\nYou must strictly use the GeneralResponse tool. Answer ONLY using the context provided. Do not use any other knowledge."),
                ("user", "Context from knowledge base:\n{context}\n\nChat History:\n{history}\n\nUser input: {input}")
            ]) | self.llm.with_structured_output(GeneralResponse)
        )

        # Fallback chain (General Knowledge)
        self.general_fallback_chain = (
            ChatPromptTemplate.from_messages([
                ("system", f"{SYSTEM_PROMPT}\n\n**Special Instruction:** The user's question was not found in the dedicated medical knowledge base. You must answer the user's general question using your own general knowledge.\n\nYou must **still** follow all core safety rules (no diagnosis, no prescription) and **must** provide the mandatory medical disclaimer. Do NOT apologize or mention the knowledge base."),
                ("user", "Chat History:\n{history}\n\nUser input: {input}")
            ]) | self.llm.with_structured_output(GeneralResponse)
        )


    # NODES
    
    def node_detect_emergency(self, state: GraphState) -> dict:
        """Node 1: Detect medical emergencies"""
        print("   🔍 Node: Detecting emergency...")
        user_input = state["current_input"]
        response = self.emergency_check_chain.invoke({"input": user_input})
        
        if response.is_emergency:
            print(f"   ⚠️  EMERGENCY DETECTED: {response.reasoning}")
            ai_msg = AIMessage(
                content="🚨 **MEDICAL EMERGENCY DETECTED**\n\n"
                        "Based on your description, this could be a medical emergency. "
                        "**Please contact your local emergency services immediately** (911 in the US) "
                        "or go to the nearest emergency room.\n\n"
                        "Do not wait for an online response. Seek immediate medical attention."
            )
            return {
                "chat_history": [HumanMessage(content=user_input), ai_msg],
                "triage_level": "Emergency"
            }
        else:
            print(f"   ✅ Not an emergency: {response.reasoning}")
            return {"triage_level": "Non-Emergency"}

    def node_run_symptom_triage(self, state: GraphState) -> dict:
        """Node 2a: Analyze symptoms and provide triage"""
        print("   🩺 Node: Running symptom triage...")
        user_input = state["current_input"]
        history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in state['chat_history']])
        response = self.triage_chain.invoke({"history": history_str, "input": "input"})
        
        # Format response with markdown
        ai_response_text = (
            f"**Assessment:** {response.assessment}\n\n"
            f"**Recommended Action:** {response.triage_level}\n\n"
            f"**Recommendation:**\n{response.recommendation}\n\n"
            f"*{response.disclaimer}*"
        )
        if response.follow_up_question and response.follow_up_question.lower() != "none":
            ai_response_text += f"\n\n**Follow-up Question:** {response.follow_up_question}"

        print(f"   ✅ Triage complete: {response.triage_level}")
        return {
            "chat_history": [HumanMessage(content=user_input), AIMessage(content=ai_response_text)]
        }

    # --- MODIFICATION 2: This whole function is changed ---
    def node_run_rag(self, state: GraphState) -> dict:
        """Node 2b: Answer general questions using RAG, with a fallback"""
        print("   📖 Node: Running RAG...")
        user_input = state["current_input"]
        history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in state['chat_history']])
        
        # NOTE: We are using COSINE SIMILARITY. HIGHER is better.
        # A score > 0.7 is a good match. A score < 0.5 is a bad match.
        RELEVANCE_THRESHOLD = 0.5 # Tune this as needed
        
        # Retrieve relevant documents WITH their scores
        # We now use 'similarity_search_with_score'
        retrieved_docs_with_scores = self.vectorstore.similarity_search_with_score(
            user_input, k=2
        )
        
        # Filter out documents that don't meet the relevance threshold
        # We check if score > RELEVANCE_THRESHOLD
        relevant_docs = [
            doc for doc, score in retrieved_docs_with_scores if score > RELEVANCE_THRESHOLD
        ]
        
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        if relevant_docs:
            # --- 1. RAG SUCCESS: Relevant context was found ---
            print(f"   📚 Found {len(relevant_docs)} relevant documents. Answering with RAG.")
            response = self.rag_chain.invoke({
                "context": context,
                "history": history_str,
                "input": user_input
            })
        else:
            # --- 2. RAG FAILED: No relevant context found, use fallback ---
            print(f"   ⚠️  No relevant documents found in RAG (score < {RELEVANCE_THRESHOLD}). Using general knowledge fallback.")
            response = self.general_fallback_chain.invoke({
                "history": history_str,
                "input": user_input
            })

        # Format the response
        ai_response_text = f"{response.response}\n\n*{response.disclaimer}*"
        
        print("   ✅ RAG/Fallback response generated")
        return {
            "chat_history": [HumanMessage(content=user_input), AIMessage(content=ai_response_text)]
        }
    # --- END OF MODIFICATION 2 ---


    # EDGES (Routing Logic)
    
    def edge_route_after_emergency_check(self, state: GraphState) -> Literal["handle_emergency", "classify_intent"]:
        """Route based on emergency detection"""
        if state["triage_level"] == "Emergency":
            return "handle_emergency"
        return "classify_intent"

    def edge_route_after_intent_classification(self, state: GraphState) -> Literal["run_symptom_triage", "run_rag"]:
        """Route based on user intent"""
        user_input = state["current_input"]
        history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in state['chat_history']])
        
        response = self.intent_classifier_chain.invoke({"history": history_str, "input": user_input})
        
        print(f"   🎯 Intent: {response.intent}")
        return "run_symptom_triage" if response.intent == "symptom_analysis" else "run_rag"

    # BUILD GRAPH
    
    def build_graph(self) -> StateGraph:
        """Assemble the complete workflow"""
        print("🔧 Building LangGraph workflow...")
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("detect_emergency", self.node_detect_emergency)
        workflow.add_node("run_symptom_triage", self.node_run_symptom_triage)
        workflow.add_node("run_rag", self.node_run_rag)
        workflow.add_node("classify_intent_edge", lambda state: {})  # Pass-through node

        # Set entry point
        workflow.set_entry_point("detect_emergency")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "detect_emergency",
            self.edge_route_after_emergency_check,
            {
                "handle_emergency": END,
                "classify_intent": "classify_intent_edge"
            }
        )
        
        workflow.add_conditional_edges(
            "classify_intent_edge",
            self.edge_route_after_intent_classification,
            {
                "run_symptom_triage": "run_symptom_triage",
                "run_rag": "run_rag"
            }
        )

        workflow.add_edge("run_symptom_triage", END)
        workflow.add_edge("run_rag", END)
        
        print("✅ Graph compiled successfully")
        return workflow.compile()

# ============================================================================
# INITIALIZE COMPONENTS
# ============================================================================

print("\n🚀 Initializing components...")

# Setup RAG retriever
vectorstore = setup_retriever()

# Initialize LLM
print("🤖 Initializing Google Gemini...")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.0,
    convert_system_message_to_human=True
)
print("✅ LLM initialized")

# Build the graph
graph_builder = MedicalGraph(llm, vectorstore)
app_graph = graph_builder.build_graph()

# In-memory session storage
sessions: Dict[str, Dict] = {}
print("✅ Session storage initialized")

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Medical Assistant API",
    description="Backend API for AI Medical Assistant with RAG and LangGraph",
    version="1.0.0"
)

# CORS middleware for Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API MODELS (Request/Response)
# ============================================================================

class ChatRequest(PydanticBaseModel):
    """Request model for chat endpoint"""
    session_id: str
    message: str

class ChatResponse(PydanticBaseModel):
    """Response model for chat endpoint"""
    session_id: str
    response: str
    timestamp: str

class SessionCreate(PydanticBaseModel):
    """Request model for creating a session"""
    session_name: Optional[str] = "New Chat"

class SessionResponse(PydanticBaseModel):
    """Response model for session info"""
    session_id: str
    session_name: str
    created_at: str

class SessionList(PydanticBaseModel):
    """Response model for listing sessions"""
    sessions: List[SessionResponse]

class MessageHistory(PydanticBaseModel):
    """Response model for chat history"""
    history: List[dict]

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Medical Assistant API is running",
        "docs": "/docs",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "sessions": len(sessions),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/sessions/create", response_model=SessionResponse)
def create_session(req: SessionCreate):
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    sessions[session_id] = {
        "session_name": req.session_name,
        "created_at": timestamp,
        "state": {
            "chat_history": [],
            "current_input": "",
            "triage_level": ""
        }
    }
    
    print(f"✅ Session created: {session_id}")
    
    return SessionResponse(
        session_id=session_id,
        session_name=req.session_name,
        created_at=timestamp
    )

@app.get("/api/sessions", response_model=SessionList)
def get_sessions():
    """Get all chat sessions"""
    session_list = [
        SessionResponse(
            session_id=sid,
            session_name=data["session_name"],
            created_at=data["created_at"]
        )
        for sid, data in sessions.items()
    ]
    return SessionList(sessions=session_list)

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Send a message and get AI response"""
    print(f"\n💬 Incoming message for session: {req.session_id}")
    print(f"   Message: {req.message}")
    
    # Validate session
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get current state
    current_state = sessions[req.session_id]["state"]
    current_state["current_input"] = req.message
    
    try:
        # Invoke the graph
        print("🔄 Processing through LangGraph...")
        result = app_graph.invoke(current_state)
        
        # Update session state
        sessions[req.session_id]["state"] = result
        
        # Get AI response
        ai_response = result['chat_history'][-1].content
        
        print(f"✅ Response generated")
        
        return ChatResponse(
            session_id=req.session_id,
            response=ai_response,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/api/sessions/{session_id}/history", response_model=MessageHistory)
def get_history(session_id: str):
    """Get chat history for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    history = sessions[session_id]["state"]["chat_history"]
    
    return MessageHistory(
        history=[
            {
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content
            }
            for msg in history
        ]
    )

@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    """Delete a chat session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    print(f"🗑️  Session deleted: {session_id}")
    
    return {"message": "Session deleted successfully"}

@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    """Get session details"""

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    return {
        "session_id": session_id,
        "session_name": session_data["session_name"],
        "created_at": session_data["created_at"],
        "message_count": len(session_data["state"]["chat_history"])
    }

# ============================================================================
# STARTUP & MAIN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("\n" + "=" * 60)
    print("🚀 Medical Assistant API Started")
    print("=" * 60)
    print(f"📍 Server: http://localhost:8000")
    print(f"📖 API Docs: http://localhost:8000/docs")
    print(f"🔧 Health: http://localhost:8000/health")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    import uvicorn
    
    print("\n🌟 Starting FastAPI server...")
    print("💡 Tip: Access API docs at http://localhost:8000/docs")
    print("💡 Tip: Use Ctrl+C to stop the server\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )