# app/streamlit_app.py

import streamlit as st
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.pipeline import IngestionPipeline
from config.settings import settings


st.set_page_config(
    page_title="Project Aegis - Enterprise RAG",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Project Aegis")
st.markdown("**Advanced Corporate Policy RAG System**")

# ====================== SIDEBAR ======================
with st.expander("⚙️ Ingestion Controls", expanded=True):
    #st.header("⚙️ Ingestion Controls")
    
    # Chunking Controls
    chunk_size = st.slider(
        "Chunk Size (tokens/characters)",
        min_value=500,
        max_value=2000,
        value=settings.CHUNK_SIZE,
        step=100
    )
    
    chunk_overlap_percent = st.slider(
        "Chunk Overlap (%)",
        min_value=0.05,
        max_value=0.20,
        value=settings.CHUNK_OVERLAP_PERCENT,
        step=0.01,
        format="%.2f"
    )
    
    st.divider()
    
    if st.button("🚀 Run Ingestion Pipeline", type="primary", use_container_width=True):
        with st.spinner("Processing all policy documents..."):
            try:
                pipeline = IngestionPipeline(data_dir="data/raw")
                
                # Override chunker settings
                pipeline.chunker.chunk_size = chunk_size
                pipeline.chunker.chunk_overlap = int(chunk_size * chunk_overlap_percent)
                
                chunks = pipeline.run()
                
                st.success(f"✅ Ingestion completed successfully!\nTotal chunks: **{len(chunks)}**")
                
            except Exception as e:
                st.error(f"❌ Ingestion failed: {e}")

    st.divider()
    st.caption("Tip: Adjust controls above before running ingestion")

# ====================== MAIN CHAT INTERFACE ======================
st.header("💬 Policy Chat Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your corporate policy assistant. How can I help you today?"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about any policy (e.g., travel per diem, security access, leave policy...)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # TODO: Replace with actual retrieval + LLM call later
            response = "This is a placeholder response. Retrieval pipeline will be connected here in the next step."
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})

# ====================== STATUS ======================
st.divider()
col1, col2 = st.columns(2)
with col1:
    st.metric("Documents Available", "4 Categories")
with col2:
    st.metric("Pinecone Index", settings.PINECONE_INDEX_NAME)