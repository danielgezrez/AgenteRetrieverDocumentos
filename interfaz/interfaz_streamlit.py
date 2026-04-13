import streamlit as st
from principal.lector_documentos import process_document
from principal.almacen_documentos import VectorStore
from principal.agente import RAGAgent
import asyncio

st.set_page_config(page_title="RAG Semantic Kernel", layout="wide")

if "store" not in st.session_state:
    st.session_state.store = VectorStore()

if "agent" not in st.session_state:
    st.session_state.agent = RAGAgent()

if "chat" not in st.session_state:
    st.session_state.chat = []


st.title("RAG Agent con Semantic Kernel")

# Upload
files = st.file_uploader(
    "Sube PDFs o TXT",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if files:
    for file in files:
        chunks = process_document(file)
        st.session_state.store.add_documents(chunks, file.name)
        st.success(f"Indexado: {file.name}")

# Chat UI
user_input = st.text_input("Haz tu pregunta")

if user_input:
    results = st.session_state.store.search(user_input)

    response = asyncio.run(
        st.session_state.agent.ask(results, user_input)
    )

    st.session_state.chat.append(("user", user_input))
    st.session_state.chat.append(("bot", response))

# Render chat
for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"**Tú:** {msg}")
    else:
        st.markdown(f"**Agente:** {msg}")