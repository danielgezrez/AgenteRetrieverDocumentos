import streamlit as st
from principal.lector_documentos import process_document
from principal.almacen_documentos import VectorStore
from principal.agente import RAGAgent
import asyncio


# Configuracion inicial de la pagina
st.set_page_config(page_title="Agente Retriever de Documentos", layout="wide")


# Si no existe objeto store, crealo (para almacenar emneddings de los textos)
if "store" not in st.session_state:
    st.session_state.store = VectorStore()

# Si no existe el objeto del agente RAG, crealo
if "agent" not in st.session_state:
    st.session_state.agent = RAGAgent()

# Si no existe el historial de chat, crealo
if "chat" not in st.session_state:
    st.session_state.chat = []


# Titulo
st.title("Agente Retriever de Documentos")

# Ventana para arrastrar y subir archivos
files = st.file_uploader(
    "Suba su archivo (PDFs o TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)


# Indexar archivos subidos y guardarlos
if files:
    for file in files:
        chunks = process_document(file)  # lector de documentos
        st.session_state.store.add_documents(chunks, file.name)
        st.success(f"Archivo almacenado: {file.name}")


# Formulario de entrada de chat
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Pregunte:")  # Ventana para input de chat
    submitted = st.form_submit_button("Enviar")


# Si hay input del usuario
if user_input and submitted:
    results = st.session_state.store.search(user_input)  # devuelve embeddings similares a la pregunta

    response = asyncio.run(
        st.session_state.agent.ask(results, user_input)
    )  # pedir al agente que genere la respuesta a partir de la pregunta y la info de los embeddings

    # Guardar en el historial del chat
    st.session_state.chat.append(("user", user_input))  # guardar en el historial del chat la pregunta del usuario
    st.session_state.chat.append(("bot", response))  # guardar en el historial del chat la respuesta del agente


# Mostrar el chat completo
for role, msg in st.session_state.chat:  # recorrer el historial del chat
    if role == "user":
        st.markdown(f"**Tú:** {msg}")  # mostrar mensajes del usuario
    else:
        st.markdown(f"**Agente:** {msg}")  # mostrar respuestas del agente
