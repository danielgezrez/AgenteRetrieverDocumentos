import streamlit as st
from principal.lector_documentos import process_document
from principal.almacen_documentos import VectorStore
from principal.agente import RAGAgent
import asyncio
from plugins.faithfulness import calc_faithfulness

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

# Si no existe un vector para almacenar el faithfulness de cada respuesta, crealo
if "faithfulness_list" not in st.session_state:
    st.session_state.faithfulness_list = []


# Titulo
st.title("Agente Retriever de Documentos")


# Ventana para arrastrar y subir archivos
files = st.file_uploader(
    "Suba su archivo (PDFs o TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)


# Fragmentar texto de archivos subidos y guardarlos
if files:
    for file in files:
        chunks = process_document(file)  # extraer texto y fragmentarlo
        st.session_state.store.add_documents(chunks, file.name)  # insertar en el vector store los documentos
        st.success(f"Archivo almacenado: {file.name}")


# Obtener documentos disponibles
available_docs = sorted(
    {meta["source"] for meta in st.session_state.store.metadata}
)

# Filtro de documentos para que el usuario seleccione los que quiere utilizar
selected_docs = st.multiselect(
    "Seleccione los documentos a utilizar:",
    options=available_docs,
    default=available_docs
)


# Ventana de entrada de chat con boton de enviar
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Pregunte:")  # Ventana para input de chat
    submitted = st.form_submit_button("Enviar")  # Pulsado boton enviar o enter


# Si hay input del usuario y se ha pulsado enter
if user_input and submitted:
    results = st.session_state.store.search(user_input, selected_sources=selected_docs)  # devuelve fragmentos similares a la pregunta entre los documentos seleccionados

    response = asyncio.run(
        st.session_state.agent.ask(results, user_input)
    )  # pedir al agente que genere la respuesta a partir de la pregunta y la info de los fragmentos

    # Guardar en el historial del chat
    st.session_state.chat.append(("user", user_input))  # guardar en el historial del chat la pregunta del usuario
    st.session_state.chat.append(("bot", response))  # guardar en el historial del chat la respuesta del agente

    # Faithfulness
    faithfulness = calc_faithfulness(results, response)
    # Guardar faithfulness asociado a esta respuesta
    st.session_state.faithfulness_list.append(faithfulness)

# Mostrar el chat completo
for role, msg in st.session_state.chat:  # recorrer el historial de mensajes del chat
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(msg)  # Mostrar mensaje


# Boton para mostrar el faithfulness
if st.button("Calcular faithfulness"):
    if st.session_state.faithfulness_list:
        # Calculo del faithfulness medio usando el de cada respuesta
        avg_faith = sum(st.session_state.faithfulness_list) / len(st.session_state.faithfulness_list)
        st.success(f"Faithfulness: {avg_faith:.4f}")
    else:
        st.warning("Realice preguntas primero.")