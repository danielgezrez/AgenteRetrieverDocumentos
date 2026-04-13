from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatHistory
from principal.config import OPENAI_API_KEY, CHAT_MODEL
from plugins.plugin_documentos import DocumentPlugin

class RAGAgent:

    def __init__(self):
        self.kernel = Kernel()

        self.kernel.add_service(
            OpenAIChatCompletion(
                service_id="chat",
                api_key=OPENAI_API_KEY,
                ai_model_id=CHAT_MODEL
            )
        )

        self.history = ChatHistory()

        self.kernel.add_plugin(
            DocumentPlugin(),
            plugin_name="doc_tools"
        )

    def build_prompt(self, context, question):
        return f"""
Eres un asistente RAG.

Usa SOLO el contexto para responder.

Contexto:
{context}

Pregunta:
{question}

Devuelve también las fuentes usadas.
"""

    async def ask(self, context_chunks, question):

        context_text = "\n\n".join(
            [f"[SOURCE: {c['source']}] {c['text']}" for c in context_chunks]
        )

        prompt = self.build_prompt(context_text, question)

        self.history.add_user_message(question)

        response = await self.kernel.invoke_prompt(prompt)

        self.history.add_assistant_message(str(response))

        return str(response)