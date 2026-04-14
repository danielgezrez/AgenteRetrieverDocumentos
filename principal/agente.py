from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatHistory
from principal.config import OPENAI_API_KEY, CHAT_MODEL
from plugins.summary_plugin import SummaryPlugin

# Clase Agente RAG
class RAGAgent:
    # Constructor
    def __init__(self):
        # Inicio del kernel
        self.kernel = Kernel()
        # Creacion de conexion al kernel de OpenAI
        self.kernel.add_service(
            OpenAIChatCompletion(
                service_id="chat",  # servicio
                api_key=OPENAI_API_KEY,  # key para conexion
                ai_model_id=CHAT_MODEL  # modelo elegido a utilizar
            )
        )
        # Creacion del historial del chat
        self.history = ChatHistory()
        # Plugin definido en summary_plugin.py
        self.kernel.add_plugin(
            SummaryPlugin(self.kernel),
            plugin_name="summary"
        )


    # Funcion para construir una prompt final a partir del contexto y de la pregunta, asegurando la respuesta en base al contexto unicamente
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

    # Funcion principal
    async def ask(self, context_chunks, question):
        # Formateo del contexto: [SOURCE: file1.pdf] Texto con la info
        context_text = "\n\n".join(
            [f"[SOURCE: {c['source']}] {c['text']}" for c in context_chunks]
        )

        # Prompt final a partir de la funcion definida justo encima
        prompt = self.build_prompt(context_text, question)

        # Añadir al historial la pregunta nueva
        self.history.add_user_message(question)

        # El LLM genera la respuesta a traves de la conexion del kernel
        response = await self.kernel.invoke_prompt(prompt)

        # Utilizar plugin nativo para resumir respuesta si es demasiado larga
        if len(str(response)) > 10000:
            final_response = await self.kernel.invoke(
                plugin_name="summary",
                function_name="summarize",
                input=str(response)
            )
        else:
            final_response = response

        # Añadir al historial la respuesta
        self.history.add_assistant_message(str(final_response))

        return str(final_response)
