import os
from dotenv import load_dotenv

# Cargar el archivo env con la info a completar del sistema operativo
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # key para la conexion
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")  # modelo a utilizar, y predeterminado en caso de que no haya uno especificado
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")  # modelo para realizar los embeddings, y predeterminado en caso de que no haya uno especificado