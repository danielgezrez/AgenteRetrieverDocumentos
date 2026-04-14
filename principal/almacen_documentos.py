import faiss
import numpy as np
from openai import OpenAI
from principal.config import OPENAI_API_KEY, EMBEDDING_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

# Clase para almacenar y recuperar los textos
class VectorStore:
    def __init__(self):
        self.dimension = 1536  # dimension de los embeddings
        self.index = faiss.IndexFlatL2(self.dimension)  # establece indice faiss como distancia euclidia
        self.texts = []  # almacena textos originales
        self.metadata = []  # almacena nombre del texto

    def embed(self, text):
        """
        Generacion de embedding del texto utilizando el modelo especificado.
        :param text: texto
        :return: vector embedding del texto
        """
        res = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return np.array(res.data[0].embedding, dtype=np.float32)

    def add_documents(self, chunks, doc_name):
        """
        Genera los embeddings del texto y guarda suindice faiss, el texto original, y el nombre del documento del que proviene.
        :param chunks: fragmentos de texto
        :param doc_name: documento de origen
        """
        for chunk in chunks:
            emb = self.embed(chunk)  # funcion de generacion de embedding
            self.index.add(np.array([emb]))  # adicion indice faiss
            self.texts.append(chunk)  # guarda texto original del chunk
            self.metadata.append({"source": doc_name})  # guarda el nombre del documento del chunk

    def search(self, query, k=5):
        """
        Busca y devuelve los fragmentos con mayor similitud a la peticion (menor distancia en el espacio euclideo).
        :param query: peticion
        :param k: numero de resultados mas similares, chunks que se utilizaran para generar la respuesta
        :return: lista de diccionarios con textos mas similares a la peticion y su documento origen
        """
        q_emb = self.embed(query).reshape(1, -1)  # genera embedding de la peticion y lo normaliza
        distances, indices = self.index.search(q_emb, k)  # busqueda del indice faiss, devuelve distancias L2 y posiciones de los vectores mas cercanos

        # recorrer indices obtenidos para recuperar el texto correspondiente y su documento origen
        results = []
        for i in indices[0]:
            if i < len(self.texts):
                results.append({
                    "text": self.texts[i],
                    "source": self.metadata[i]["source"]
                })

        return results
