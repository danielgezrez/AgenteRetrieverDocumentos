import faiss
import numpy as np
from openai import OpenAI
from principal.config import OPENAI_API_KEY, EMBEDDING_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

# Clase para almacenar y recuperar los textos
class VectorStore:
    def __init__(self):
        self.dimension = 1536  # dimension de los embeddings
        self.index = faiss.IndexFlatL2(self.dimension)  # establece indice faiss y metodo de busqueda como distancia euclidia
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
        Genera los embeddings del texto y guarda el embedding cal vector indices faiss, el texto original, y el nombre del documento del que proviene.
        :param chunks: fragmentos de texto
        :param doc_name: documento de origen
        """
        for chunk in chunks:
            emb = self.embed(chunk)  # funcion de generacion de embedding
            self.index.add(np.array([emb]))  # guarda embedding al vector indices faiss
            self.texts.append(chunk)  # guarda texto original del chunk
            self.metadata.append({"source": doc_name})  # guarda el nombre del documento del chunk

    def search(self, query, k=5, selected_sources=None):
        """
        Busca y devuelve los fragmentos con mayor similitud a la peticion (menor distancia en el espacio euclideo).
        :param query: peticion
        :param k: numero de resultados mas similares, chunks que se utilizaran para generar la respuesta
        :param selected_sources: documentos seleccionados por el usuario
        :return: lista de diccionarios con textos mas similares a la peticion y su documento origen
        """
        q_emb = self.embed(query).reshape(1, -1)  # genera embedding de la peticion y lo normaliza

        # Si no hay filtro, usar la lista de fragmentos completa
        if not selected_sources:
            distances, indices = self.index.search(q_emb, k)  # busqueda de embeddings mas cercanos a la peticion, devuelve sus distancias L2 e indice
        else:
            # Filtrar indices de embeddings segun las fuentes seleccionadas y guardarlos en una lista
            filtered_indices = [
                i for i, meta in enumerate(self.metadata)
                if meta["source"] in selected_sources
            ]

            # Si no hay indices, no devolver ningun fragmento
            if not filtered_indices:
                return []

            # Reconstruir embeddings desde FAISS de los fragmentos de los documentos filtrados por el usuario
            filtered_embeddings = np.array(
                [self.index.reconstruct(i) for i in filtered_indices],
                dtype=np.float32
            )

            # Crear un subindice temporal para los embeddings de los fragmentos de los documentos filtrados por el usuario
            temp_index = faiss.IndexFlatL2(self.dimension)
            temp_index.add(filtered_embeddings)

            distances, temp_indices = temp_index.search(q_emb, min(k, len(filtered_indices)))  # busqueda de embeddings mas cercanos a la peticion, devuelve sus distancias L2 e indice

            # Mapear índices al original
            indices = np.array([
                [filtered_indices[i] for i in temp_indices[0]]
            ])

        # recorrer indices obtenidos para recuperar el texto correspondiente y su documento origen
        results = []
        for i in indices[0]:
            if i < len(self.texts):
                results.append({
                    "text": self.texts[i],
                    "source": self.metadata[i]["source"]
                })

        return results
