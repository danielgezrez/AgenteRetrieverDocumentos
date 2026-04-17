from openai import OpenAI
from principal.config import OPENAI_API_KEY, EMBEDDING_MODEL
import numpy as np

client = OpenAI(api_key=OPENAI_API_KEY)

def embed(text):
    """
    Generacion de embedding del texto utilizando el modelo especificado.
    :param text: texto
    :return: vector embedding del texto
    """
    emb = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return np.array(emb.data[0].embedding, dtype=np.float32)

def calc_faithfulness(results, response):
    """
    Calcula el faithfulness de una respuesta respecto a los fragmentos recuperados.:
    :param results: lista de fragmentos (dict con 'text')
    :param response: respuesta generada por el modelo
    :return: similitud entre la respuesta y el fragmento con mayor similitud
    """

    if not results:
        return 0.0

    # Embedding de la respuesta
    response_emb = embed(response)

    # Busqueda de fragmento usado mas cercano a la repsuesta
    similitud = 0.0
    for result in results:
        fragmento = result['text']
        # Embedding del fragmento actual
        fragmento_emb = embed(fragmento)

        # Similitud coseno entre el fragmento actual y la respuesta
        similitud_actual = np.dot(fragmento_emb, response_emb) / (np.linalg.norm(fragmento_emb) * np.linalg.norm(response_emb))

        # Si es el fragmento usado mas cercano por ahora, guardar esa similitud
        if similitud_actual > similitud:
            similitud = similitud_actual

    return similitud
