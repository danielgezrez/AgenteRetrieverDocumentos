import faiss
import numpy as np
from openai import OpenAI
from principal.config import OPENAI_API_KEY, EMBEDDING_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

class VectorStore:
    def __init__(self):
        self.dimension = 1536
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        self.metadata = []

    def embed(self, text):
        res = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return np.array(res.data[0].embedding, dtype=np.float32)

    def add_documents(self, chunks, doc_name):
        for chunk in chunks:
            emb = self.embed(chunk)
            self.index.add(np.array([emb]))
            self.texts.append(chunk)
            self.metadata.append({"source": doc_name})

    def search(self, query, k=5):
        q_emb = self.embed(query).reshape(1, -1)
        distances, indices = self.index.search(q_emb, k)

        results = []
        for i in indices[0]:
            if i < len(self.texts):
                results.append({
                    "text": self.texts[i],
                    "source": self.metadata[i]["source"]
                })

        return results