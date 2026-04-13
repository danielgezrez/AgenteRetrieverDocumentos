from pypdf import PdfReader
from principal.procesador_textos import chunk_text

def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    else:
        raise ValueError("Formato no soportado")


def process_document(file):
    text = extract_text(file)
    chunks = chunk_text(text)

    return chunks