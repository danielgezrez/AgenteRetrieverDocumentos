from pypdf import PdfReader
import tiktoken

def extract_text(file):
    """
    Funcion que extrae el texto del archivo subido por el usuario.
    :param file: archivo subido por el usuario
    :return: texto extraido
    """
    # Tratamiento si es pdf
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    # Tratamiento si es txt
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    # Cualquier otro formato no podra leerse
    else:
        raise ValueError("Formato no soportado")

def chunk_text(text, chunk_size=800, overlap=150):
    """
    Funcion que divide el texto en fragmentos mas pequeños.
    :param text: Texto a dividir en fragmentos
    :param chunk_size: Numero de tokens del fragmento
    :param overlap: Numero de tokens que se repiten entre fragmentos consecutivos
    :return: Fragmentos del texto
    """
    enc = tiktoken.get_encoding("cl100k_base")  # codificador
    tokens = enc.encode(text)  # codificacion del texto en tokens enteros

    chunks = []
    start = 0

    while start < len(tokens):  # mientras queden tokens
        end = start + chunk_size  # indice final del fragmento actual
        chunk = tokens[start:end]  # tokens fragmento actual
        chunks.append(enc.decode(chunk))  # decodificar y guardar fragmento actual
        start += chunk_size - overlap  # actualizar principio del fragmento actual

    return chunks

def process_document(file):
    """
    Funcion que extrae el texto del archivo y lo divide en fragmentos mas pequeños.
    :param file: Archivo
    :return: Fragmentos del texto
    """
    # Extraer texto del archivo con la funcion extract text definida arriba
    text = extract_text(file)
    # Dividir texto en fragmentos pequeños con la funcion chunk_text definida arriba
    chunks = chunk_text(text)

    return chunks
