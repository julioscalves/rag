CHUNK_SIZE = 512

DB_FILENAME = "documents.db"

ENABLE_PERF_LOGGING = True

BM25_SEARCH_WEIGHT = 0.2
EMBEDDINGS_SEARCH_WEIGHT = 0.8

ALLOWED_UPLOAD_FORMAT = {".txt", ".doc", ".docx", ".pdf"}
UPLOAD_FOLDER = "\\data"

CROSSENCODER_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
EMBEDDINGS_MODEL = "intfloat/multilingual-e5-small"  # "all-MiniLM-L6-v2"
FAISS_DIMENSION = 384  # Property from the embedding model
MIN_CHARS_PER_CHUNK = 128

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_PARAMETERS = {
    "temperature": 0.70,
    "num_ctx": 4096,
    "top_k": 30,
    "top_p": 0.8,
    "mirostat": 2,
}
OLLAMA_SYSTEM_PROMPT = """
                        Você é um assistente acadêmico de informações da Univerdade Federal Rural do Semi-Árido (UFERSA). 

                        Responda a pergunta do usuário da forma mais precisa possível, considerando o contexto acadêmico da UFERSA 
                        de forma estrita.

                        Sempre que disponível, serão fornecidas informações institucionais como contextualização para a sua
                        resposta. Avalie a pergunta e o contexto para formular a sua resposta

                        NÃO formule opinião sobre qualquer membro da comunidade acadêmica.
                        NÃO realize comentários de cunho político, religioso ou que afete convicções pessoais.
                        """.strip()
