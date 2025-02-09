import nltk
import requests

from flask import Flask
from flask.globals import request

import settings

from models.database import session
from services import text_processing, embeddings, retrieval
from utils.logging import logger


app = Flask(__name__)
embedding = embeddings.Embeddings(session=session)
wordnet_syn = embeddings.WordnetSyn(lang="por")
faiss_index = retrieval.FAISSIndex(session=session, embedder=embedding)
graph = retrieval.Graph(session, embedder=embedding)


def setup():
    from models import schema
    from models import database

    database.Base.metadata.create_all(bind=database.engine)
    data = text_processing.parse_pdfs(session)

    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("omw-1.4")
    nltk.download("wordnet")

    for key in data.keys():
        embedding.process_data(data[key])

    wordnet_syn._precompute_mapping()
    faiss_index.build_index()
    graph.build_graph_network()


setup()


@app.route("/question", methods=["POST"])
def question() -> dict:
    query = request.json.get("query")

    if not query:
        return {}

    prompt = f"Pergunta: {query}"

    embedding_context = embedding.retrieve(query)
    faiss_context = faiss_index.search(query)
    graph_context = graph.retrieve(query)

    for row in embedding_context:
        print(f"\n{row.get('content')} - {row.get('cosine_similarity')}\n\n")
        prompt += f"\n\nContexto: {row['content']}\nFonte: {row['name']}"

    logger.info(f"[Query consolidada]: {prompt}")

    payload = {
        "model": settings.OLLAMA_MODEL,
        "system": settings.OLLAMA_SYSTEM_PROMPT,
        "prompt": prompt,
        "options": settings.OLLAMA_PARAMETERS,
        "stream": False,
    }

    logger.info(f"[Payload enviado]: {payload}")

    response = requests.post(url=settings.OLLAMA_ENDPOINT, json=payload)
    response_text = response.json()["response"]
    print(response_text)

    return {"response": response_text}


if __name__ == "__main__":
    setup()
