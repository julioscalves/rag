import nltk
import requests

from flask import Flask
from flask.globals import request
from flask_cors import CORS

import settings

from models import crud, database, serializers
from services import text_processing, embeddings
from utils.logging import logger


app = Flask(__name__)
CORS(app, origins="*")

nltk.download("wordnet")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("omw-1.4")

embedding = embeddings.Embeddings(session=database.session)
wordnet_syn = embeddings.WordnetSyn(lang="por")
# faiss_index = retrieval.FAISSIndex(session=session, embedder=embedding)
# graph = retrieval.Graph(session, embedder=embedding)


def _setup():
    from models import database

    session = database.LocalSession()

    database.Base.metadata.create_all(bind=database.engine)
    data = text_processing.parse_pdfs(session)

    for key in data.keys():
        embedding.process_data(data[key])

    wordnet_syn._precompute_mapping()
    # faiss_index.build_index()
    # graph.build_graph_network()


_setup()


@app.route("/documents", methods=["GET"])
def documents() -> dict:
    session = database.LocalSession()

    try:
        documents = crud.get_all_documents(session=session)
        serialized_documents = [serializers.document_serializer(document) for document in documents]

        return {"documents": serialized_documents}
    
    except Exception as exc:
        session.rollback()

        return {
            "error": str(exc)
        }
    
    finally:
        session.close()


@app.route("/question", methods=["POST"])
def question() -> dict:
    query = request.json.get("query")

    if not query:
        return {}

    prompt = f"Pergunta: {query}"

    context = embedding.retrieve(query, top_k=5, rerank=True)
    # context = faiss_index.search(query, top_k=20, rerank=True)
    # context = graph.retrieve(query)

    for row in context:
        print(f"\n{row.get('content')} - {row.get('cosine_similarity')}\n\n")
        prompt += f"\n\n[CONTEXTO]: {row['content']}\nFonte: {row['name']}"

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
