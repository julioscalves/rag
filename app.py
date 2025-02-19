import nltk
import requests

from flask import Flask
from flask.globals import request
from flask_cors import CORS

import settings

from models import crud, database, serializers
from utils import setup


from utils.logging import logger


app = Flask(__name__)
CORS(app, origins="*")

embedding = setup.initialize()


@app.route("/text/<int:text_id>", methods=["PATCH"])
def update_text_status() -> dict:
    session = database.LocalSession()

    try:
        text_id = request.json.get("text_id")
        is_active = request.json.get("is_active")

        text = crud.update_text_active_status(session=session, text_id=text_id, is_active=is_active)

        return {
            "text": text
        }

    
    except Exception as exc:
        session.rollback()

        return {"error": str(exc)}

    finally:
        session.close()

    


@app.route("/documents", methods=["GET"])
def get_all_documents() -> dict:
    session = database.LocalSession()

    try:
        documents = crud.get_all_documents(session=session)
        serialized_documents = [
            serializers.document_serializer(document) for document in documents
        ]

        return {"documents": serialized_documents}

    except Exception as exc:
        session.rollback()

        return {"error": str(exc)}

    finally:
        session.close()


@app.route("/document/<int:document_id>")
def get_text_from_document(document_id: int):
    session = database.LocalSession()

    try:
        texts = crud.get_texts_from_document_id(
            session=session, document_id=document_id
        )
        document = crud.get_document_by_id(session=session, document_id=document_id)

        serialized_document = serializers.document_serializer(document)
        serialized_texts = [serializers.text_serializer(text) for text in texts]

        return {
            "document": serialized_document,
            "texts": serialized_texts
        }

    except Exception as exc:
        session.rollback()

        return {"error": str(exc)}

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
        prompt += f"\n\n[CONTEXTO]: {row['content']}\nFonte: {row['name']}\nCosine similarity: {row['cosine_similarity']}"

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
