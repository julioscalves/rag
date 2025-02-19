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


@app.route("/document/update/<int:document_id>", methods=["PUT"])
def update_document_status(document_id: int) -> dict:
    session = database.LocalSession()

    try:
        document_id = document_id
        filename = request.json.get("filename")
        name = request.json.get("name")
        content = request.json.get("content")
        is_active = request.json.get("is_active")

        document = crud.update_document(
            session=session,
            document_id=document_id,
            filename=filename,
            name=name,
            content=content,
            is_active=is_active,
            embedding_model=embedding,
        )

        if document:
            session.commit()
            serialized_document = serializers.document_serializer(document)

            return {"document": serialized_document}

        session.rollback()

        return {"error": "document not found"}

    except Exception as exc:
        session.rollback()

        return {"error": str(exc)}

    finally:
        session.close()


@app.route("/text/<int:text_id>", methods=["PUT"])
def update_text_status() -> dict:
    session = database.LocalSession()

    try:
        text_id = request.json.get("text_id")
        content = request.json.get("content")
        is_active = request.json.get("is_active")

        text = crud.update_text_active_status(
            session=session,
            text_id=text_id,
            is_active=is_active,
            content=content,
            embedding=embedding,
        )

        if text:
            session.commit()

            return {"text": text}

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

        return {"document": serialized_document, "texts": serialized_texts}

    except Exception as exc:
        session.rollback()

        return {"error": str(exc)}

    finally:
        session.close()


@app.route("/question", methods=["POST"])
def question() -> dict:
    session = database.LocalSession()

    query = request.json.get("query")
    chat_id = request.json.get("chat_id")

    if not query:
        return {"error": "missing query"}

    chat = crud.get_or_create_chat(session=session, chat_id=chat_id)

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

    crud.create_message(
        session=session, message=query, chat_id=chat.id, is_output=False
    )
    crud.create_message(
        session=session, message=response_text, chat_id=chat.id, is_output=True
    )

    chat_id = chat.chat_id
    session.close()

    return {"response": response_text, "chat_id": chat_id}
