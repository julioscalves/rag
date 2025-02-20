import base64
import json
import os

import requests

from flask import Flask, Response, stream_with_context
from flask.globals import request
from flask_cors import CORS

import settings

from services import text_processing
from models import crud, database, serializers
from utils import setup


from utils.logging import logger


app = Flask(__name__)
CORS(app, origins="*")

embedding = setup.initialize()


@app.route(f"/{settings.API_VERSION}/document/upload/", methods=["POST"])
def upload_document() -> dict:
    session = database.LocalSession()
    data = request.get_json()

    if not data:
        return {"error": "missing file"}

    filename = data.get("filename")
    name = data.get("name")
    content_base64 = data.get("content")
    filetype = data.get("filetype")

    if not filetype:
        return {"error": "unknown file type"}

    if not content_base64:
        return {"error": "missing content"}

    try:
        file_content = base64.b64decode(content_base64)

    except Exception as exc:
        return {"error": f"failed to decode file content: {str(exc)}"}

    current_dir = os.path.join(os.getcwd(), settings.UPLOAD_FOLDER)
    file_destination = os.path.join(current_dir, filename)

    try:
        with open(file_destination, "wb") as file:
            file.write(file_content)

    except Exception as exc:
        return {"error": f"failed to save file: {str(exc)}"}

    try:
        data = text_processing.parse_upload_file(
            session=session,
            filename=filename,
            name=name,
            content=file_content,
            filepath=file_destination,
        )
        embedding.process_data(data)

    except Exception as exc:
        return {"error": f"failed processing the file: {str(exc)}"}

    return {"message": "File save successfully!"}


@app.route(f"/{settings.API_VERSION}/document/update/", methods=["PUT"])
def update_document() -> dict:
    session = database.LocalSession()

    try:
        document_id = request.json.get("document_id")
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


@app.route(f"/{settings.API_VERSION}/documents", methods=["GET"])
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


@app.route(f"/{settings.API_VERSION}/text/", methods=["PUT"])
def update_text() -> dict:
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


@app.route(f"/{settings.API_VERSION}/document/texts")
def get_text_from_document():
    session = database.LocalSession()
    document_id = request.json.get("document_id")

    if not document_id:
        return {"error": "missing document id"}

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


@app.route(f"/{settings.API_VERSION}/question", methods=["POST"])
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


@app.route(f"/{settings.API_VERSION}/chat", methods=["POST"])
def chat() -> dict:
    session = database.LocalSession()
    query = request.json.get("query")

    chat_id = request.json.get("chat_id")
    chat = crud.get_or_create_chat(session=session, chat_id=chat_id)

    prompt = f"Mensagem: {query}"
    context = embedding.retrieve(query, top_k=5, rerank=True)

    for row in context:
        prompt += f"\n\n[CONTEXTO]: {row['content']}\nFonte: {row['name']}\nCosine similarity: {row['cosine_similarity']}"

    payload = {
        "model": settings.OLLAMA_MODEL,
        "system": settings.OLLAMA_SYSTEM_PROMPT,
        "prompt": prompt,
        "options": settings.OLLAMA_PARAMETERS,
        "stream": True,
    }

    response = requests.post(url=settings.OLLAMA_ENDPOINT, json=payload, stream=True)
    response_text = ""

    def generate():
        nonlocal response_text

        try:
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue

                    token = chunk.get("response", "")
                    response_text += token

                    yield f"data: {token}\n\n"

        finally:
            crud.create_message(
                session=session, message=query, chat_id=chat.id, is_output=False
            )
            crud.create_message(
                session=session,
                message=response_text,
                chat_id=chat.id,
                is_output=True,
            )
            session.close()

    return Response(stream_with_context(generate()), mimetype="text/event-stream")
