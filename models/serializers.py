import base64

from models import schema


def document_serializer(document: schema.Document):
    return {
        "id": document.id,
        "filename": document.filename,
        "name": document.name,
        "hash": document.hash,
        "content": document.content,
        "is_active": document.is_active,
        "texts": len(document.texts),
    }


def text_serializer(text: schema.Text):
    return {
        "id": text.id,
        "content": text.content,
        "hash": text.hash,
        "is_active": text.is_active,
        "embedding": base64.b64encode(text.embedding).decode("utf-8")
        if text.embedding
        else None,
    }
