from sqlalchemy import select
from sqlalchemy.orm import Session

from models.schema import Document, Text

from utils import helpers


@helpers.measure_time
def create_document(
    session: Session,
    filepath: str,
    filename: str,
    document_name: str,
    content: str,
    is_active: bool = True,
) -> Document:
    document_hash = helpers.generate_hash_from_file(filepath)
    document = Document(
        filename=filename,
        name=document_name,
        hash=document_hash,
        content=content,
        is_active=is_active,
    )

    session.add(document)
    session.commit()

    return document


@helpers.measure_time
def get_document_by_id(session: Session, id: int) -> Document:
    return session.query(Document).filter_by(id=id).first()


@helpers.measure_time
def get_document_by_hash(session: Session, hash: str) -> Document:
    return session.query(Document).filter_by(hash=hash).first()


@helpers.measure_time
def get_documents(session: Session) -> list[Document]:
    return session.query(Document).all()


@helpers.measure_time
def get_active_documents(session: Session) -> list[Document]:
    return session.query(Document).filter_by(is_active=True).all()


@helpers.measure_time
def get_document_hashes(session: Session) -> set[str] | set:
    query = session.scalars(select(Document.hash)).all()

    if query:
        return set(query)

    return set()


@helpers.measure_time
def update_document_active_status(
    session: Session, id: int, is_active: bool
) -> Document | None:
    document = session.query(Document).filter_by(id=id).first()

    if not document:
        return None

    document.is_active = is_active

    return document


@helpers.measure_time
def delete_document(session: Session, id: int) -> None:
    document = session.query(Document).filter_by(id=id).first()

    if document:
        session.delete(document)
        session.commit()


@helpers.measure_time
def create_text(
    session: Session, document_id: int, content: str, embedding: bytes
) -> Text:
    text_hash = helpers.generate_hash_from_string(content)
    text = Text(
        document_id=document_id, content=content, hash=text_hash, embedding=embedding
    )

    session.add(text)
    session.commit()

    return text


@helpers.measure_time
def get_text_by_id(session: Session, text_id: int) -> Text:
    return session.query(Text).filter_by(id=text_id).first()


@helpers.measure_time
def get_texts_from_document_id(session: Session, document_id: int) -> list[Text]:
    texts = session.query(Text).filter_by(document_id=document_id).all()
    return texts


@helpers.measure_time
def get_texts(session: Session) -> list[Text]:
    return session.query(Text).all()


@helpers.measure_time
def get_texts_from_active_documents(session: Session) -> list[Text]:
    return session.query(Text).join(Document).filter(Document.is_active == True).all()


@helpers.measure_time
def get_active_texts_from_active_documents(session: Session) -> list[Text]:
    return session.query(Text).join(Document).filter(Document.is_active == True).filter(Text.is_active == True).all()


@helpers.measure_time
def get_texts_by_hash(session: Session, hash: str) -> Text:
    return session.query(Text).filter_by(hash=hash).first()
