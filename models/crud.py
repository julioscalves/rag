from sqlalchemy import select
from sqlalchemy.orm import Session

from models.schema import Document, Text, Chat, Message

from services import embeddings

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
def get_document_by_id(session: Session, document_id: int) -> Document:
    return session.query(Document).filter_by(id=document_id).first()


@helpers.measure_time
def get_document_by_hash(session: Session, hash: str) -> Document:
    return session.query(Document).filter_by(hash=hash).first()


@helpers.measure_time
def get_all_documents(session: Session) -> list[Document]:
    return session.query(Document).all()


@helpers.measure_time
def get_all_active_documents(session: Session) -> list[Document]:
    return session.query(Document).filter_by(is_active=True).all()


@helpers.measure_time
def get_all_document_hashes(session: Session) -> set[str] | set:
    query = session.scalars(select(Document.hash)).all()

    if query:
        return set(query)

    return set()


@helpers.measure_time
def update_document_active_status(
    session: Session, document_id: int, is_active: bool
) -> Document | None:
    document = session.query(Document).filter_by(id=document_id).first()

    if not document:
        return None

    document.is_active = is_active

    return document


@helpers.measure_time
def delete_document(session: Session, document_id: int) -> None:
    document = session.query(Document).filter_by(id=document_id).first()

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
    return session.query(Text).filter_by(document_id=document_id).all()


@helpers.measure_time
def get_all_texts(session: Session) -> list[Text]:
    return session.query(Text).all()


@helpers.measure_time
def get_texts_from_active_documents(session: Session) -> list[Text]:
    return session.query(Text).join(Document).filter(Document.is_active == True).all()


@helpers.measure_time
def get_active_texts_from_active_documents(session: Session) -> list[Text]:
    return (
        session.query(Text)
        .join(Document)
        .filter(Document.is_active == True)
        .filter(Text.is_active == True)
        .all()
    )


@helpers.measure_time
def get_texts_by_hash(session: Session, hash: str) -> Text:
    return session.query(Text).filter_by(hash=hash).first()


@helpers.measure_time
def get_all_text_hashes_in_list(session: Session, hash_list: list[str]) -> list[str]:
    return session.query(Text.hash).filter(Text.hash.in_(hash_list)).all()


@helpers.measure_time
def get_texts_in_id_list(session: Session, id_list: list[int]) -> list[Text]:
    return session.query(Text).filter(Text.id.in_(id_list)).all()


@helpers.measure_time
def update_text_active_status(
    session: Session, text_id: int, is_active: bool
) -> Text | None:
    text = session.query(Text).filter_by(id=text_id).first()

    if not text:
        return None

    text.is_active = is_active

    return text


@helpers.measure_time
def update_document(
    session: Session,
    document_id: int,
    filename: str = None,
    name: str = None,
    content: str = None,
    is_active: bool = None,
    embedding_model: embeddings.Embeddings = None,
):
    document = get_document_by_id(session=session, document_id=document_id)

    if not document:
        return None

    if filename:
        document.filename = filename

    if name:
        document.name = name

    if content:
        document.content = content
        document.hash = helpers.generate_hash_from_string(content)

        delete_texts_by_document_id(session=session, document_id=document_id)

        text_chunks = embedding_model.generate_chunks()
        text_embeddings = embedding_model.generate_embeddings(text_chunks)

        new_data = [
            {
                "document_id": document_id,
                "content": chunk,
                "hash": helpers.generate_hash_from_string(chunk),
                "embedding": embedding.tobytes(),
            }
            for chunk, embedding in zip(text_chunks, text_embeddings)
        ]

        session.bulk_insert_mappings(Text, new_data)

    if is_active is not None:
        document.is_active = is_active

    return document

@helpers.measure_time
def delete_texts_by_document_id(session: Session, document_id: int):
    texts = get_texts_from_document_id(session=session, document_id=document_id)

    if not texts:
        return 0

    delete_count = 0
    for text in texts:
        delete_count += 1
        session.delete(text)

    session.commit()
    return delete_count


@helpers.measure_time
def update_text(
    session: Session,
    text_id: int,
    content: str = None,
    is_active: bool = None,
    embedding_model: embeddings.Embeddings = None,
):
    text = get_text_by_id(session=session, text_id=text_id)

    if not text:
        return None

    if content:
        text.content = content
        text.hash = helpers.generate_hash_from_string(content)
        text.embedding = embedding_model.model.encode(content)

    if is_active is not None:
        text.is_active = is_active

    return text


@helpers.measure_time
def get_chat_by_string_id(session: Session, chat_id: str):
    return session.query(Chat).filter_by(chat_id=chat_id).first()


@helpers.measure_time
def get_chat_by_id(session: Session, chat_id: int):
    return session.query(Chat).filter_by(id=chat_id).first()


@helpers.measure_time
def create_chat(session: Session) -> Chat:
    chat_id = helpers.generate_random_id()

    while get_chat_by_string_id(session=session, chat_id=chat_id):
        chat_id = helpers.generate_random_id()

    chat = Chat(chat_id=chat_id)

    session.add(chat)
    session.commit()

    return chat


@helpers.measure_time
def get_or_create_chat(session: Session, chat_id: int = None) -> Chat:
    chat = None

    if chat_id and type(chat_id) == str:
        chat = get_chat_by_string_id(session=session, chat_id=chat_id)

    elif chat_id and type(chat_id) == int:
        chat = get_chat_by_id(session=session, chat_id=chat_id)

    if not chat:
        chat = create_chat(session=session)

    return chat


@helpers.measure_time
def create_message(
    session: Session,
    message: str,
    chat_id: int,
    is_output: bool = False,
    liked: bool = False,
) -> Message:
    chat = get_chat_by_id(session=session, chat_id=chat_id)

    if not chat:
        chat = create_chat(session=session)

    message = Message(
        content=message, is_output=is_output, liked=liked, chat_id=chat.id
    )

    session.add(message)
    session.commit()

    return message
