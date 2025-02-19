import datetime

from sqlalchemy import ForeignKey, LargeBinary, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models import database


class Document(database.Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(nullable=False)
    name: Mapped[str] = mapped_column(nullable=False)
    hash: Mapped[str] = mapped_column(nullable=False, unique=True)
    content: Mapped[str] = mapped_column(nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True)

    texts: Mapped[list["Text"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"[{self.filename}] - {self.name}"


class Text(database.Base):
    __tablename__ = "texts"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id"), nullable=False, index=True
    )
    content: Mapped[str] = mapped_column(nullable=False)
    hash: Mapped[str] = mapped_column(nullable=False, unique=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    document: Mapped["Document"] = relationship(back_populates="texts")

    def __repr__(self) -> str:
        return f"[{self.document_id}] - [{self.id}]"


class Chat(database.Base):
    __tablename__ = "chats"

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_id: Mapped[str] = mapped_column(unique=True, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
    messages: Mapped[list["Message"]] = relationship(
        back_populates="chat", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"Chat(id={self.id}, chat_id={self.chat_id}, created_at={self.created_at}, updated_at={self.updated_at})"


class Message(database.Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    content: Mapped[str] = mapped_column(nullable=False)
    timestamp: Mapped[datetime.datetime] = mapped_column(server_default=func.now())
    is_output: Mapped[bool] = mapped_column(default=False)
    liked: Mapped[bool] = mapped_column(default=False)

    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id"))
    chat: Mapped["Chat"] = relationship(back_populates="messages")

    def __repr__(self) -> str:
        return f"Message(id={self.id}, content={self.content[:20]}..., timestamp={self.timestamp}, is_output={self.is_output}, chat_id={self.chat_id})"
