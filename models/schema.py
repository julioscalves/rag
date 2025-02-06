from sqlalchemy import ForeignKey, LargeBinary
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models import database


class Document(database.Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(nullable=False)
    name: Mapped[str] = mapped_column(nullable=False)
    hash: Mapped[str] = mapped_column(nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True)

    texts: Mapped[list["Text"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"[{self.filename}] - {self.name}"


class Text(database.Base):
    __tablename__ = "texts"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id"), nullable=False, index=True
    )
    content: Mapped[str] = mapped_column(nullable=False)
    hash: Mapped[str] = mapped_column(nullable=False, unique=True)
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    document: Mapped["Document"] = relationship(back_populates="texts")

    def __repr__(self):
        return f"[{self.document_id}] - [{self.id}]"
