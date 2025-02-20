import glob
import os
import re

from pathlib import Path

from pypdf import PdfReader
from sqlalchemy.orm import Session

from models import crud
from models import schema
from utils import helpers
from utils.logging import logger


@helpers.measure_time
def parse_pdfs(
    session: Session, path: str = os.path.join(os.getcwd(), "data"), skip: int = 0
) -> dict:
    existing_documents = {
        document.file_hash: document for document in crud.get_all_documents(session)
    }
    current_hashes = set()
    data = {}

    def _get_filename(pdf_path: str) -> str:
        return os.path.splitext(os.path.basename(pdf_path))

    def _should_skip_file(file_hash: str, document: schema.Document) -> bool:
        if file_hash in current_hashes:
            logger.info(
                f"skipping [{pdf_path}] ({file_hash}) as its' hash has already been processed"
            )
            return True

        if document and len(document.texts) > 0:
            logger.info(f"skipping [{pdf_path}] since it is already in the database")
            return True

        return False

    def _process_pdf(pdf_path: str) -> tuple[dict, str]:
        file_hash = helpers.generate_hash_from_file(pdf_path)
        document = existing_documents.get(file_hash)

        if _should_skip_file(file_hash=file_hash, document=document):
            return None, None

        content = extract_text_from_pdf(pdf_path)
        filename, _ = _get_filename(pdf_path)

        if not document:
            document = crud.create_document(
                session, pdf_path, filename, document_name=filename, content=content
            )
            logger.info(f"created new database entry for [{pdf_path}]: [{document.id}]")

        return {
            "filename": filename,
            "name": filename,
            "document_id": document.id,
            "file_hash": file_hash,
            "content_hash": file_hash,
            "content": content,
        }, file_hash

    for pdf_path in glob.glob(os.path.join(path, "*.pdf")):
        logger.info(f"processing: [{pdf_path}]")

        try:
            entry, file_hash = _process_pdf(pdf_path)

            if entry and file_hash:
                data[entry.get("filename")] = entry
                current_hashes.add(file_hash)

        except Exception as exc:
            logger.error(
                f"error processing [{pdf_path}]: {str(exc)} - skipping file...",
                exc_info=True,
            )

    return data


@helpers.measure_time
def extract_text_from_pdf(pdf_path: str, skip: int = 0) -> str:
    filepath = Path(pdf_path)

    if not filepath.exists():
        logger.error(f"File not found: {pdf_path}")
        return ""

    def _clean_page_text(text: str) -> str:
        return " ".join(
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().isdigit()
        )

    def fix_hyphenation(text: str) -> str:
        return re.sub(r"-\s+", "", text)

    try:
        reader = PdfReader(pdf_path)
        text_parts = []

        for page in range(skip, len(reader.pages)):
            current_page = reader.pages[page]
            raw_text = current_page.extract_text(extraction_mode="plain")

            if not raw_text:
                continue

            cleaned_text = _clean_page_text(raw_text)
            text_parts.append(cleaned_text)

        full_text = " ".join(text_parts).strip()
        full_text = fix_hyphenation(full_text)

        return full_text

    except Exception as exc:
        logger.error(f"[{str(exc)}] @ {pdf_path} - skipping file...", exc_info=True)
        return ""


@helpers.measure_time
def parse_upload_file(
    session: Session, filename: str, name: str, content: str, filepath: str
):
    existing_documents = {
        document.file_hash: document for document in crud.get_all_documents(session)
    }
    file_hash = helpers.generate_hash_from_file(filepath=filepath)

    if file_hash in existing_documents:
        logger.info(f"skipping [{filepath}] since it is already in the database")

        return

    document = crud.create_document(
        session=session,
        filepath=filepath,
        filename=filename,
        document_name=name,
        content=content,
    )
    data = {
        "filename": filename,
        "name": filename,
        "document_id": document.id,
        "file_hash": file_hash,
        "content_hash": file_hash,
        "content": content,
    }

    return data
