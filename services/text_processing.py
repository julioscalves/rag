import glob
import os

from pathlib import Path

from pypdf import PdfReader
from sqlalchemy.orm import Session

from models import crud
from utils import helpers
from utils.logging import logger


def parse_pdfs(session: Session, path: str = os.getcwd() + "\\data", skip: int = 0) -> dict:
    existing_hashes = crud.get_document_hashes(session)
    data = {}

    for pdf_path in glob.glob(os.path.join(path, "*.pdf")):
        logger.info(f"parsing [{pdf_path}] file...")
        
        try:
            file_hash = helpers.generate_hash_from_file(pdf_path)

            if file_hash in existing_hashes:
                logger.info(f"...[{pdf_path}] already stored into the db, skipping...")
                continue

            content = extract_text_from_pdf(pdf_path)
            filename = pdf_path.split("\\")[-1][:-4]

            document = crud.create_document(
                session, pdf_path, filename, document_name=filename, content=content
            )

            data[filename] = {
                "filename": filename,
                "name": filename,
                "document_id": document.id,
                "hash": file_hash,
                "content": content,
            }
            logger.info(
                f"...done! [{filename}] ({file_hash}) stored with id [{document.id}]!"
            )

        except Exception as exc:
            logger.error(f"[{str(exc)}] @ {pdf_path} - skipping file...", exc_info=True)

    return data


def extract_text_from_pdf(pdf_path: str, skip: int = 0) -> str:
    if not Path(pdf_path).exists():
        logger.error(f"File not found: {pdf_path}")
        return ""

    try:
        reader = PdfReader(pdf_path)
        text_parts = []

        for page in range(skip, len(reader.pages)):
            current_page = reader.pages[page]
            page_text = current_page.extract_text(extraction_mode="plain")

            if page_text:
                cleaned_lines = [
                    line.strip()
                    for line in page_text.splitlines()
                    if line.strip() and not line.strip().isdigit()
                ]
                text_parts.append(" ".join(cleaned_lines))

        return " ".join(text_parts).strip().replace("- ", "")

    except Exception as exc:
        logger.error(f"[{str(exc)}] @ {pdf_path} - skipping file...", exc_info=True)
        return ""
