from sentence_transformers import SentenceTransformer
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from chonkie import RecursiveChunker, RecursiveRules

import settings

from utils import helpers
from utils.logging import logger
from models import crud, schema


class Embeddings:
    def __init__(self, session):
        self.session = session
        self.model = SentenceTransformer(settings.EMBEDDINGS_MODEL)
        logger.info(f"initializing the embeddings class [model: {settings.EMBEDDINGS_MODEL}]")

    def tokenize(self, chunks: list):
        return [self.model.tokenize(chunk, return_tensors="pt") for chunk in chunks]

    def generate_embeddings(self, chunks: list):
        return self.model.encode(chunks, convert_to_numpy=True)

    def generate_chunks(self, text):
        chunker = RecursiveChunker(
            tokenizer=settings.EMBEDDINGS_MODEL,
            chunk_size=settings.CHUNK_SIZE,
            rules=RecursiveRules(),
            min_characters_per_chunk=128,
        )
        chunks = chunker.chunk(text)

        return list(set([chunk.text for chunk in chunks]))

    def process_data(self, data: dict) -> None:
        logger.info(f"processing data for {data.get("filename")}...")
        chunks = self.generate_chunks(data.get("content"))
        total_chunks = len(chunks)
        logger.info(f"[{data.get("filename")}] [{total_chunks}] chunks generated!")
        
        insert_data = []

        for chunk in range(0, total_chunks):
            text_hash = helpers.generate_hash_from_string(chunks[chunk])

            if not crud.get_texts_by_hash(self.session, hash=text_hash):
                embedding = self.generate_embeddings(chunks[chunk])
                insert_data.append(
                    {
                        "document_id": data["document_id"],
                        "content": chunks[chunk],
                        "hash": text_hash,
                        "embedding": embedding.tobytes(),
                    }
                )

        self.session.bulk_insert_mappings(schema.Text, insert_data)
        self.session.commit()
        logger.info(f"[{data.get("filename")}] [{len(insert_data)}] embeddings saved!")

    @staticmethod
    def _pack_data(text: schema.Text, similarity: float) -> dict:
        return {
            "id": text.id,
            "filename": text.document.filename,
            "name": text.document.name,
            "content": text.content,
            "cosine_similarity": similarity,
        }

    def retrieve(self, query, top_k: int = 5) -> list:
        query_embedding = self.model.encode(query)
        active_texts = crud.get_active_texts_from_active_documents(self.session)
        results = []

        for text in active_texts:
            document_embeddings = np.frombuffer(text.embedding, dtype=np.float32)
            similarity = cosine_similarity([query_embedding], [document_embeddings])
            results.append(self._pack_data(text, similarity[0][0]))

        results = sorted(
            results, key=lambda x: x.get("cosine_similarity"), reverse=True
        )[:top_k]

        return results
