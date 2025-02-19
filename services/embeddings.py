import re

from collections import defaultdict
from typing import Optional

import numpy as np

from chonkie import RecursiveChunker, RecursiveRules
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from rank_bm25 import BM25Okapi

import settings

from utils import helpers
from utils.logging import logger
from models import crud, schema


class WordnetSyn:
    """
    This helper class preloads Wordnet's synsets, speeding up the
    synonym search.
    """

    def __init__(self, lang: str = "por", preload: bool = True):
        self.lang = lang
        self.syn_mapping = defaultdict(set)

        if preload:
            self._precompute_mapping()

    @helpers.measure_time
    def _precompute_mapping(self):
        for synset in wordnet.all_synsets(lang=self.lang):
            lemmas = synset.lemma_names(self.lang)
            processed_lemmas = {lemma.lower().replace("_", " ") for lemma in lemmas}

            for lemma in lemmas:
                self.syn_mapping[lemma].update(processed_lemmas)

        self.syn_mapping = {
            lemma: frozenset(synonyms) for lemma, synonyms in self.syn_mapping.items()
        }

    @helpers.measure_time
    def get_synonyms(self, token: str) -> frozenset:
        return self.syn_mapping.get(token, frozenset())


class Embeddings:
    def __init__(self, session: Session):
        self.session = session
        self.model = SentenceTransformer(settings.EMBEDDINGS_MODEL)
        self.cross_encoder = CrossEncoder(settings.CROSSENCODER_MODEL)
        self.chunker = RecursiveChunker(
            tokenizer=settings.EMBEDDINGS_MODEL,
            chunk_size=settings.CHUNK_SIZE,
            rules=RecursiveRules(),
            min_characters_per_chunk=settings.MIN_CHARS_PER_CHUNK,
        )
        self.wordnet_syn = None
        logger.info(
            f"initializing the embeddings class [model: {settings.EMBEDDINGS_MODEL}]"
        )

    def tokenize(self, chunks: list) -> list:
        return [self.model.tokenize(chunk, return_tensors="pt") for chunk in chunks]

    def generate_embeddings(self, chunks: list):
        return self.model.encode(chunks, convert_to_numpy=True)

    def _remove_meaningless_chunks(self, chunks: list[str]) -> list[str]:
        filtered_chunks = [
            string for string in chunks if re.search(r"[a-zA-Z0-9]", string)
        ]
        logger.info(f"[{len(chunks) - len(filtered_chunks)}] chunks filtered out")
        return filtered_chunks

    def generate_chunks(self, text: str) -> list:
        chunks = list(set([chunk.text for chunk in self.chunker.chunk(text)]))
        chunks = self._remove_meaningless_chunks(chunks)

        return chunks

    def process_data(self, data: dict) -> None:
        logger.info(f"processing data for {data.get('filename')}...")
        chunks = self.generate_chunks(data.get("content"))
        total_chunks = len(chunks)
        logger.info(f"[{data.get('filename')}] [{total_chunks}] chunks generated!")

        chunk_hashes = [helpers.generate_hash_from_string(chunk) for chunk in chunks]
        existing_hashes = crud.get_all_text_hashes_in_list(self.session, chunk_hashes)
        existing_hash_set = {hash_tuple[0] for hash_tuple in existing_hashes}

        insert_data = []

        for chunk, chunk_hash in zip(chunks, chunk_hashes):
            if chunk_hash in existing_hash_set:
                continue

            embedding = self.generate_embeddings(chunk)
            insert_data.append(
                {
                    "document_id": data["document_id"],
                    "content": chunk,
                    "hash": chunk_hash,
                    "embedding": embedding.tobytes(),
                }
            )

        if insert_data:
            self.session.bulk_insert_mappings(schema.Text, insert_data)
            self.session.commit()

        logger.info(f"[{data.get('filename')}] [{len(insert_data)}] embeddings saved!")

    @staticmethod
    def _pack_data(text: schema.Text, similarity: float) -> dict:
        return {
            "id": text.id,
            "filename": text.document.filename,
            "name": text.document.name,
            "content": text.content,
            "cosine_similarity": similarity,
        }

    @helpers.measure_time
    def _fetch_results(self, query: str, active_texts: Optional[list] = None) -> list:
        query_embedding = self.model.encode(query)

        if active_texts is None:
            active_texts = crud.get_active_texts_from_active_documents(self.session)

        document_embeddings = np.stack(
            [np.frombuffer(text.embedding, dtype=np.float32) for text in active_texts]
        )
        similarities = cosine_similarity([query_embedding], document_embeddings)[0]

        results = [
            self._pack_data(text, similarity)
            for text, similarity in zip(active_texts, similarities)
        ]

        return results

    @helpers.measure_time
    def _rerank_results(
        self, query: str, results: list, rerank_top_k: int = 5, threshold: float = -5.0
    ) -> list:
        logger.info("reranking results...")

        query_document_pairs = [(query, result["content"]) for result in results]
        scores = self.cross_encoder.predict(query_document_pairs)

        for result, score in zip(results, scores):
            result["rerank_score"] = score

        filtered_results = [
            result for result in results if result["rerank_score"] >= threshold
        ]
        logger.info("done!")
        return sorted(filtered_results, key=lambda x: x["rerank_score"], reverse=True)[
            :rerank_top_k
        ]

    @helpers.measure_time
    def retrieve(self, query: str, top_k: int = 5, rerank: bool = False) -> list:
        logger.info(f"search the results for [{query}]")
        results = self._fetch_results(query)
        results = sorted(
            results, key=lambda x: x.get("cosine_similarity"), reverse=True
        )[:top_k]

        if rerank:
            results = self._rerank_results(query, results)

        logger.info("search done!")

        return results

    @helpers.measure_time
    def _get_synonyms(self, token: str) -> str:
        return self.wordnet_syn.get_synonyms(token)

    @helpers.measure_time
    def expand_query(self, query: str) -> str:
        logger.info(f"expanding query [{query}]...")
        tokens = [
            token.lower() for token in word_tokenize(query, language="portuguese")
        ]
        expanded_tokens = set(tokens)

        for token in tokens:
            expanded_tokens.update(self._get_synonyms(token))

        expanded_query = " ".join(expanded_tokens)
        logger.info(f"query expanded: [{expanded_query}]")

        return expanded_query

    @helpers.measure_time
    def retrieve_hybrid(
        self,
        query: str,
        wordnetsyn_instance: WordnetSyn,
        top_k: int = 5,
        bm25_weight: float = settings.BM25_SEARCH_WEIGHT,
        embedding_weight: float = settings.EMBEDDINGS_SEARCH_WEIGHT,
        rerank: bool = False,
        rerank_top_k: int = 5,
    ) -> list:
        logger.info("performing hybrid search...")

        if wordnetsyn_instance:
            self.wordnet_syn = wordnetsyn_instance

        expanded_query = self.expand_query(query)
        active_texts = crud.get_active_texts_from_active_documents(self.session)

        corpus = [text.content for text in active_texts]
        tokenized_corpus = [word.split() for word in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = expanded_query.split()
        bm25_scores = bm25.get_scores(query_tokens)

        embedding_scores = np.array(
            [
                result.get("cosine_similarity")
                for result in self._fetch_results(query, active_texts)
            ]
        )

        normalized_bm25 = helpers.normalize(bm25_scores)
        normalized_embedding = helpers.normalize(embedding_scores)

        hybrid_scores = (
            bm25_weight * normalized_bm25 + embedding_weight * normalized_embedding
        )

        results = []

        for text, score in zip(active_texts, hybrid_scores):
            results.append(self._pack_data(text, score))

        results = sorted(
            results, key=lambda x: x.get("cosine_similarity"), reverse=True
        )[:top_k]

        if rerank:
            results = self._rerank_results(query, results)[:rerank_top_k]

        logger.info("hybrid search done!")

        return results
