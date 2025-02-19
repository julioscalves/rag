import faiss
import heapq
import networkx as nx
import numpy as np

from operator import itemgetter
from sqlalchemy.orm import Session
from sklearn.metrics.pairwise import cosine_similarity

import settings

from services import embeddings
from models import crud
from utils import helpers
from utils.logging import logger


class FAISSIndex:
    def __init__(
        self,
        session: Session,
        embedder: embeddings.Embeddings,
        dimension: int = settings.FAISS_DIMENSION,
    ):
        self.session = session
        self.embedder = embedder
        self.dimension = dimension
        self.index = None

    @helpers.measure_time
    def build_index(self):
        logger.info("building FAISS index...")
        texts = crud.get_texts_from_active_documents(self.session)
        embeddings = []
        document_ids = []

        for text in texts:
            embedding = np.frombuffer(text.embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)

            if norm > 1e-6:
                embedding = embedding / norm

            embeddings.append(embedding)
            document_ids.append(text.id)

        embeddings = np.vstack(embeddings).astype(np.float32)

        index_flat = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap(index_flat)
        id_array = np.array(document_ids, dtype=np.int64)
        self.index.add_with_ids(embeddings, id_array)

        logger.info(f"FAISS index built: {self.index.ntotal} vectors")

    @helpers.measure_time
    def _rerank(
        self, query: str, results: list, rerank_top_k: int = 5, threshold: float = -5.0
    ) -> list:
        logger.info("reranking results...")
        filtered_results = []

        query_document_pairs = [(query, result["content"]) for result in results]
        scores = self.embedder.cross_encoder.predict(query_document_pairs)

        for result, score in zip(results, scores):
            result["rerank_score"] = score

            if score >= threshold:
                filtered_results.append(result)

        final_results = heapq.nlargest(
            rerank_top_k, filtered_results, key=itemgetter("rerank_score")
        )
        logger.info("done!")

        return final_results

    @helpers.measure_time
    def search(
        self, query: str, top_k: int = 5, rerank: bool = False, rerank_top_k: int = 5
    ) -> list:
        logger.info(f"search for [{query}] via FAISS...")

        query_embedding = self.embedder.model.encode(query)
        norm = np.linalg.norm(query_embedding)

        if norm > 1e-6:
            query_embedding = query_embedding / norm

        query_embedding = np.expand_dims(query_embedding, axis=0).astype(np.float32)
        distances, retrieved_ids = self.index.search(query_embedding, top_k)

        valid_ids = [
            int(document_id) for document_id in retrieved_ids[0] if document_id != -1
        ]
        text_objects = {
            text.id: text for text in crud.get_texts_in_id_list(self.session, valid_ids)
        }

        results = []

        for document_id, score in zip(retrieved_ids[0], distances[0]):
            if document_id == -1:
                continue

            text_object = text_objects.get(document_id)

            if text_object:
                results.append(self.embedder._pack_data(text_object, float(score)))

        if rerank:
            results = self._rerank(query, results, rerank_top_k)

        logger.info("done!")

        return results


class Graph:
    def __init__(self, session: Session, embedder: embeddings.Embeddings):
        self.session = session
        self.embedder = embedder
        self.graph = nx.Graph()

    @helpers.measure_time
    def build_graph_network(self, similarity_threshold=0.8):
        logger.info("assembling graph network...")
        active_texts = crud.get_texts_from_active_documents(self.session)

        if not active_texts:
            logger.warning("no texts available to build the graph network!")
            return

        node_ids = []
        embeddings = []

        for text in active_texts:
            embedding = np.frombuffer(text.embedding, dtype=np.float32)
            embeddings.append(embedding)
            node_ids.append(text.id)
            self.graph.add_node(
                text.id,
                context=text.content,
                document_id=text.document.id,
                filename=text.document.filename,
                name=text.document.name,
            )

        embeddings_array = np.stack(embeddings)
        similarity_matrix = cosine_similarity(embeddings_array)

        i_upper, j_upper = np.triu_indices(len(node_ids), k=1)
        mask = similarity_matrix[i_upper, j_upper] >= similarity_threshold

        for i, j in zip(i_upper[mask], j_upper[mask]):
            self.graph.add_edge(
                node_ids[i], node_ids[j], weight=similarity_matrix[i, j]
            )

        logger.info(
            f"graph built with [{self.graph.number_of_nodes()}] nodes and [{self.graph.number_of_edges()}] edges!"
        )

    @helpers.measure_time
    def retrieve(self, query: str, top_k: int = 5, graph_expansion_steps: int = 1):
        logger.info("retrieving information for the graph network...")
        query_embedding = self.embedder.model.encode(query)
        active_texts = crud.get_texts_from_active_documents(self.session)
        results = []

        for text in active_texts:
            embedding = np.frombuffer(text.embedding, dtype=np.float32)
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            results.append((text.id, similarity))

        seed_nodes = [
            node_id
            for node_id, similarity in sorted(
                results, key=lambda x: x[1], reverse=True
            )[:top_k]
        ]
        logger.info(f"seed nodes from direct retrieval: {seed_nodes}")
        expanded_nodes = set(seed_nodes)

        for _ in range(graph_expansion_steps):
            neighbors = set()

            for node in list(expanded_nodes):
                neighbors.update(self.graph.neighbors(node))

            expanded_nodes.update(neighbors)

        final_results = []

        for node_id in expanded_nodes:
            text_object = crud.get_text_by_id(self.session, node_id)

            if text_object:
                embedding = np.frombuffer(text_object.embedding, dtype=np.float32)
                similarity = cosine_similarity([query_embedding], [embedding])[0][0]
                final_results.append(self.embedder._pack_data(text_object, similarity))

        final_results = sorted(
            final_results, key=lambda x: x["cosine_similarity"], reverse=True
        )[:top_k]

        return final_results
