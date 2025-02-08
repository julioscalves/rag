import faiss
import networkx as nx
import numpy as np

from sqlalchemy.orm import Session
from sklearn.metrics.pairwise import cosine_similarity

from services import embeddings
from models import crud
from utils.logging import logger


class FAISSIndex:
    def __init__(
        self, session: Session, embedder: embeddings.Embeddings, dimension: int
    ):
        self.session = session
        self.embedder = embedder
        self.index = None
        self.id_mapping = []
        self.dimension = dimension

    def build_index(self):
        logger.info(f"building FAISS index...")
        texts = crud.get_texts_from_active_documents(self.session)
        embeddings = []
        self.id_mapping = []

        for text in texts:
            embedding = np.frombuffer(text.embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)

            if norm > 0:
                embedding = embedding / norm

            embeddings.append(embedding)
            self.id_mapping.append(text.id)

        embeddings = np.stack(embeddings).astype(np.float32)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        logger.info(f"FAISS index built: {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = 5) -> list:
        query_embedding = self.embedder.embed(query)
        norm = np.linalg.norm(query_embedding)

        if norm > 0:
            query_embedding = query_embedding / norm

        query_embedding = np.expand_dims(query_embedding, axis=0).astype(np.float32)
        distances, indices = self.index.search(query_embedding, top_k)
        results = []

        for index, score in zip(indices[0], distances[0]):

            if index == -1:
                continue

            text_id = self.id_mapping[index]
            text_object = crud.get_text_by_id(self.session, text_id)

            if text_object:
                results.append(self.embedder._pack_data(text_object, float(score)))

        return results


class Graph:
    def __init__(self, session: Session, embedder: embeddings.Embeddings):
        self.session = session
        self.embedder = embedder(session)
        self.graph = nx.Graph()

    def build_graph_index(self, similarity_threshold=0.8):
        logger.info("assembling graph network...")
        active_texts = crud.get_texts_from_active_documents(self.session)
        embeddings = []
        node_ids = []

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

        if not embeddings:
            logger.warning("no texts available to build the graph network!")

        embeddings_array = np.stack(embeddings)
        similarity_matrix = cosine_similarity(embeddings_array)
        num_texts = len(node_ids)

        for i in range(num_texts):
            for j in range(i + 1, num_texts):
                similarity = similarity_matrix[i, j]
                if similarity >= similarity_threshold:
                    self.graph.add_edge(node_ids[i], node_ids[j], weight=similarity)

        logger.info(
            f"graph built with [{self.graph.number_of_nodes()}] nodes and [{self.graph.number_of_edges()}] edges!"
        )

    def retrieve(self, query: str, top_k: int = 5, graph_expansion_steps: int = 1):
        logger.info("retrieving information for the graph network...")
        query_embedding = self.embedder.embed(query)
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
