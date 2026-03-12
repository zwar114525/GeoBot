import uuid
import builtins
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, models
from loguru import logger
from config.settings import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_DIMENSION
from src.utils.embeddings import embed_texts, embed_query

_LOCAL_CLIENT = None
_LOCAL_CLIENT_KEY = "_GEOBOT_QDRANT_LOCAL_CLIENT"


def _get_local_client():
    return getattr(builtins, _LOCAL_CLIENT_KEY, None)


def _set_local_client(client):
    setattr(builtins, _LOCAL_CLIENT_KEY, client)


class GeoVectorStore:
    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        collection_name: str = COLLECTION_NAME,
        use_local: bool = False,
    ):
        global _LOCAL_CLIENT
        if use_local:
            if _LOCAL_CLIENT is None:
                _LOCAL_CLIENT = _get_local_client()
            if _LOCAL_CLIENT is None:
                _LOCAL_CLIENT = QdrantClient(path="./qdrant_local_storage")
                _set_local_client(_LOCAL_CLIENT)
                logger.info("Using local file-based Qdrant storage")
            else:
                logger.info("Reusing local file-based Qdrant client")
            self.client = _LOCAL_CLIENT
        else:
            self.client = QdrantClient(host=host, port=port)
            logger.info(f"Connected to Qdrant at {host}:{port}")
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
            )
            for field_name in [
                "document_type",
                "document_id",
                "clause_id",
                "content_type",
                "regulatory_strength",
            ]:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            logger.info(f"Created collection: {self.collection_name}")
        else:
            logger.info(f"Collection already exists: {self.collection_name}")

    def add_document(self, processed_doc) -> int:
        if not processed_doc.chunks:
            logger.warning(f"No chunks to add for {processed_doc.document_name}")
            return 0
        texts = [chunk.text for chunk in processed_doc.chunks]
        embeddings = embed_texts(texts)
        points = []
        for chunk, embedding in zip(processed_doc.chunks, embeddings):
            points.append(PointStruct(id=str(uuid.uuid4()), vector=embedding, payload={"text": chunk.text, **chunk.metadata}))
        for i in range(0, len(points), 100):
            self.client.upsert(collection_name=self.collection_name, points=points[i : i + 100])
        logger.info(f"Added {len(points)} chunks from '{processed_doc.document_name}'")
        return len(points)

    def search(
        self,
        query: str,
        top_k: int = 10,
        document_type: str | None = None,
        document_id: str | None = None,
        clause_id: str | None = None,
        content_type: str | None = None,
        regulatory_strength: str | None = None,
        score_threshold: float = 0.3,
    ) -> list[dict]:
        query_embedding = embed_query(query)
        filter_conditions = []
        if document_type:
            filter_conditions.append(FieldCondition(key="document_type", match=MatchValue(value=document_type)))
        if document_id:
            filter_conditions.append(FieldCondition(key="document_id", match=MatchValue(value=document_id)))
        if clause_id:
            filter_conditions.append(FieldCondition(key="clause_id", match=MatchValue(value=clause_id)))
        if content_type:
            filter_conditions.append(FieldCondition(key="content_type", match=MatchValue(value=content_type)))
        if regulatory_strength:
            filter_conditions.append(
                FieldCondition(key="regulatory_strength", match=MatchValue(value=regulatory_strength))
            )
        search_filter = Filter(must=filter_conditions) if filter_conditions else None
        if hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=search_filter,
                score_threshold=score_threshold,
            )
        else:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                query_filter=search_filter,
                score_threshold=score_threshold,
            )
            results = response.points
        return [
            {"text": hit.payload.get("text", ""), "metadata": {k: v for k, v in hit.payload.items() if k != "text"}, "score": hit.score}
            for hit in results
        ]

    def list_documents(self) -> list[dict]:
        all_points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=["document_id", "document_name", "document_type"],
        )
        docs = {}
        for point in all_points:
            doc_id = point.payload.get("document_id", "unknown")
            if doc_id not in docs:
                docs[doc_id] = {
                    "document_id": doc_id,
                    "document_name": point.payload.get("document_name", ""),
                    "document_type": point.payload.get("document_type", ""),
                    "chunk_count": 0,
                }
            docs[doc_id]["chunk_count"] += 1
        return list(docs.values())

    def delete_document(self, document_id: str):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
            ),
        )
        logger.info(f"Deleted all chunks for document: {document_id}")

    def get_document_chunks(self, document_id: str, limit: int = 5) -> list[dict]:
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]),
        )
        rows = []
        for p in points:
            payload = p.payload or {}
            rows.append(
                {
                    "id": str(p.id),
                    "text": payload.get("text", ""),
                    "metadata": {k: v for k, v in payload.items() if k != "text"},
                }
            )
        return rows
