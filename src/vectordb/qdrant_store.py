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

    def _search_collection(
        self,
        collection_name: str,
        query_embedding: list,
        top_k: int,
        search_filter: Filter | None,
        score_threshold: float,
    ) -> list:
        try:
            if hasattr(self.client, "search"):
                hits = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    query_filter=search_filter,
                    score_threshold=score_threshold,
                )
            else:
                response = self.client.query_points(
                    collection_name=collection_name,
                    query=query_embedding,
                    limit=top_k,
                    query_filter=search_filter,
                    score_threshold=score_threshold,
                )
                hits = response.points
            return [
                {
                    "text": h.payload.get("text", ""),
                    "metadata": {k: v for k, v in (h.payload or {}).items() if k != "text"},
                    "score": h.score or 0.0,
                }
                for h in hits
            ]
        except Exception:
            return []

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

        # Dual-index: search main + section + rule collections
        collections_to_search = [self.collection_name]
        section_col = f"{self.collection_name}_sections"
        rule_col = f"{self.collection_name}_rules"
        colls = [c.name for c in self.client.get_collections().collections]
        if section_col in colls:
            collections_to_search.append(section_col)
        if rule_col in colls:
            collections_to_search.append(rule_col)

        all_results = []
        per_collection = max(top_k, top_k * 2)
        for coll in collections_to_search:
            hits = self._search_collection(
                coll, query_embedding, per_collection, search_filter, score_threshold
            )
            all_results.extend(hits)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    def list_documents(self) -> list[dict]:
        collections_to_scan = [self.collection_name]
        section_col = f"{self.collection_name}_sections"
        rule_col = f"{self.collection_name}_rules"
        colls = [c.name for c in self.client.get_collections().collections]
        if section_col in colls:
            collections_to_scan.append(section_col)
        if rule_col in colls:
            collections_to_scan.append(rule_col)
        docs = {}
        for coll in collections_to_scan:
            all_points, _ = self.client.scroll(
                collection_name=coll,
                limit=10000,
                with_payload=["document_id", "document_name", "document_type"],
            )
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
        filter_sel = models.FilterSelector(
            filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
        )
        collections_to_clear = [self.collection_name]
        section_col = f"{self.collection_name}_sections"
        rule_col = f"{self.collection_name}_rules"
        colls = [c.name for c in self.client.get_collections().collections]
        if section_col in colls:
            collections_to_clear.append(section_col)
        if rule_col in colls:
            collections_to_clear.append(rule_col)
        for coll in collections_to_clear:
            self.client.delete(collection_name=coll, points_selector=filter_sel)
        logger.info(f"Deleted all chunks for document: {document_id}")

    def get_document_chunks(self, document_id: str, limit: int = 5) -> list[dict]:
        filter_ = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
        collections_to_scan = [self.collection_name]
        section_col = f"{self.collection_name}_sections"
        rule_col = f"{self.collection_name}_rules"
        colls = [c.name for c in self.client.get_collections().collections]
        if section_col in colls:
            collections_to_scan.append(section_col)
        if rule_col in colls:
            collections_to_scan.append(rule_col)
        rows = []
        per_coll = max(1, limit // len(collections_to_scan))
        for coll in collections_to_scan:
            pts, _ = self.client.scroll(
                collection_name=coll,
                limit=per_coll,
                with_payload=True,
                with_vectors=False,
                scroll_filter=filter_,
            )
            for p in pts:
                payload = p.payload or {}
                rows.append(
                    {
                        "id": str(p.id),
                        "text": payload.get("text", ""),
                        "metadata": {k: v for k, v in payload.items() if k != "text"},
                    }
                )
            if len(rows) >= limit:
                break
        return rows[:limit]

    # ─── Dual-index support (section + rule collections) ───

    def create_section_index(self) -> str:
        section_collection = f"{self.collection_name}_sections"
        collections = self.client.get_collections().collections
        exists = any(c.name == section_collection for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=section_collection,
                vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
            )
            for field_name in ["document_type", "document_id", "clause_id", "content_type"]:
                self.client.create_payload_index(
                    collection_name=section_collection,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            logger.info(f"Created section index: {section_collection}")
        return section_collection

    def create_rule_index(self) -> str:
        rule_collection = f"{self.collection_name}_rules"
        collections = self.client.get_collections().collections
        exists = any(c.name == rule_collection for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=rule_collection,
                vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
            )
            for field_name in ["document_type", "document_id", "clause_id", "content_type", "regulatory_strength"]:
                self.client.create_payload_index(
                    collection_name=rule_collection,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            logger.info(f"Created rule index: {rule_collection}")
        return rule_collection

    def get_section_store(self) -> "GeoVectorStore":
        section_store = GeoVectorStore.__new__(GeoVectorStore)
        section_store.client = self.client
        section_store.collection_name = f"{self.collection_name}_sections"
        section_store._ensure_collection = lambda: None
        return section_store

    def get_rule_store(self) -> "GeoVectorStore":
        rule_store = GeoVectorStore.__new__(GeoVectorStore)
        rule_store.client = self.client
        rule_store.collection_name = f"{self.collection_name}_rules"
        rule_store._ensure_collection = lambda: None
        return rule_store
