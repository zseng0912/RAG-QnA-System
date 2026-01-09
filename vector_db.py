from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=3072):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        # Sentence-level index collection for window retrieval and entity reasoning
        self.sent_collection = "sentences"
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        if not self.client.collection_exists(self.sent_collection):
            self.client.create_collection(
                collection_name=self.sent_collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        # Document/chunk-level upsert into the vector index
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def upsert_sentences(self, ids, vectors, payloads):
        # Sentence-level upsert into the sentence-window index with metadata (entities, indices)
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.sent_collection, points=points)

    def search(self, query_vector, top_k: int = 5):
        # Simple vector search over document/chunk collection. Returns payload texts and unique sources
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        )

        contexts = []
        sources = set()

        for r in results.points:
            payload = r.payload or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {
            "contexts": contexts,
            "sources": list(sources),
        }

    def search_docs_detailed(self, query_vector, top_k: int = 5):
        # Vector search returning detailed items including per-point score
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        )
        items = []
        for r in results.points:
            payload = r.payload or {}
            items.append({
                "id": r.id,
                "text": payload.get("text", ""),
                "source": payload.get("source", ""),
                "score": getattr(r, "score", None),
            })
        return items

    def search_sentences_detailed(self, query_vector, top_k: int = 10):
        # Sentence-level vector search; returns sentence metadata for window assembly and entity overlap checks
        results = self.client.query_points(
            collection_name=self.sent_collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        )
        items = []
        for r in results.points:
            payload = r.payload or {}
            items.append({
                "id": r.id,
                "text": payload.get("text", ""),
                "source": payload.get("source", ""),
                "chunk_index": payload.get("chunk_index", 0),
                "sentence_index": payload.get("sentence_index", 0),
                "entities": payload.get("entities", []),
                "score": getattr(r, "score", None),
            })
        return items

    def retrieve_sentences_by_ids(self, ids):
        # Fetch sentence payloads by IDs to build surrounding context windows
        recs = self.client.retrieve(self.sent_collection, ids=ids)
        out = {}
        for r in recs:
            out[r.id] = r.payload or {}
        return out



