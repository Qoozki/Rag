import logging
import uuid
from dataclasses import dataclass, field

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.models import Distance, VectorParams

import config
from loaders import chunk_file, discover_files, file_sha256

log = logging.getLogger(__name__)
_NS = uuid.UUID("6f1d2c3a-5b7e-4c1a-9d3e-2a4b6c8d0e1f")


def point_id(source_id: str, file_hash: str, idx: int) -> str:
    return str(uuid.uuid5(_NS, f"{source_id}:{file_hash}:{idx}"))


@dataclass
class IndexedSource:
    source_id: str
    file_name: str = ""
    file_type: str = ""
    file_hash: str = ""
    embed_model: str = ""
    chunks: int = 0
    total_chunks: int = 0

    @property
    def complete(self) -> bool:
        return self.chunks == self.total_chunks


@dataclass
class SyncReport:
    added: list = field(default_factory=list)
    updated: list = field(default_factory=list)
    deleted: list = field(default_factory=list)
    unchanged: list = field(default_factory=list)
    failed: dict = field(default_factory=dict)
    recreated: bool = False

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("added", "updated", "deleted", "unchanged", "failed", "recreated")}

    def summary(self) -> str:
        s = (f"{len(self.added)} added, {len(self.updated)} updated, {len(self.deleted)} deleted, "
             f"{len(self.unchanged)} unchanged")
        return s + (f", {len(self.failed)} FAILED" if self.failed else "")


def _source_filter(source_id: str):
    return qm.Filter(must=[qm.FieldCondition(key="metadata.source_id", match=qm.MatchValue(value=source_id))])


def list_indexed(client, collection: str) -> dict[str, IndexedSource]:
    out: dict[str, IndexedSource] = {}
    if not client.collection_exists(collection):
        return out
    offset = None
    while True:
        records, offset = client.scroll(collection, limit=512, offset=offset, with_payload=True, with_vectors=False)
        for r in records:
            md = (r.payload or {}).get("metadata") or {}
            sid = md.get("source_id")
            if not sid:
                continue
            e = out.setdefault(sid, IndexedSource(sid, md.get("file_name", ""), md.get("file_type", "") or "",
                                                  md.get("file_hash", ""), md.get("embed_model", ""),
                                                  total_chunks=md.get("total_chunks", 0)))
            e.chunks += 1
        if offset is None:
            break
    return out


def delete_source(client, collection: str, source_id: str) -> None:
    if client.collection_exists(collection):
        client.delete(collection, points_selector=qm.FilterSelector(filter=_source_filter(source_id)))


def sync_index(client, vectorstore, *, data_path, collection, embed_model, chunk_size, chunk_overlap):
    """
    Brings the index in line with the files on disk, embedding only what changed:
      new file -> added | changed file (SHA-256) -> old chunks removed, new ones added |
      file removed from disk -> its chunks deleted | unchanged -> skipped (no embedding).
    Files that were only half-indexed (crash) or embedded with another model are re-indexed.
    Returns (chunks_by_id, report). chunks_by_id holds every chunk (for neighbour expansion).
    """
    report = SyncReport()
    chunks_by_id: dict = {}
    indexed = list_indexed(client, collection)
    on_disk = {f.source_id: f for f in discover_files(data_path)}

    for sid in sorted(indexed.keys() - on_disk.keys()):
        delete_source(client, collection, sid)
        report.deleted.append(sid)
        log.info("Removed from index (file deleted): %s", sid)

    for sid, sf in on_disk.items():
        cur = indexed.get(sid)
        try:
            digest = file_sha256(sf.path)
            chunks = chunk_file(sf, digest, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                embed_model=embed_model)
        except Exception as exc:
            log.exception("Could not read %s", sid)
            report.failed[sid] = f"{type(exc).__name__}: {exc}"
            if cur:                                   # keep index and memory consistent
                delete_source(client, collection, sid)
            continue

        for c in chunks:
            chunks_by_id[c.metadata["chunk_id"]] = c
        if cur and cur.file_hash == digest and cur.complete and cur.embed_model == embed_model:
            report.unchanged.append(sid)
            continue
        try:
            if cur:
                delete_source(client, collection, sid)
            vectorstore.add_documents(chunks, ids=[point_id(sid, digest, c.metadata["chunk_index"]) for c in chunks])
            (report.updated if cur else report.added).append(sid)
            log.info("Indexed %s (%d chunks, %s)", sid, len(chunks), "updated" if cur else "new")
        except Exception as exc:
            log.exception("Failed to index %s", sid)
            report.failed[sid] = f"{type(exc).__name__}: {exc}"
            for c in chunks:
                chunks_by_id.pop(c.metadata["chunk_id"], None)
            try:
                delete_source(client, collection, sid)
            except Exception:
                log.exception("Cleanup after failed indexing of %s failed", sid)
    log.info("Index sync finished: %s", report.summary())
    return chunks_by_id, report


def initialize_vector_store(force_recreate: bool = False):
    """
    Opens Qdrant and syncs it with the documents folder.
    Recreates the collection if forced, if it doesn't exist, or if the embedding dimension changed
    (e.g. after switching embedding models). Returns (client, vectorstore, chunks_by_id, report).
    """
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)
    embed_dim = len(embeddings.embed_query("dimension probe"))

    client = QdrantClient(path=config.QDRANT_PATH)
    try:
        exists = client.collection_exists(config.COLLECTION_NAME)
        recreated = False
        if exists and not force_recreate:
            vectors_cfg = client.get_collection(config.COLLECTION_NAME).config.params.vectors
            existing_size = getattr(vectors_cfg, "size", None)
            if existing_size != embed_dim:
                log.warning("Embedding size changed (%s -> %s); recreating collection...", existing_size, embed_dim)
                force_recreate = True
        if force_recreate and exists:
            log.info("Force-recreating collection '%s'...", config.COLLECTION_NAME)
            client.delete_collection(config.COLLECTION_NAME)
            exists, recreated = False, True
        if not exists:
            client.create_collection(config.COLLECTION_NAME,
                                     vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE))
            recreated = True
        vectorstore = QdrantVectorStore(client=client, collection_name=config.COLLECTION_NAME, embedding=embeddings)
        chunks_by_id, report = sync_index(
            client, vectorstore, data_path=config.DATA_PATH, collection=config.COLLECTION_NAME,
            embed_model=config.EMBED_MODEL, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        report.recreated = recreated
    except Exception:
        client.close()      # release the Qdrant folder lock so a retry can reopen it
        raise
    return client, vectorstore, chunks_by_id, report
