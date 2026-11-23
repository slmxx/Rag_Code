from rag_pipeline.store import get_collection

col = get_collection()
r = col.get(where={"path": "CLAUDE.md"}, include=["documents", "metadatas"])
for doc, meta in zip(r["documents"], r["metadatas"]):
    if "mvnw" in doc:
        print(meta)
        print(doc)
        print("=" * 60)