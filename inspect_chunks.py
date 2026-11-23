"""Montre comment les chunks sont stockés dans ChromaDB (id, texte, métadonnées, vecteur), en JSON.

Usage  :
    python inspect_chunks.py                                   # 3 premiers chunks de l'index Voyage
    python inspect_chunks.py --n 5
    python inspect_chunks.py --path auth-service/src/main/resources/application.yaml
    python inspect_chunks.py --embedder gemini_embedding

"""
import argparse
import json

from rag_pipeline.store import get_collection

ap = argparse.ArgumentParser()
ap.add_argument("--embedder", default="voyage_code_4", choices=["voyage_code_4", "gemini_embedding"])
ap.add_argument("--n", type=int, default=3, help="nombre de chunks affichés")
ap.add_argument("--path", default=None, help="chemin exact d'un fichier indexé")
args = ap.parse_args()

col = get_collection(args.embedder)
kwargs = {"include": ["documents", "metadatas", "embeddings"]}
if args.path:
    kwargs["where"] = {"path": args.path}
else:
    kwargs["limit"] = args.n
res = col.get(**kwargs)

items = []
for i, cid in enumerate(res["ids"]):
    vec = res["embeddings"][i]
    items.append({
        "id": cid,                                            # identifiant : chemin::numéro::hash du contenu
        "metadata": res["metadatas"][i],                      # infos de contexte (fichier, lignes, ...)
        "document": res["documents"][i],                      # texte stocké = en-tête + code, tel qu'envoyé à Mistral
        "embedding": {"dimension": len(vec), "premieres_valeurs": [round(float(x), 4) for x in vec[:5]]},
    })

print(f"Collection : {col.name} | {col.count()} chunks au total | {len(items)} affichés\n")
print(json.dumps(items, ensure_ascii=False, indent=2))
with open("chunks_sample.json", "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)
print("\nFichier créé : chunks_sample.json")