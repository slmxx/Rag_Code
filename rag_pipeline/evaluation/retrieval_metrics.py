def _is_relevant(path: str, relevant_files: list[str]) -> bool:
    path = path.replace("\\", "/").lower()
    return any(r.replace("\\", "/").lower() in path for r in relevant_files)


def hit_at_k(retrieved_paths: list[str], relevant_files: list[str], k: int) -> float:
    return float(any(_is_relevant(p, relevant_files) for p in retrieved_paths[:k]))


def recall_at_k(retrieved_paths: list[str], relevant_files: list[str], k: int) -> float:
    if not relevant_files:
        return 0.0
    top = retrieved_paths[:k]
    found = sum(
        any(r.replace("\\", "/").lower() in p.replace("\\", "/").lower() for p in top)
        for r in relevant_files
    )
    return found / len(relevant_files)


def mrr(retrieved_paths: list[str], relevant_files: list[str]) -> float:
    for rank, p in enumerate(retrieved_paths, start=1):
        if _is_relevant(p, relevant_files):
            return 1.0 / rank
    return 0.0


def compute_retrieval_metrics(retrieved_paths, relevant_files, k: int) -> dict:
    return {
        f"hit@{k}": hit_at_k(retrieved_paths, relevant_files, k),
        f"recall@{k}": recall_at_k(retrieved_paths, relevant_files, k),
        "mrr": mrr(retrieved_paths, relevant_files),
    }