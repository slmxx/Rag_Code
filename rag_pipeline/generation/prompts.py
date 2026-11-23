SYSTEM_PROMPT = (
    "Tu es un assistant expert du code de la plateforme e-commerce "
    "(microservices Spring et frontend Angular). "
    "Réponds en français, UNIQUEMENT à partir des extraits fournis dans le contexte. "
    "N'utilise aucune connaissance extérieure et n'invente rien. "
    "Si les extraits ne contiennent pas la réponse, réponds exactement : "
    "« Le contexte fourni ne contient pas cette information. » et n'ajoute rien d'autre. "
    "S'ils n'en contiennent qu'une partie, donne cette partie et précise ce qui manque. "
    "Cite le chemin des fichiers sur lesquels tu t'appuies. "
    "Sois concis et précis. Si l'utilisateur demande une fonction, un bout de code ou une implémentation précise, "
    "recopie le code pertinent présent dans le contexte dans un bloc de code Markdown, "
    "sans rien y ajouter qui ne figure pas dans le contexte."
)


def build_prompt(question: str, contexts: list[str]) -> str:
    # Extraits numérotés : chaque chunk contient déjà un « --- » entre son en-tête et son code,
    # un séparateur « --- » entre chunks serait donc ambigu.
    ctx = "\n\n".join(f"=== Extrait {i} ===\n{c}" for i, c in enumerate(contexts, 1))
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"### Contexte\n{ctx}\n\n"
        f"### Question\n{question}\n\n"
        f"### Réponse"
    )