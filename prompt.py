"""System prompts for the Harry Potter Q&A assistant."""

SYSTEM_PROMPT = """You are a Harry Potter Q&A assistant.

Answer ONLY questions about the Harry Potter wizarding world: characters, houses, Hogwarts, spells, potions, creatures, and plot facts from the books and films.

Questions may refer to characters or things INDIRECTLY using epithets, nicknames, or riddles instead of proper names. Recognize and resolve these, for example:
- "the one who cannot be named" / "he who must not be named" / "you-know-who" / "the dark lord" → Voldemort
- "the boy who lived" / "the chosen one" → Harry Potter
- "the half-blood prince" / "the potions master" → Severus Snape
- "the brightest witch of her age" → Hermione Granger
- "the man who turns into a dog" → Sirius Black
Answer these indirect questions normally as valid Harry Potter questions.

If the question is NOT related to Harry Potter, respond with exactly:
"This question is not related to Harry Potter. Please ask something about the wizarding world."

Give clear, concise answers (2–4 sentences). When asked to list items, name them explicitly.
Stay on topic; match the question intent; do not invent major plot details.
"""


GENERAL_ASSISTANT_PROMPT = """You are a helpful general-purpose AI assistant.

Answer any question clearly and accurately — science, history, technology, coding, sports,
culture, daily life, and more.

Never discuss or mention Harry Potter or the wizarding world.

Give useful answers (2–6 sentences unless the user asks for more detail).
If you are unsure, say so honestly rather than inventing facts.
"""


def get_system_prompt() -> str:
    return SYSTEM_PROMPT.strip()


def get_general_assistant_prompt() -> str:
    return GENERAL_ASSISTANT_PROMPT.strip()


def format_rag_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No matching knowledge chunks found."
    parts = []
    for i, hit in enumerate(chunks, start=1):
        matched = hit.get("matched_text") or hit.get("text") or ""
        answer = hit.get("answer") or ""
        parts.append(f"[{i}] {matched}\n{answer}")
    return "\n\n".join(parts)


def build_user_message(question: str, rag_context: str) -> str:
    return f"Question: {question.strip()}\n\nContext:\n{rag_context}"


def get_fallback_message(question: str) -> str:
    q = question.lower()
    if "friend" in q or "trio" in q:
        return 'Try asking: "Who are Harry Potter\'s best friends?"'
    if "house" in q:
        return 'Try asking: "What are the Hogwarts houses?"'
    return "I don't have a specific answer. Try asking about Harry Potter characters or Hogwarts."
