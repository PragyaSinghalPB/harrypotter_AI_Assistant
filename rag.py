"""
RAG knowledge base for Harry Potter Q&A.

FAQ data is stored inline — no external files required.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

from prompt import build_user_message, format_rag_context, get_system_prompt

TOKEN_RE = re.compile(r"[a-z0-9']+")

# ---------------------------------------------------------------------------
# Inline FAQ data (previously faq.txt / knowledge_chunks.json)
# ---------------------------------------------------------------------------

_FAQ_BLOCKS: list[dict] = [
    {
        "keywords": ["best friends", "harry potter friends", "harry's friends", "three friends", "who are harry's friends"],
        "questions": ["Who are Harry Potter's best friends?", "Name three friends of Harry Potter."],
        "answer": "Harry Potter's closest friends are Ron Weasley and Hermione Granger. Neville Longbottom, Ginny Weasley, Luna Lovegood, and Hagrid are also important friends.",
    },
    {
        "keywords": ["golden trio", "harry ron hermione trio"],
        "questions": ["Who is the golden trio?"],
        "answer": "The Golden Trio refers to Harry Potter, Ron Weasley, and Hermione Granger — the three Gryffindor friends at the heart of the series.",
    },
    {
        "keywords": ["harry potter", "boy who lived", "chosen one"],
        "questions": ["Who is Harry Potter?"],
        "answer": "Harry James Potter is the Boy Who Lived, son of James and Lily Potter. He is a Gryffindor and famous for defeating Lord Voldemort.",
    },
    {
        "keywords": ["hermione", "hermione granger"],
        "questions": ["Who is Hermione Granger?"],
        "answer": "Hermione Jean Granger is Harry's best friend, a brilliant Gryffindor witch known for her intelligence and loyalty.",
    },
    {
        "keywords": ["ron", "ron weasley", "weasley"],
        "questions": ["Who is Ron Weasley?"],
        "answer": "Ronald Bilius Weasley is Harry's best friend, a loyal Gryffindor from the Weasley family.",
    },
    {
        "keywords": ["voldemort", "tom riddle", "dark lord"],
        "questions": ["Who is Voldemort?"],
        "answer": "Lord Voldemort (Tom Marvolo Riddle) is the dark wizard who killed Harry's parents and was defeated at the Battle of Hogwarts.",
    },
    {
        "keywords": ["dumbledore", "albus dumbledore"],
        "questions": ["Who is Dumbledore?"],
        "answer": "Albus Dumbledore was Headmaster of Hogwarts, a powerful wizard, and leader of the Order of the Phoenix.",
    },
    {
        "keywords": ["snape", "severus snape"],
        "questions": ["Who is Snape?"],
        "answer": "Severus Snape was Potions Master at Hogwarts, Head of Slytherin House, and played a pivotal role in Harry's life.",
    },
    {
        "keywords": ["draco", "draco malfoy", "malfoy"],
        "questions": ["Who is Draco Malfoy?"],
        "answer": "Draco Malfoy is a Slytherin student at Hogwarts, son of Lucius and Narcissa Malfoy, and often rivals Harry Potter.",
    },
    {
        "keywords": ["luna", "luna lovegood"],
        "questions": ["Who is Luna Lovegood?"],
        "answer": "Luna Lovegood is a Ravenclaw student known for her unique beliefs, kindness, and friendship with Harry, Ron, and Hermione.",
    },
    {
        "keywords": ["ginny", "ginny weasley"],
        "questions": ["Who is Ginny Weasley?"],
        "answer": "Ginny Weasley is the youngest Weasley daughter, a talented Gryffindor witch and Quidditch player, and later Harry Potter's wife.",
    },
    {
        "keywords": ["hagrid", "rubeus hagrid"],
        "questions": ["Who is Hagrid?"],
        "answer": "Rubeus Hagrid is Hogwarts' Keeper of Keys and Grounds, Care of Magical Creatures teacher, and one of Harry's first friends.",
    },
    {
        "keywords": ["sirius", "sirius black", "godfather", "god father", "harry godfather", "harry's godfather", "god father of harry potter"],
        "questions": ["Who is Harry Potter's godfather?", "Who is Sirius Black?"],
        "answer": "Sirius Black was Harry Potter's godfather, an Animagus who could turn into a black dog, and a member of the Order of the Phoenix.",
    },
    {
        "keywords": ["mcgonagall", "minerva mcgonagall", "professor mcgonagall"],
        "questions": ["Who is Professor McGonagall?"],
        "answer": "Professor Minerva McGonagall teaches Transfiguration, is Head of Gryffindor House, and later became Headmistress of Hogwarts. She is an Animagus who can transform into a tabby cat.",
    },
    {
        "keywords": ["hogwarts", "houses", "gryffindor", "hufflepuff", "ravenclaw", "slytherin"],
        "questions": ["What are the Hogwarts houses?"],
        "answer": "Hogwarts School of Witchcraft and Wizardry has four houses: Gryffindor, Hufflepuff, Ravenclaw, and Slytherin.",
    },
]


# ---------------------------------------------------------------------------
# Data types & helpers
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    id: str
    block_id: str
    chunk_type: str  # "question" | "block"
    text: str
    answer: str
    keywords: list[str]
    search_text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def _normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# Chunk building (from inline data)
# ---------------------------------------------------------------------------

def build_chunks_from_faq() -> list[Chunk]:
    """Build retrieval chunks from the inline FAQ data."""
    chunks: list[Chunk] = []
    for i, block in enumerate(_FAQ_BLOCKS):
        block_id = f"block_{i}"
        keywords = block["keywords"]
        answer = block["answer"]
        questions = block["questions"]
        keyword_blob = " ".join(keywords)

        block_search = _normalize(f"{keyword_blob} {' '.join(questions)} {answer}")
        chunks.append(
            Chunk(
                id=f"{block_id}_overview",
                block_id=block_id,
                chunk_type="block",
                text=questions[0] if questions else keyword_blob,
                answer=answer,
                keywords=keywords,
                search_text=block_search,
            )
        )

        for j, question in enumerate(questions):
            q_search = _normalize(f"{question} {keyword_blob} {answer}")
            chunks.append(
                Chunk(
                    id=f"{block_id}_q{j}",
                    block_id=block_id,
                    chunk_type="question",
                    text=question,
                    answer=answer,
                    keywords=keywords,
                    search_text=q_search,
                )
            )

    return chunks


def keyword_answer_map() -> dict[str, str]:
    """Keyword -> answer index (for fast FAQ lookup)."""
    return {kw.lower(): block["answer"] for block in _FAQ_BLOCKS for kw in block["keywords"]}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_chunk(query: str, chunk: Chunk) -> float:
    q_norm = _normalize(query)
    q_tokens = _tokenize(query)
    search_tokens = _tokenize(chunk.search_text)

    jaccard = _jaccard(q_tokens, search_tokens)
    seq = SequenceMatcher(None, q_norm, chunk.search_text).ratio()

    if chunk.chunk_type == "question":
        seq = max(seq, SequenceMatcher(None, q_norm, _normalize(chunk.text)).ratio())

    keyword_bonus = 0.0
    q_lower = query.lower()
    for kw in chunk.keywords:
        if kw in q_lower:
            keyword_bonus = max(keyword_bonus, min(0.35, len(kw) / 40))

    return min(1.0, (0.45 * jaccard) + (0.45 * seq) + keyword_bonus)


# ---------------------------------------------------------------------------
# RAGStore
# ---------------------------------------------------------------------------

class RAGStore:
    """In-memory index over built FAQ chunks."""

    def __init__(self, chunks: list[Chunk] | None = None, min_score: float = 0.22):
        self.chunks = chunks if chunks is not None else build_chunks_from_faq()
        self.min_score = min_score

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not query or not query.strip():
            return []

        scored: list[tuple[float, Chunk]] = [
            (score_chunk(query, chunk), chunk)
            for chunk in self.chunks
        ]
        scored = [(s, c) for s, c in scored if s > 0]
        scored.sort(key=lambda x: x[0], reverse=True)

        seen_answers: set[str] = set()
        results: list[dict] = []

        for score, chunk in scored[: top_k * 2]:
            if chunk.answer in seen_answers:
                continue
            seen_answers.add(chunk.answer)
            results.append(
                {
                    "score": round(score, 4),
                    "chunk_id": chunk.id,
                    "block_id": chunk.block_id,
                    "type": chunk.chunk_type,
                    "matched_text": chunk.text,
                    "answer": chunk.answer,
                    "keywords": chunk.keywords,
                }
            )
            if len(results) >= top_k:
                break

        return results

    def answer(self, query: str) -> str | None:
        hits = self.retrieve(query, top_k=1)
        if not hits or hits[0]["score"] < self.min_score:
            return None
        return hits[0]["answer"]

    def get_prompt_context(self, query: str, top_k: int = 3) -> dict:
        """Bundle system prompt + retrieved chunks for the ask pipeline."""
        hits = self.retrieve(query, top_k=top_k)
        context_text = format_rag_context(hits)
        return {
            "system_prompt": get_system_prompt(),
            "rag_context": context_text,
            "chunks": hits,
            "user_message": build_user_message(query, context_text),
        }

    def similar_questions(self, query: str, top_k: int = 5) -> list[dict]:
        """Return related sample questions from the knowledge base."""
        q_norm = _normalize(query)
        question_chunks = [c for c in self.chunks if c.chunk_type == "question"]
        ranked: list[tuple[float, Chunk]] = []

        for chunk in question_chunks:
            if _normalize(chunk.text) == q_norm:
                continue
            score = SequenceMatcher(None, q_norm, _normalize(chunk.text)).ratio()
            score = max(score, score_chunk(query, chunk) * 0.9)
            ranked.append((score, chunk))

        ranked.sort(key=lambda x: x[0], reverse=True)

        seen: set[str] = set()
        related: list[dict] = []
        for score, chunk in ranked:
            text = chunk.text.strip()
            if text.lower() in seen:
                continue
            seen.add(text.lower())
            related.append(
                {
                    "question": text,
                    "score": round(score, 4),
                    "answer_preview": chunk.answer[:120] + ("..." if len(chunk.answer) > 120 else ""),
                }
            )
            if len(related) >= top_k:
                break

        return related


# ---------------------------------------------------------------------------
# Singleton & rebuild
# ---------------------------------------------------------------------------

_store: RAGStore | None = None


def get_rag_store() -> RAGStore:
    global _store
    if _store is None:
        _store = RAGStore()
    return _store


def rebuild_knowledge_base() -> int:
    """Rebuild the in-memory store from inline FAQ data."""
    global _store
    chunks = build_chunks_from_faq()
    _store = RAGStore(chunks)
    return len(chunks)
