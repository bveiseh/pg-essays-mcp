"""Read-only search over the currently active Paul Graham essay index."""

from __future__ import annotations

import os
import pathlib
import sqlite3
import json
import time
from collections import defaultdict
from contextlib import closing
from functools import lru_cache
from typing import Literal

import numpy as np
import regex as re
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

ROOT = pathlib.Path(__file__).parent
DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR", ROOT / "data"))
CURRENT = DATA_ROOT / "current"
_embedding_function = DefaultEmbeddingFunction()
_vector_cache: tuple[pathlib.Path, np.ndarray, list[dict[str, object]]] | None = None


def _connect() -> sqlite3.Connection:
    db_path = CURRENT / "index.sqlite3"
    if not db_path.exists():
        raise RuntimeError("index is not ready")
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def status() -> dict[str, object]:
    with closing(_connect()) as db:
        values = {row["key"]: row["value"] for row in db.execute("SELECT key, value FROM metadata")}
    generated = datetime_from_iso(values["generated_at"])
    age_days = max(0.0, (time.time() - generated) / 86400)
    refresh_file = DATA_ROOT / "refresh-status.json"
    refresh = json.loads(refresh_file.read_text()) if refresh_file.exists() else None
    refresh_stuck = bool(
        refresh and refresh.get("state") == "running"
        and time.time() - datetime_from_iso(str(refresh["attempted_at"])) > 6 * 60 * 60
    )
    degraded = age_days > 9 or refresh_stuck or bool(refresh and refresh.get("state") == "failed")
    return {
        "ready": True,
        "degraded": degraded,
        "age_days": round(age_days, 2),
        "build_id": values["build_id"],
        "generated_at": values["generated_at"],
        "essay_count": int(values["essay_count"]),
        "chunk_count": int(values["chunk_count"]),
        "embedding_model": values["embedding_model"],
        "refresh": refresh,
    }


def datetime_from_iso(value: str) -> float:
    from datetime import datetime
    return datetime.fromisoformat(value).timestamp()


def _row_result(row: sqlite3.Row, score: float, mode: str) -> dict[str, object]:
    return {
        "essay": row["title"], "slug": row["slug"], "url": row["url"],
        "published": row["published"], "chunk": row["chunk_index"],
        "start_line": row["start_line"], "end_line": row["end_line"],
        "score": round(float(score), 6), "search_mode": mode, "passage": row["text"],
    }


@lru_cache(maxsize=256)
def _embed_query(query: str) -> np.ndarray:
    return np.asarray(_embedding_function([query])[0], dtype=np.float32)


def _vectors() -> tuple[np.ndarray, list[dict[str, object]]]:
    global _vector_cache
    index_path = (CURRENT / "index.sqlite3").resolve()
    if _vector_cache is not None and _vector_cache[0] == index_path:
        return _vector_cache[1], _vector_cache[2]
    with closing(_connect()) as db:
        rows = db.execute("""
            SELECT c.slug, c.chunk_index, c.start_line, c.end_line, c.text, c.embedding,
                   e.title, e.url, e.published
            FROM chunks c JOIN essays e ON e.slug = c.slug
        """).fetchall()
    vectors = np.vstack([
        np.frombuffer(row["embedding"], dtype="<f4") for row in rows
    ]).astype(np.float32, copy=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1, norms)
    metadata = [{key: row[key] for key in row.keys() if key != "embedding"} for row in rows]
    _vector_cache = (index_path, vectors, metadata)
    return vectors, metadata


def _semantic(query: str, limit: int) -> list[dict[str, object]]:
    query_vector = _embed_query(query)
    query_vector = query_vector / (np.linalg.norm(query_vector) or 1.0)
    vectors, rows = _vectors()
    scores = vectors @ query_vector
    indices = np.argpartition(scores, -min(limit, len(scores)))[-limit:]
    indices = indices[np.argsort(scores[indices])[::-1]]
    return [_row_result(rows[index], float(scores[index]), "semantic") for index in indices]


def _fts_query(query: str) -> str:
    terms = re.findall(r"[\w'-]+", query, flags=re.UNICODE)
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms[:20])


def _expand_query(query: str) -> str:
    normalized = query.casefold()
    additions = []
    if "what to build" in normalized or "startup idea" in normalized:
        additions.extend(["startup ideas", "problems users want"])
    if "raise money" in normalized or "fundrais" in normalized:
        additions.extend(["funding investors", "ramen profitable"])
    return " ".join([query, *additions])


def _lexical(query: str, limit: int) -> list[dict[str, object]]:
    fts_query = _fts_query(query)
    if not fts_query:
        return []
    with closing(_connect()) as db:
        rows = db.execute("""
            SELECT c.slug, c.chunk_index, c.start_line, c.end_line, c.text,
                   e.title, e.url, e.published, bm25(chunks_fts, 8.0, 1.0) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN essays e ON e.slug = c.slug
            WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?
        """, (fts_query, limit)).fetchall()
    return [_row_result(row, -float(row["rank"]), "lexical") for row in rows]


def _diversify(results: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    deferred: list[dict[str, object]] = []
    seen: set[str] = set()
    for result in results:
        slug = str(result["slug"])
        if slug in seen:
            deferred.append(result)
        else:
            seen.add(slug)
            selected.append(result)
        if len(selected) == limit:
            return selected
    return (selected + deferred)[:limit]


def search(query: str, mode: Literal["hybrid", "semantic", "lexical"] = "hybrid", limit: int = 8) -> list[dict[str, object]]:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    limit = max(1, min(limit, 20))
    if mode == "semantic":
        return _diversify(_semantic(query, max(limit * 4, 20)), limit)
    if mode == "lexical":
        return _diversify(_lexical(query, max(limit * 4, 20)), limit)
    if mode != "hybrid":
        raise ValueError("mode must be hybrid, semantic, or lexical")

    expanded = _expand_query(query)
    semantic = _semantic(expanded, max(limit * 4, 20))
    lexical = _lexical(expanded, max(limit * 4, 20))
    by_key: dict[tuple[str, int], dict[str, object]] = {}
    scores: defaultdict[tuple[str, int], float] = defaultdict(float)
    for weight, results in ((2.0, semantic), (1.0, lexical)):
        for rank, result in enumerate(results, 1):
            key = (str(result["slug"]), int(result["chunk"]))
            by_key[key] = result
            scores[key] += weight / (60 + rank)
    query_terms = {term.casefold() for term in re.findall(r"[\w'-]+", query) if len(term) > 2}
    for key, result in by_key.items():
        title_terms = {term.casefold() for term in re.findall(r"[\w'-]+", str(result["essay"]))}
        scores[key] += 0.008 * len(query_terms & title_terms)
    ranked = sorted(scores, key=scores.get, reverse=True)
    output = []
    for key in ranked:
        result = dict(by_key[key])
        result["score"] = round(scores[key], 6)
        result["search_mode"] = "hybrid"
        output.append(result)
    return _diversify(output, limit)


def list_essays(query: str | None = None, limit: int = 250) -> list[dict[str, object]]:
    limit = max(1, min(limit, 250))
    with closing(_connect()) as db:
        if query:
            escaped = query.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pattern = f"%{escaped}%"
            rows = db.execute("""
                SELECT slug, title, url, published, word_count FROM essays
                WHERE title LIKE ? ESCAPE '\\' OR slug LIKE ? ESCAPE '\\' ORDER BY title LIMIT ?
            """, (pattern, pattern, limit)).fetchall()
        else:
            rows = db.execute("SELECT slug, title, url, published, word_count FROM essays ORDER BY title LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def get_essay(slug: str, start_line: int = 1, end_line: int = 300) -> dict[str, object]:
    if start_line < 1:
        raise ValueError("start_line must be at least 1")
    with closing(_connect()) as db:
        row = db.execute("SELECT slug, title, url, published, text FROM essays WHERE slug = ?", (slug,)).fetchone()
    if row is None:
        raise ValueError(f"unknown essay slug: {slug}")
    lines = row["text"].splitlines()
    if start_line > len(lines):
        raise ValueError(f"start_line {start_line} exceeds essay length {len(lines)}")
    if end_line < start_line:
        raise ValueError("end_line must be greater than or equal to start_line")
    end_line = min(end_line, start_line + 999)
    selected = lines[start_line - 1:end_line]
    return {
        "slug": row["slug"], "title": row["title"], "url": row["url"],
        "published": row["published"], "start_line": start_line,
        "end_line": min(end_line, len(lines)), "total_lines": len(lines),
        "text": "\n".join(f"{number}: {line}" for number, line in enumerate(selected, start_line)),
    }


def get_full_essay(slug: str) -> str:
    with closing(_connect()) as db:
        row = db.execute("SELECT text FROM essays WHERE slug = ?", (slug,)).fetchone()
    if row is None:
        raise ValueError(f"unknown essay slug: {slug}")
    return str(row["text"])


def grep(pattern: str, slug: str | None = None, ignore_case: bool = True,
         context_lines: int = 2, limit: int = 30) -> list[dict[str, object]]:
    if len(pattern) > 300:
        raise ValueError("pattern must be 300 characters or fewer")
    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        raise ValueError(f"invalid regular expression: {exc}") from exc
    context_lines = max(0, min(context_lines, 10))
    limit = max(1, min(limit, 100))
    with closing(_connect()) as db:
        if slug:
            rows = db.execute("SELECT slug, title, url, text FROM essays WHERE slug = ?", (slug,)).fetchall()
            if not rows:
                raise ValueError(f"unknown essay slug: {slug}")
        else:
            rows = db.execute("SELECT slug, title, url, text FROM essays ORDER BY title").fetchall()
    results = []
    last_match: dict[str, int] = {}
    deadline = time.monotonic() + 5
    for row in rows:
        lines = row["text"].splitlines()
        for index, line in enumerate(lines):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("regular expression exceeded the 5 second execution limit")
            try:
                matched = regex.search(line, timeout=min(0.1, remaining))
            except TimeoutError as exc:
                raise ValueError("regular expression exceeded the execution limit") from exc
            if not matched:
                continue
            if index <= last_match.get(row["slug"], -100) + context_lines * 2 + 1:
                continue
            first = max(0, index - context_lines)
            last = min(len(lines), index + context_lines + 1)
            results.append({
                "essay": row["title"], "slug": row["slug"], "url": row["url"],
                "line": index + 1,
                "context": "\n".join(f"{number + 1}: {lines[number]}" for number in range(first, last)),
            })
            last_match[row["slug"]] = index
            if len(results) >= limit:
                return results
    return results
