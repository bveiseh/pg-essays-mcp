#!/usr/bin/env python3
"""Build an atomic, locally embedded SQLite index of Paul Graham's essays."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import re
import shutil
import sqlite3
import struct
import sys
import time
import uuid
from datetime import UTC, datetime

import numpy as np
import requests
from bs4 import BeautifulSoup, Comment
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

BASE_URL = "https://www.paulgraham.com/"
ROOT = pathlib.Path(__file__).parent
DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR", ROOT / "data"))
VERSIONS = DATA_ROOT / "versions"
CURRENT = DATA_ROOT / "current"
LOCK = DATA_ROOT / ".refresh.lock"
REFRESH_STATUS = DATA_ROOT / "refresh-status.json"
SCHEMA_VERSION = "2"
USER_AGENT = "pg-essays-mcp/1.0 (+https://pg-essays-mcp.fly.dev)"
NAV = {
    "index.html", "articles.html", "rss.html", "books.html", "faq.html",
    "bio.html", "arc.html", "bel.html", "lisp.html", "antispam.html",
    "kedrosky.html", "raq.html", "quo.html", "ind.html",
}
MONTHS = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)
DATE_RE = re.compile(r"(?m)^(" + "|".join(MONTHS) + r")\s+\d{4}$")
PROMO_LINES = {"Want to start a startup?", "Get funded by", "Y Combinator", "."}


def fetch(url: str) -> str:
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def essay_links(index_html: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(index_html, "html.parser")
    seen: set[str] = set()
    links: list[tuple[str, str]] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not re.fullmatch(r"[a-zA-Z0-9_-]+\.html", href):
            continue
        if href in NAV or anchor.find_parent("map") or href in seen:
            continue
        seen.add(href)
        links.append((href, anchor.get_text(" ", strip=True)))
    return links


def extract_text(html: str, title: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    for tag in soup(["script", "style", "map", "noscript"]):
        tag.decompose()
    lines = [re.sub(r"\s+", " ", line).strip() for line in soup.get_text("\n").splitlines()]
    lines = [line for line in lines if line]
    if lines and lines[0].casefold() == title.casefold():
        lines.pop(0)
    while lines and lines[0] in PROMO_LINES:
        lines.pop(0)
    # Essay pages sometimes prepend YC promotional copy before the publication date.
    for index, line in enumerate(lines[:12]):
        if DATE_RE.fullmatch(line):
            lines = lines[index:]
            break
    return "\n".join(lines)


def chunk_text(text: str, max_words: int = 300, overlap_words: int = 45) -> list[tuple[str, int, int]]:
    lines = text.splitlines()
    chunks: list[tuple[str, int, int]] = []
    start = 0
    while start < len(lines):
        end = start
        words = 0
        while end < len(lines) and (words < max_words or end == start):
            words += len(lines[end].split())
            end += 1
        chunks.append(("\n".join(lines[start:end]), start + 1, end))
        if end == len(lines):
            break
        overlap = 0
        next_start = end
        while next_start > start and overlap < overlap_words:
            next_start -= 1
            overlap += len(lines[next_start].split())
        start = max(start + 1, next_start)
    return chunks


def acquire_lock():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    lock = LOCK.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("another refresh is already running") from exc
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()
    return lock


def write_json_atomic(path: pathlib.Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def activate(root: pathlib.Path) -> None:
    next_link = DATA_ROOT / ".current-next"
    next_link.unlink(missing_ok=True)
    next_link.symlink_to(root.relative_to(DATA_ROOT), target_is_directory=True)
    next_link.replace(CURRENT)


def ensure_valid_index(seed_root: pathlib.Path) -> dict[str, object]:
    try:
        return validate_index()
    except Exception as exc:
        print(f"active index invalid, restoring bundled seed: {exc}", file=sys.stderr, flush=True)
    seed_manifest = validate_index(seed_root / "current")
    source = (seed_root / "current").resolve()
    destination = VERSIONS / source.name
    VERSIONS.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            validate_index(destination)
        except Exception:
            shutil.rmtree(destination)
    if not destination.exists():
        shutil.copytree(source, destination)
    activate(destination)
    write_json_atomic(REFRESH_STATUS, {
        "attempted_at": datetime.now(UTC).isoformat(), "state": "seed-restored",
        "build_id": seed_manifest["build_id"],
    })
    return validate_index()


def validate_index(root: pathlib.Path | None = None) -> dict[str, object]:
    root = root or CURRENT
    database = root / "index.sqlite3"
    manifest_file = root / "manifest.json"
    if not database.is_file() or not manifest_file.is_file():
        raise RuntimeError("index files are missing")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    db = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("SQLite quick check failed")
        metadata = dict(db.execute("SELECT key, value FROM metadata"))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported index schema {metadata.get('schema_version')!r}; expected {SCHEMA_VERSION}"
            )
        essay_count = db.execute("SELECT count(*) FROM essays").fetchone()[0]
        chunk_count = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        fts_count = db.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
        if essay_count < 200 or chunk_count < 1500 or chunk_count != fts_count:
            raise RuntimeError("index row counts are incomplete or inconsistent")
        if int(manifest["essay_count"]) != essay_count or int(manifest["chunk_count"]) != chunk_count:
            raise RuntimeError("manifest does not match index row counts")
        return manifest
    finally:
        db.close()


def current_corpus() -> dict[str, str]:
    try:
        validate_index()
    except (OSError, RuntimeError, ValueError, sqlite3.Error, json.JSONDecodeError):
        return {}
    db = sqlite3.connect(f"file:{CURRENT / 'index.sqlite3'}?mode=ro", uri=True)
    try:
        return dict(db.execute("SELECT slug, sha256 FROM essays"))
    finally:
        db.close()


def validate_corpus_change(previous: dict[str, str], candidate: dict[str, str],
                           allow_removals: bool = False) -> tuple[list[str], int]:
    removed = sorted(set(previous) - set(candidate))
    changed = sum(previous[slug] != candidate[slug] for slug in set(previous) & set(candidate))
    if previous and removed and not allow_removals:
        raise RuntimeError(f"refusing corpus with removed essays: {', '.join(removed[:20])}")
    if previous and changed > max(10, len(previous) // 4):
        raise RuntimeError(f"refusing suspicious corpus rewrite: {changed}/{len(previous)} essays changed")
    return removed, changed


def create_database(path: pathlib.Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.executescript("""
        PRAGMA journal_mode = DELETE;
        PRAGMA synchronous = FULL;
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE essays (
            slug TEXT PRIMARY KEY, title TEXT NOT NULL, url TEXT NOT NULL,
            published TEXT, text TEXT NOT NULL, word_count INTEGER NOT NULL,
            sha256 TEXT NOT NULL
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY, slug TEXT NOT NULL, chunk_index INTEGER NOT NULL,
            start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
            title TEXT NOT NULL, text TEXT NOT NULL, embedding BLOB NOT NULL,
            FOREIGN KEY (slug) REFERENCES essays(slug)
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(title, text, content='chunks', content_rowid='id');
        CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
          INSERT INTO chunks_fts(rowid, title, text) VALUES (new.id, new.title, new.text);
        END;
        CREATE INDEX chunks_slug_idx ON chunks(slug, chunk_index);
    """)
    return db


def build_index(force: bool = False, allow_removals: bool = False) -> dict[str, object]:
    lock = acquire_lock()
    build_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    staging = VERSIONS / ("." + build_id)
    final = VERSIONS / build_id
    try:
        VERSIONS.mkdir(parents=True, exist_ok=True)
        for abandoned in VERSIONS.glob(".*"):
            if abandoned.is_dir() and time.time() - abandoned.stat().st_mtime > 24 * 60 * 60:
                shutil.rmtree(abandoned, ignore_errors=True)
        staging.mkdir()
        corpus = staging / "corpus"
        corpus.mkdir()
        raw_cache = DATA_ROOT / "raw"
        raw_cache.mkdir(exist_ok=True)

        links = essay_links(fetch(BASE_URL + "articles.html"))
        essays: list[dict[str, object]] = []
        chunks: list[tuple[str, int, int, int, str]] = []
        for number, (filename, title) in enumerate(links, 1):
            cached = raw_cache / filename
            if force or not cached.exists():
                html = fetch(BASE_URL + filename)
                cached.write_text(html, encoding="utf-8")
                time.sleep(0.2)
            else:
                html = cached.read_text(encoding="utf-8", errors="replace")
            text = extract_text(html, title)
            if len(text) < 200:
                raise RuntimeError(f"refusing short scrape for {filename}: {len(text)} chars")
            slug = filename.removesuffix(".html")
            published = (match.group(0) if (match := DATE_RE.search(text[:3000])) else None)
            essay = {
                "slug": slug,
                "title": title,
                "url": BASE_URL + filename,
                "published": published,
                "text": text,
                "word_count": len(text.split()),
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
            essays.append(essay)
            (corpus / f"{slug}.txt").write_text(text + "\n", encoding="utf-8")
            chunks.extend(
                (slug, i, start_line, end_line, chunk)
                for i, (chunk, start_line, end_line) in enumerate(chunk_text(text))
            )
            if number % 25 == 0:
                print(f"scraped {number}/{len(links)}", flush=True)

        if len(essays) < 200:
            raise RuntimeError(f"refusing incomplete corpus with only {len(essays)} essays")

        previous = current_corpus()
        candidate = {str(essay["slug"]): str(essay["sha256"]) for essay in essays}
        removed, changed = validate_corpus_change(previous, candidate, allow_removals)

        embed = DefaultEmbeddingFunction()
        titles = {str(essay["slug"]): str(essay["title"]) for essay in essays}
        texts = [f"{titles[chunk[0]]}\n{chunk[4]}" for chunk in chunks]
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), 96):
            vectors.extend(embed(texts[start:start + 96]))
            print(f"embedded {min(start + 96, len(texts))}/{len(texts)}", flush=True)

        db = create_database(staging / "index.sqlite3")
        generated_at = datetime.now(UTC).isoformat()
        db.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [("build_id", build_id), ("generated_at", generated_at),
             ("essay_count", str(len(essays))), ("chunk_count", str(len(chunks))),
             ("embedding_model", "all-MiniLM-L6-v2 (Chroma ONNX)"),
             ("schema_version", SCHEMA_VERSION)],
        )
        db.executemany(
            "INSERT INTO essays VALUES (:slug, :title, :url, :published, :text, :word_count, :sha256)",
            essays,
        )
        db.executemany(
            "INSERT INTO chunks(slug, chunk_index, start_line, end_line, title, text, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(slug, index, start_line, end_line, titles[slug], text,
              sqlite3.Binary(struct.pack(f"<{len(vector)}f", *vector)))
             for (slug, index, start_line, end_line, text), vector in zip(chunks, vectors, strict=True)],
        )
        db.commit()
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("SQLite integrity check failed")
        db.close()

        manifest = {
            "build_id": build_id,
            "generated_at": generated_at,
            "essay_count": len(essays),
            "chunk_count": len(chunks),
            "word_count": sum(int(essay["word_count"]) for essay in essays),
            "embedding_model": "all-MiniLM-L6-v2 (Chroma ONNX)",
            "schema_version": SCHEMA_VERSION,
            "added_slugs": sorted(set(candidate) - set(previous)),
            "removed_slugs": removed,
            "changed_essays": changed,
        }
        write_json_atomic(staging / "manifest.json", manifest)
        validate_index(staging)
        staging.rename(final)
        activate(final)

        old_versions = sorted((p for p in VERSIONS.iterdir() if not p.name.startswith(".")), reverse=True)
        for old in old_versions[2:]:
            shutil.rmtree(old)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def mark_interrupted() -> None:
    try:
        status = json.loads(REFRESH_STATUS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if status.get("state") != "running":
        return
    write_json_atomic(REFRESH_STATUS, {
        "attempted_at": status.get("attempted_at"),
        "finished_at": datetime.now(UTC).isoformat(),
        "state": "failed", "error": "refresh process was interrupted",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="redownload cached essay HTML")
    parser.add_argument("--if-stale-days", type=float, help="skip when the current index is newer")
    parser.add_argument("--allow-removals", action="store_true", help="allow essays to disappear")
    parser.add_argument("--validate", action="store_true", help="validate the active index and exit")
    parser.add_argument("--ensure-valid", metavar="SEED_DIR", help="restore bundled seed if active index is invalid")
    parser.add_argument("--mark-interrupted", action="store_true", help="record a failed refresh if a previous attempt was killed")
    args = parser.parse_args()
    if args.mark_interrupted:
        mark_interrupted()
        return
    if args.ensure_valid:
        print(json.dumps(ensure_valid_index(pathlib.Path(args.ensure_valid)), indent=2))
        return
    if args.validate:
        print(json.dumps(validate_index(), indent=2))
        return
    attempted_at = datetime.now(UTC).isoformat()
    if args.if_stale_days is not None and (CURRENT / "manifest.json").exists():
        manifest = json.loads((CURRENT / "manifest.json").read_text())
        generated = datetime.fromisoformat(manifest["generated_at"])
        if (datetime.now(UTC) - generated).total_seconds() < args.if_stale_days * 86400:
            write_json_atomic(REFRESH_STATUS, {
                "attempted_at": attempted_at, "finished_at": datetime.now(UTC).isoformat(),
                "state": "skipped-fresh", "build_id": manifest["build_id"],
            })
            print(json.dumps({"skipped": True, "reason": "index is fresh", **manifest}, indent=2))
            return
    write_json_atomic(REFRESH_STATUS, {"attempted_at": attempted_at, "state": "running"})
    try:
        manifest = build_index(force=args.force, allow_removals=args.allow_removals)
    except Exception as exc:
        write_json_atomic(REFRESH_STATUS, {
            "attempted_at": attempted_at, "finished_at": datetime.now(UTC).isoformat(),
            "state": "failed", "error": f"{type(exc).__name__}: {exc}",
        })
        raise
    write_json_atomic(REFRESH_STATUS, {
        "attempted_at": attempted_at, "finished_at": datetime.now(UTC).isoformat(),
        "state": "succeeded", "build_id": manifest["build_id"],
    })
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
