"""Public, read-only MCP server for Paul Graham's essays."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Literal
from urllib.parse import urlparse

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import DEFAULT_MAX_REQUEST_BODY_SIZE, TransportSecuritySettings
from mcp.shared.exceptions import MCPError
from mcp.types import INVALID_PARAMS, ToolAnnotations
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send
import uvicorn

import search
import usage

logging.getLogger("mcp").setLevel(logging.WARNING)


class SearchHit(BaseModel):
    essay: str
    slug: str
    url: str
    published: str | None
    chunk: int
    start_line: int
    end_line: int
    score: float
    search_mode: str
    passage: str


class GrepHit(BaseModel):
    essay: str
    slug: str
    url: str
    line: int
    context: str


class EssaySummary(BaseModel):
    slug: str
    title: str
    url: str
    published: str | None
    word_count: int


class EssayExcerpt(BaseModel):
    slug: str
    title: str
    url: str
    published: str | None
    start_line: int
    end_line: int
    total_lines: int
    text: str


mcp = MCPServer(
    "Paul Graham Essays",
    title="Paul Graham Essays",
    version="1.0.0",
    description="Semantic and exact-text search across Paul Graham's public essays.",
    website_url="https://www.paulgraham.com/articles.html",
    instructions=(
        "Use search_pg for conceptual questions and grep_pg for exact phrases or terminology. "
        "Use read_essay with line ranges to inspect surrounding source text. Cite essay titles "
        "and source URLs. Distinguish direct quotations from synthesis, and do not imply Paul "
        "Graham endorsed a conclusion unless the retrieved text supports it."
    ),
)

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
WORK_LIMIT = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENT_SEARCHES", "3")))
WORKERS = ThreadPoolExecutor(max_workers=int(os.environ.get("MAX_CONCURRENT_SEARCHES", "3")))


async def run_bounded(function, *args):
    try:
        await asyncio.wait_for(WORK_LIMIT.acquire(), timeout=2)
    except TimeoutError as exc:
        raise ToolError("server is busy; retry shortly") from exc
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(WORKERS, function, *args)
    release_on_exit = True
    try:
        return await asyncio.wait_for(asyncio.shield(future), timeout=15)
    except TimeoutError as exc:
        release_on_exit = False
        future.add_done_callback(lambda _: WORK_LIMIT.release())
        raise ToolError("search exceeded the 15 second execution limit") from exc
    except asyncio.CancelledError:
        release_on_exit = False
        future.add_done_callback(lambda _: WORK_LIMIT.release())
        raise
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    finally:
        if release_on_exit:
            WORK_LIMIT.release()


async def run_tool(name: str, function, *args):
    usage.record(f"tool_calls.{name}")
    try:
        return await run_bounded(function, *args)
    except ToolError:
        usage.record(f"tool_errors.{name}")
        raise


@mcp.tool(title="Search PG Essays", annotations=READ_ONLY, structured_output=True)
async def search_pg(
    query: str = Field(min_length=1, max_length=500, description="Natural-language question or concepts to retrieve."),
    mode: Literal["hybrid", "semantic", "lexical"] = Field(
        default="hybrid",
        description="hybrid is recommended; semantic finds concepts; lexical favors exact terms.",
    ),
    limit: int = Field(default=8, ge=1, le=20, description="Number of passages to return."),
) -> list[SearchHit]:
    """Retrieve grounded passages for RAG. Results include essay titles, stable slugs, URLs, and full passage text."""
    return [SearchHit(**item) for item in await run_tool("search_pg", search.search, query, mode, limit)]


@mcp.tool(title="Grep PG Essays", annotations=READ_ONLY, structured_output=True)
async def grep_pg(
    pattern: str = Field(min_length=1, max_length=300, description="Python regular expression to search line by line."),
    slug: str | None = Field(default=None, description="Optional essay slug to restrict the search."),
    ignore_case: bool = Field(default=True, description="Perform case-insensitive matching."),
    context_lines: int = Field(default=2, ge=0, le=10, description="Lines before and after each match."),
    limit: int = Field(default=30, ge=1, le=100, description="Maximum matches to return."),
) -> list[GrepHit]:
    """Run regex grep across the plaintext corpus and return line-addressable context."""
    return [GrepHit(**item) for item in await run_tool(
        "grep_pg", search.grep, pattern, slug, ignore_case, context_lines, limit
    )]


@mcp.tool(title="Read PG Essay", annotations=READ_ONLY, structured_output=True)
async def read_essay(
    slug: str = Field(description="Stable essay slug returned by search_pg, grep_pg, or list_essays."),
    start_line: int = Field(default=1, ge=1, description="First 1-based line to return."),
    end_line: int = Field(default=300, ge=1, description="Last 1-based line; at most 1000 lines per call."),
) -> EssayExcerpt:
    """Read a bounded line range from one essay, including canonical source metadata."""
    return EssayExcerpt(**await run_tool("read_essay", search.get_essay, slug, start_line, end_line))


@mcp.tool(title="List PG Essays", annotations=READ_ONLY, structured_output=True)
async def list_essays(
    query: str | None = Field(default=None, description="Optional title or slug substring."),
    limit: int = Field(default=250, ge=1, le=250, description="Maximum essays to return."),
) -> list[EssaySummary]:
    """List the indexed essay catalog; use this to discover stable slugs for read_essay."""
    return [EssaySummary(**item) for item in await run_tool("list_essays", search.list_essays, query, limit)]


@mcp.resource("pg://catalog", title="PG Essay Catalog", mime_type="application/json")
def catalog() -> list[dict[str, object]]:
    """All indexed essay titles, slugs, dates, URLs, and word counts."""
    usage.record("resource_reads.pg_catalog")
    return search.list_essays()


@mcp.resource("pg://essay/{slug}", title="PG Essay", mime_type="text/plain")
def essay_resource(slug: str) -> str:
    """The full plaintext of one essay, addressed by stable slug."""
    usage.record("resource_reads.pg_essay")
    return search.get_full_essay(slug)


@mcp.resource("pg://status", title="PG Index Status", mime_type="application/json")
def index_status() -> dict[str, object]:
    """Corpus freshness, size, build ID, and embedding model."""
    usage.record("resource_reads.pg_status")
    return search.status()


@mcp.prompt(name="what_would_pg_do", title="What Would PG Do?")
async def what_would_pg_do(
    question: str = Field(default="", description="Decision or question to analyze (1 to 500 characters)."),
) -> str:
    """Create a source-grounded prompt for reasoning from Paul Graham's published essays."""
    question = question.strip()
    if not question or len(question) > 500:
        usage.record("prompt_errors.what_would_pg_do")
        raise MCPError(INVALID_PARAMS, "question must contain 1 to 500 characters")
    hits = await run_bounded(search.search, question, "hybrid", 5)
    context = "\n\n".join(
        f"[{hit['essay']}]({hit['url']})\n{hit['passage']}" for hit in hits
    )
    usage.record("prompt_renders.what_would_pg_do")
    return (
        "Answer the question by reconstructing Paul Graham's likely advice from the source "
        "passages below. Separate direct claims from your inference. Be concise, preserve "
        "important caveats, and cite essay titles with their URLs. Do not imitate his identity "
        "or claim he personally answered.\n\n"
        f"Question: {question}\n\nSource passages:\n{context}"
    )


@mcp.custom_route("/", methods=["GET"], include_in_schema=False)
async def root(_: Request) -> JSONResponse:
    return JSONResponse({
        "name": "Paul Graham Essays MCP",
        "mcp_endpoint": "/mcp",
        "transport": "Streamable HTTP",
        "authentication": "none",
        "source": "https://www.paulgraham.com/articles.html",
        "tools": ["search_pg", "grep_pg", "read_essay", "list_essays"],
        "metrics_endpoint": "/metrics",
        "status": search.status(),
    })


@mcp.custom_route("/metrics", methods=["GET"], include_in_schema=False)
async def metrics(_: Request) -> JSONResponse:
    return JSONResponse(usage.snapshot())


@mcp.custom_route("/healthz", methods=["GET"], include_in_schema=False)
async def health(_: Request) -> JSONResponse:
    try:
        return JSONResponse(search.status())
    except Exception as exc:
        return JSONResponse({"ready": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=503)


class PublicOriginMiddleware:
    """Validate browser Origins while allowing this intentionally public MCP."""

    def __init__(self, app: ASGIApp, public_host: str):
        self.app = app
        self.public_host = public_host
        configured = os.environ.get("ALLOWED_ORIGINS", "")
        self.allowed_origins = {origin.strip() for origin in configured.split(",") if origin.strip()}
        self.allowed_origins.add(f"https://{public_host}")
        self.requests: defaultdict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def rate_limited(self, client: str) -> bool:
        now = time.monotonic()
        with self.lock:
            timestamps = self.requests[client]
            while timestamps and timestamps[0] < now - 60:
                timestamps.popleft()
            if len(timestamps) >= 120:
                return True
            timestamps.append(now)
            if len(self.requests) > 10000:
                for key in list(self.requests):
                    if not self.requests[key] or self.requests[key][-1] < now - 60:
                        del self.requests[key]
            return False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        origin: str | None = None
        if scope["type"] == "http" and scope.get("path") == "/mcp":
            usage.record("mcp_http_requests")
            headers = {key.decode().lower(): value.decode() for key, value in scope["headers"]}
            host = headers.get("host", "").split(":", 1)[0]
            if host not in {self.public_host, "localhost", "127.0.0.1"}:
                await Response("Invalid Host header", status_code=421)(scope, receive, send)
                return
            origin = headers.get("origin")
            if origin:
                parsed = urlparse(origin)
                local = parsed.hostname in {"localhost", "127.0.0.1"}
                if origin not in self.allowed_origins and not (local and parsed.scheme == "http"):
                    await Response("Invalid Origin header", status_code=403)(scope, receive, send)
                    return
            if scope["method"] == "OPTIONS":
                await Response(status_code=204, headers={
                    "Access-Control-Allow-Origin": origin or f"https://{self.public_host}",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "content-type, mcp-protocol-version, mcp-session-id",
                    "Access-Control-Max-Age": "86400",
                })(scope, receive, send)
                return
            client = headers.get("fly-client-ip") or headers.get("x-forwarded-for", "").split(",", 1)[0]
            if self.rate_limited(client or "unknown"):
                usage.record("rate_limited_429")
                await Response("Rate limit exceeded", status_code=429, headers={"Retry-After": "60"})(scope, receive, send)
                return
            if scope["method"] == "GET":
                await Response("Method Not Allowed", status_code=405, headers={"Allow": "POST, OPTIONS"})(scope, receive, send)
                return

        async def send_with_cors(message):
            if origin and message["type"] == "http.response.start":
                message.setdefault("headers", []).append((b"access-control-allow-origin", origin.encode()))
                message["headers"].append((b"vary", b"Origin"))
            await send(message)

        await self.app(scope, receive, send_with_cors)


def report_usage_periodically(interval: float) -> None:
    while True:
        time.sleep(interval)
        usage.report()


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    public_host = os.environ.get("PUBLIC_HOST", "pg-essays-mcp.fly.dev")
    app = create_app(host, public_host)
    interval = float(os.environ.get("USAGE_REPORT_SECONDS", "600"))
    threading.Thread(target=report_usage_periodically, args=(interval,), daemon=True).start()
    uvicorn.run(app, host=host, port=int(os.environ.get("PORT", "8080")),
                log_level="info", access_log=False)


def create_app(host: str = "0.0.0.0", public_host: str = "pg-essays-mcp.fly.dev") -> ASGIApp:
    app = mcp.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        max_request_body_size=DEFAULT_MAX_REQUEST_BODY_SIZE,
        host=host,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    return PublicOriginMiddleware(app, public_host)


if __name__ == "__main__":
    main()
