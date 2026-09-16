from __future__ import annotations

import json
import os
import pathlib
import shutil
import tempfile
import unittest

from httpx import ASGITransport, AsyncClient
from mcp import Client
from mcp.shared.exceptions import MCPError
from starlette.testclient import TestClient

if "DATA_DIR" not in os.environ:
    raise RuntimeError("set DATA_DIR to a completed test index before running tests")

import search
import server
import indexer


class SearchTests(unittest.TestCase):
    def test_index_is_complete(self) -> None:
        corpus = search.status()
        self.assertGreaterEqual(corpus["essay_count"], 200)
        self.assertGreaterEqual(corpus["chunk_count"], 1500)

    def test_hybrid_search_returns_sources(self) -> None:
        results = search.search("How should a founder decide what to build?", limit=5)
        self.assertEqual(len(results), 5)
        self.assertTrue(all(result["url"].startswith("https://www.paulgraham.com/") for result in results))
        self.assertTrue(all(result["passage"] for result in results))
        self.assertTrue(all(result["start_line"] <= result["end_line"] for result in results))
        self.assertEqual(len({result["slug"] for result in results}), len(results))

    def test_judged_relevance(self) -> None:
        cases = {
            "How do I get startup ideas?": "startupideas",
            "Should I raise money or become ramen profitable?": "ramenprofitable",
            "How should founders do things that do not scale?": "ds",
        }
        for query, expected in cases.items():
            slugs = [result["slug"] for result in search.search(query, limit=5)]
            self.assertIn(expected, slugs, query)

    def test_grep_returns_line_context(self) -> None:
        results = search.grep(r"ramen profitable", limit=5)
        self.assertTrue(results)
        self.assertTrue(all(result["line"] >= 1 for result in results))

    def test_read_is_bounded(self) -> None:
        essay = search.get_essay("startupideas", 1, 10)
        self.assertLessEqual(essay["end_line"], 10)
        self.assertIn("1:", essay["text"])
        self.assertNotIn("Want to start a startup?", essay["text"])
        with self.assertRaisesRegex(ValueError, "start_line"):
            search.get_essay("startupideas", 999999, 999999)
        with self.assertRaisesRegex(ValueError, "end_line"):
            search.get_essay("startupideas", 20, 1)

    def test_catalog_search_treats_wildcards_literally(self) -> None:
        results = search.list_essays("%")
        self.assertEqual([result["slug"] for result in results], ["95"])

    def test_unknown_grep_slug_is_actionable(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown essay slug"):
            search.grep("founder", slug="not-an-essay")

    def test_status_contains_refresh_state(self) -> None:
        corpus = search.status()
        self.assertIn("degraded", corpus)
        self.assertIn("age_days", corpus)


class IndexLifecycleTests(unittest.TestCase):
    def test_corpus_regression_is_rejected(self) -> None:
        previous = {"one": "a", "two": "b"}
        with self.assertRaisesRegex(RuntimeError, "removed essays"):
            indexer.validate_corpus_change(previous, {"one": "a"})
        removed, changed = indexer.validate_corpus_change(previous, {"one": "a"}, allow_removals=True)
        self.assertEqual(removed, ["two"])
        self.assertEqual(changed, 0)

    def test_ensure_valid_restores_seed_over_corrupt_index(self) -> None:
        original = indexer.DATA_ROOT
        with tempfile.TemporaryDirectory() as directory:
            data = pathlib.Path(directory) / "data"
            seed = pathlib.Path(directory) / "seed"
            shutil.copytree(original, seed)
            (data / "versions" / "bad").mkdir(parents=True)
            (data / "versions" / "bad" / "index.sqlite3").write_text("not sqlite")
            (data / "versions" / "bad" / "manifest.json").write_text("{}")
            (data / "current").symlink_to("versions/bad")
            indexer.DATA_ROOT = data
            indexer.VERSIONS = data / "versions"
            indexer.CURRENT = data / "current"
            indexer.LOCK = data / ".refresh.lock"
            indexer.REFRESH_STATUS = data / "refresh-status.json"
            try:
                restored = indexer.ensure_valid_index(seed)
                self.assertEqual(restored["schema_version"], indexer.SCHEMA_VERSION)
                self.assertEqual(json.loads((data / "refresh-status.json").read_text())["state"], "seed-restored")
            finally:
                indexer.DATA_ROOT = original
                indexer.VERSIONS = original / "versions"
                indexer.CURRENT = original / "current"
                indexer.LOCK = original / ".refresh.lock"
                indexer.REFRESH_STATUS = original / "refresh-status.json"


class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_tools_resources_and_prompt(self) -> None:
        async with Client(server.mcp) as client:
            tools = (await client.list_tools()).tools
            self.assertEqual(
                {tool.name for tool in tools},
                {"search_pg", "grep_pg", "read_essay", "list_essays"},
            )
            self.assertTrue(all(tool.output_schema for tool in tools))
            read_schema = next(tool.output_schema for tool in tools if tool.name == "read_essay")
            self.assertIn("start_line", read_schema["properties"])

            result = await client.call_tool(
                "search_pg", {"query": "How should a founder decide what to build?", "limit": 3}
            )
            self.assertFalse(result.is_error)
            self.assertEqual(len(result.structured_content["result"]), 3)

            prompts = (await client.list_prompts()).prompts
            self.assertIn("what_would_pg_do", {prompt.name for prompt in prompts})
            resources = (await client.list_resources()).resources
            self.assertIn("pg://catalog", {str(resource.uri) for resource in resources})

            rendered = await client.get_prompt("what_would_pg_do", {"question": "Is it worth applying to YC?"})
            self.assertIn("paulgraham.com", rendered.messages[0].content.text)
            with self.assertRaises(MCPError) as empty:
                await client.get_prompt("what_would_pg_do", {"question": ""})
            self.assertEqual(empty.exception.code, -32602)
            with self.assertRaises(MCPError) as missing:
                await client.get_prompt("what_would_pg_do", {})
            self.assertEqual(missing.exception.code, -32602)

    async def test_http_origin_and_cors(self) -> None:
        app = server.create_app(public_host="testserver")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            denied = await client.post(
                "/mcp", headers={"Origin": "http://evil.example"}, json={"jsonrpc": "2.0"}
            )
            self.assertEqual(denied.status_code, 403)
            preflight = await client.options(
                "/mcp",
                headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
            )
            self.assertEqual(preflight.status_code, 204)
            self.assertEqual(preflight.headers["access-control-allow-origin"], "http://localhost:3000")
            stream = await client.get("/mcp", headers={"Accept": "text/event-stream"})
            self.assertEqual(stream.status_code, 405)
            self.assertEqual(stream.headers["allow"], "POST, OPTIONS")


class UsageTests(unittest.TestCase):
    def test_usage_metrics_are_aggregate_only(self) -> None:
        app = server.create_app(public_host="testserver")
        marker = "PRIVATE_QUERY_MARKER"
        with TestClient(app) as client:
            client.post(
                "/mcp",
                headers={
                    "Origin": "http://localhost:3000",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={
                    "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "search_pg", "arguments": {"query": marker}},
                },
            )
            metrics = client.get("/metrics")
            self.assertEqual(metrics.status_code, 200)
            payload = metrics.json()
            self.assertIn("tool_calls.search_pg", payload["counts"])
            self.assertIn("mcp_http_requests", payload["counts"])
            self.assertNotIn(marker, metrics.text)


if __name__ == "__main__":
    unittest.main()
