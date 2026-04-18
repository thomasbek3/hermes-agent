"""Semtools workspace plugin — semantic search via @llamaindex/semtools.

semtools is a Rust CLI that does semantic search using model2vec.
It auto-indexes files on first search, so index() is mostly a no-op.
"""

import fnmatch
import json
import logging
import shutil
import subprocess

from workspace.base import BaseIndexer
from workspace.config import WorkspaceConfig
from workspace.types import IndexSummary, SearchResult

log = logging.getLogger(__name__)


class SemtoolsIndexer(BaseIndexer):
    def __init__(self, config: WorkspaceConfig) -> None:
        self._config = config
        pc = config.plugin_config
        self._workspace = pc.get("workspace_name", "hermes")
        self._top_k = pc.get("top_k", 20)

    def index(self, *, progress=None) -> IndexSummary:
        """Discover files but skip actual indexing — semtools auto-indexes on search."""
        self._ensure_semtools()
        from workspace.files import discover_workspace_files

        discovery = discover_workspace_files(self._config)
        return IndexSummary(
            files_indexed=0,
            files_skipped=len(discovery.files),
            files_pruned=0,
            files_errored=0,
            chunks_created=0,
            duration_seconds=0.0,
            errors=[],
            errors_truncated=False,
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        path_prefix: str | None = None,
        file_glob: str | None = None,
    ) -> list[SearchResult]:
        """Run semtools search against discovered workspace files."""
        self._ensure_semtools()
        from workspace.files import discover_workspace_files

        discovery = discover_workspace_files(self._config)
        files = [str(p) for _, p in discovery.files]

        if path_prefix:
            files = [f for f in files if f.startswith(path_prefix)]
        if file_glob:
            pattern = file_glob if file_glob.startswith("*") else "*" + file_glob
            files = [f for f in files if fnmatch.fnmatch(f, pattern)]

        if not files:
            return []

        cmd = [
            "semtools",
            "search",
            query,
            *files,
            "--json",
            "--top-k",
            str(limit),
            "--workspace",
            self._workspace,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            log.error("semtools search failed: %s", e.stderr)
            raise RuntimeError(f"semtools search failed: {e.stderr}") from e
        except FileNotFoundError as e:
            raise RuntimeError(
                "semtools binary not found. Install with: npm i -g @llamaindex/semtools"
            ) from e

        return self._parse_results(result.stdout)

    def status(self) -> dict:
        installed = shutil.which("semtools") is not None
        return {
            "backend": "semtools",
            "installed": installed,
            "workspace_name": self._workspace,
        }

    def _ensure_semtools(self) -> None:
        """Install semtools if not already present (idempotent)."""
        if shutil.which("semtools"):
            return
        if not shutil.which("npm"):
            raise RuntimeError(
                "npm is required to install semtools. Install Node.js first."
            )
        try:
            subprocess.run(
                ["npm", "i", "-g", "@llamaindex/semtools"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to install semtools via npm: {e.stderr}") from e
        if not shutil.which("semtools"):
            raise RuntimeError(
                "semtools installed but not found on PATH after npm install"
            )

    @staticmethod
    def _parse_results(stdout: str) -> list[SearchResult]:
        """Parse semtools JSON output into SearchResult objects.

        semtools outputs::

            {
              "results": [
                {
                  "filename": "/path/to/file.py",
                  "start_line_number": 0,
                  "end_line_number": 7,
                  "match_line_number": 3,
                  "distance": 0.219,
                  "content": "..."
                },
                ...
              ]
            }

        Distance is a dissimilarity metric (lower = better match).
        We convert to a similarity score: score = 1.0 - distance.
        Line numbers from semtools are 0-based; we convert to 1-based.
        """
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            log.warning("Failed to parse semtools JSON output")
            return []

        results_raw = data.get("results", [])
        results: list[SearchResult] = []

        for i, item in enumerate(results_raw):
            distance = item.get("distance", 1.0)
            score = max(0.0, 1.0 - distance)

            start_line = item.get("start_line_number", 0) + 1
            end_line = item.get("end_line_number", 0) + 1
            content = item.get("content", "")

            results.append(
                SearchResult(
                    path=item.get("filename", ""),
                    line_start=start_line,
                    line_end=end_line,
                    section=None,
                    chunk_index=i,
                    score=round(score, 6),
                    tokens=0,
                    modified="",
                    content=content,
                )
            )

        return results


def register(ctx):
    ctx.register_workspace_indexer(SemtoolsIndexer)
