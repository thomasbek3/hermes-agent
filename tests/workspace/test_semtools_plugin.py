"""Tests for the semtools workspace plugin."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from workspace.base import BaseIndexer
from workspace.config import WorkspaceConfig
from workspace.types import IndexSummary, SearchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEMTOOLS_JSON_OUTPUT = json.dumps(
    {
        "results": [
            {
                "filename": "/workspace/foo.py",
                "start_line_number": 0,
                "end_line_number": 7,
                "match_line_number": 3,
                "distance": 0.2,
                "content": "def foo():\n    return 42",
            },
            {
                "filename": "/workspace/bar.md",
                "start_line_number": 4,
                "end_line_number": 10,
                "match_line_number": 6,
                "distance": 0.55,
                "content": "# Bar\n\nSome documentation.",
            },
        ]
    }
)


@pytest.fixture
def semtools_config(tmp_path):
    ws_root = tmp_path / "workspace"
    ws_root.mkdir()
    return WorkspaceConfig(
        workspace_root=ws_root,
        indexer="semtools",
        plugin_config={"workspace_name": "test_ws", "top_k": 10},
    )


@pytest.fixture
def semtools_config_defaults(tmp_path):
    ws_root = tmp_path / "workspace"
    ws_root.mkdir()
    return WorkspaceConfig(
        workspace_root=ws_root,
        indexer="semtools",
    )


def _make_indexer(config):
    from plugins.workspace.semtools import SemtoolsIndexer

    return SemtoolsIndexer(config)


# ---------------------------------------------------------------------------
# 1. Subclass check
# ---------------------------------------------------------------------------


def test_semtools_indexer_is_base_indexer_subclass():
    from plugins.workspace.semtools import SemtoolsIndexer

    assert issubclass(SemtoolsIndexer, BaseIndexer)


# ---------------------------------------------------------------------------
# 2. Plugin discovery
# ---------------------------------------------------------------------------


def test_plugin_discovery_finds_semtools():
    from plugins.workspace import load_workspace_indexer

    cls = load_workspace_indexer("semtools")
    assert cls is not None
    assert cls.__name__ == "SemtoolsIndexer"


# ---------------------------------------------------------------------------
# 3. get_indexer factory
# ---------------------------------------------------------------------------


def test_get_indexer_returns_semtools(semtools_config):
    from plugins.workspace.semtools import SemtoolsIndexer
    from workspace import get_indexer

    indexer = get_indexer(semtools_config)
    assert isinstance(indexer, SemtoolsIndexer)
    assert isinstance(indexer, BaseIndexer)


# ---------------------------------------------------------------------------
# 4. Constructor config
# ---------------------------------------------------------------------------


def test_constructor_reads_plugin_config(semtools_config):
    indexer = _make_indexer(semtools_config)
    assert indexer._workspace == "test_ws"
    assert indexer._top_k == 10


def test_constructor_uses_defaults(semtools_config_defaults):
    indexer = _make_indexer(semtools_config_defaults)
    assert indexer._workspace == "hermes"
    assert indexer._top_k == 20


# ---------------------------------------------------------------------------
# 5. _ensure_semtools
# ---------------------------------------------------------------------------


def test_ensure_semtools_noop_when_installed(semtools_config):
    indexer = _make_indexer(semtools_config)
    with patch("shutil.which", return_value="/usr/bin/semtools"):
        indexer._ensure_semtools()  # should not raise


def test_ensure_semtools_installs_when_missing(semtools_config):
    indexer = _make_indexer(semtools_config)
    call_count = 0

    def which_side_effect(name):
        nonlocal call_count
        if name == "semtools":
            call_count += 1
            # First call: not installed, second call: installed
            return None if call_count <= 1 else "/usr/bin/semtools"
        if name == "npm":
            return "/usr/bin/npm"
        return None

    with (
        patch("plugins.workspace.semtools.shutil.which", side_effect=which_side_effect),
        patch("plugins.workspace.semtools.subprocess.run") as mock_run,
    ):
        indexer._ensure_semtools()
        mock_run.assert_called_once_with(
            ["npm", "i", "-g", "@llamaindex/semtools"],
            check=True,
            capture_output=True,
            text=True,
        )


def test_ensure_semtools_raises_without_npm(semtools_config):
    indexer = _make_indexer(semtools_config)
    with patch("plugins.workspace.semtools.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="npm is required"):
            indexer._ensure_semtools()


def test_ensure_semtools_raises_on_install_failure(semtools_config):
    indexer = _make_indexer(semtools_config)

    def which_side_effect(name):
        if name == "npm":
            return "/usr/bin/npm"
        return None

    with (
        patch("plugins.workspace.semtools.shutil.which", side_effect=which_side_effect),
        patch(
            "plugins.workspace.semtools.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "npm", stderr="install error"),
        ),
    ):
        with pytest.raises(RuntimeError, match="Failed to install"):
            indexer._ensure_semtools()


def test_ensure_semtools_raises_when_not_on_path_after_install(semtools_config):
    indexer = _make_indexer(semtools_config)

    def which_side_effect(name):
        if name == "npm":
            return "/usr/bin/npm"
        return None  # semtools never appears

    with (
        patch("plugins.workspace.semtools.shutil.which", side_effect=which_side_effect),
        patch("plugins.workspace.semtools.subprocess.run"),
    ):
        with pytest.raises(RuntimeError, match="not found on PATH"):
            indexer._ensure_semtools()


# ---------------------------------------------------------------------------
# 6. index()
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeDiscovery:
    files: list
    complete: bool = True
    filtered_count: int = 0


def test_index_returns_summary(semtools_config):
    indexer = _make_indexer(semtools_config)
    fake_files = [("/root", Path("/root/a.py")), ("/root", Path("/root/b.md"))]
    fake_discovery = _FakeDiscovery(files=fake_files)

    with (
        patch(
            "plugins.workspace.semtools.shutil.which", return_value="/usr/bin/semtools"
        ),
        patch(
            "workspace.files.discover_workspace_files",
            return_value=fake_discovery,
        ),
    ):
        summary = indexer.index()

    assert isinstance(summary, IndexSummary)
    assert summary.files_indexed == 0
    assert summary.files_skipped == 2
    assert summary.chunks_created == 0
    assert summary.errors == []


# ---------------------------------------------------------------------------
# 7. search()
# ---------------------------------------------------------------------------


def test_search_calls_semtools_cli(semtools_config):
    indexer = _make_indexer(semtools_config)
    fake_files = [("/root", Path("/root/a.py")), ("/root", Path("/root/b.md"))]
    fake_discovery = _FakeDiscovery(files=fake_files)
    mock_result = MagicMock()
    mock_result.stdout = SEMTOOLS_JSON_OUTPUT

    with (
        patch(
            "plugins.workspace.semtools.shutil.which", return_value="/usr/bin/semtools"
        ),
        patch(
            "workspace.files.discover_workspace_files",
            return_value=fake_discovery,
        ),
        patch(
            "plugins.workspace.semtools.subprocess.run", return_value=mock_result
        ) as mock_run,
    ):
        indexer.search("test query", limit=5)

    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert call_args[0] == "semtools"
    assert call_args[1] == "search"
    assert call_args[2] == "test query"
    assert "/root/a.py" in call_args
    assert "/root/b.md" in call_args
    assert "--json" in call_args
    assert "--top-k" in call_args
    assert "5" in call_args
    assert "--workspace" in call_args
    assert "test_ws" in call_args


def test_search_parses_results(semtools_config):
    indexer = _make_indexer(semtools_config)
    fake_files = [("/root", Path("/root/a.py"))]
    fake_discovery = _FakeDiscovery(files=fake_files)
    mock_result = MagicMock()
    mock_result.stdout = SEMTOOLS_JSON_OUTPUT

    with (
        patch(
            "plugins.workspace.semtools.shutil.which", return_value="/usr/bin/semtools"
        ),
        patch(
            "workspace.files.discover_workspace_files",
            return_value=fake_discovery,
        ),
        patch("plugins.workspace.semtools.subprocess.run", return_value=mock_result),
    ):
        results = indexer.search("test query")

    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)

    # First result
    assert results[0].path == "/workspace/foo.py"
    assert results[0].line_start == 1  # 0-based -> 1-based
    assert results[0].line_end == 8  # 7 + 1
    assert results[0].score == pytest.approx(0.8, abs=0.001)
    assert results[0].content == "def foo():\n    return 42"
    assert results[0].section is None
    assert results[0].chunk_index == 0

    # Second result
    assert results[1].path == "/workspace/bar.md"
    assert results[1].line_start == 5  # 4 + 1
    assert results[1].score == pytest.approx(0.45, abs=0.001)
    assert results[1].chunk_index == 1


def test_search_returns_empty_for_no_files(semtools_config):
    indexer = _make_indexer(semtools_config)
    fake_discovery = _FakeDiscovery(files=[])

    with (
        patch(
            "plugins.workspace.semtools.shutil.which", return_value="/usr/bin/semtools"
        ),
        patch(
            "workspace.files.discover_workspace_files",
            return_value=fake_discovery,
        ),
    ):
        results = indexer.search("test query")

    assert results == []


def test_search_handles_invalid_json(semtools_config):
    indexer = _make_indexer(semtools_config)
    fake_files = [("/root", Path("/root/a.py"))]
    fake_discovery = _FakeDiscovery(files=fake_files)
    mock_result = MagicMock()
    mock_result.stdout = "not valid json"

    with (
        patch(
            "plugins.workspace.semtools.shutil.which", return_value="/usr/bin/semtools"
        ),
        patch(
            "workspace.files.discover_workspace_files",
            return_value=fake_discovery,
        ),
        patch("plugins.workspace.semtools.subprocess.run", return_value=mock_result),
    ):
        results = indexer.search("test query")

    assert results == []


def test_search_raises_on_cli_failure(semtools_config):
    indexer = _make_indexer(semtools_config)
    fake_files = [("/root", Path("/root/a.py"))]
    fake_discovery = _FakeDiscovery(files=fake_files)

    with (
        patch(
            "plugins.workspace.semtools.shutil.which", return_value="/usr/bin/semtools"
        ),
        patch(
            "workspace.files.discover_workspace_files",
            return_value=fake_discovery,
        ),
        patch(
            "plugins.workspace.semtools.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "semtools", stderr="boom"),
        ),
    ):
        with pytest.raises(RuntimeError, match="semtools search failed"):
            indexer.search("test query")


# ---------------------------------------------------------------------------
# 8. search() filtering: path_prefix and file_glob
# ---------------------------------------------------------------------------


def test_search_filters_by_path_prefix(semtools_config):
    indexer = _make_indexer(semtools_config)
    fake_files = [
        ("/root", Path("/root/src/a.py")),
        ("/root", Path("/root/docs/b.md")),
        ("/root", Path("/root/src/c.py")),
    ]
    fake_discovery = _FakeDiscovery(files=fake_files)
    mock_result = MagicMock()
    mock_result.stdout = json.dumps({"results": []})

    with (
        patch(
            "plugins.workspace.semtools.shutil.which", return_value="/usr/bin/semtools"
        ),
        patch(
            "workspace.files.discover_workspace_files",
            return_value=fake_discovery,
        ),
        patch(
            "plugins.workspace.semtools.subprocess.run", return_value=mock_result
        ) as mock_run,
    ):
        indexer.search("query", path_prefix="/root/src")

    call_args = mock_run.call_args[0][0]
    # Files appear between the query and the --json flag
    json_idx = call_args.index("--json")
    file_args = call_args[3:json_idx]
    assert "/root/src/a.py" in file_args
    assert "/root/src/c.py" in file_args
    assert "/root/docs/b.md" not in file_args


def test_search_filters_by_file_glob(semtools_config):
    indexer = _make_indexer(semtools_config)
    fake_files = [
        ("/root", Path("/root/a.py")),
        ("/root", Path("/root/b.md")),
        ("/root", Path("/root/c.py")),
    ]
    fake_discovery = _FakeDiscovery(files=fake_files)
    mock_result = MagicMock()
    mock_result.stdout = json.dumps({"results": []})

    with (
        patch(
            "plugins.workspace.semtools.shutil.which", return_value="/usr/bin/semtools"
        ),
        patch(
            "workspace.files.discover_workspace_files",
            return_value=fake_discovery,
        ),
        patch(
            "plugins.workspace.semtools.subprocess.run", return_value=mock_result
        ) as mock_run,
    ):
        indexer.search("query", file_glob="*.py")

    call_args = mock_run.call_args[0][0]
    json_idx = call_args.index("--json")
    file_args = call_args[3:json_idx]
    assert "/root/a.py" in file_args
    assert "/root/c.py" in file_args
    assert "/root/b.md" not in file_args


def test_search_returns_empty_when_all_filtered(semtools_config):
    indexer = _make_indexer(semtools_config)
    fake_files = [("/root", Path("/root/a.py"))]
    fake_discovery = _FakeDiscovery(files=fake_files)

    with (
        patch(
            "plugins.workspace.semtools.shutil.which", return_value="/usr/bin/semtools"
        ),
        patch(
            "workspace.files.discover_workspace_files",
            return_value=fake_discovery,
        ),
    ):
        results = indexer.search("query", path_prefix="/nonexistent")

    assert results == []


# ---------------------------------------------------------------------------
# 9. status()
# ---------------------------------------------------------------------------


def test_status_when_installed(semtools_config):
    indexer = _make_indexer(semtools_config)
    with patch(
        "plugins.workspace.semtools.shutil.which", return_value="/usr/bin/semtools"
    ):
        status = indexer.status()

    assert status["backend"] == "semtools"
    assert status["installed"] is True
    assert status["workspace_name"] == "test_ws"


def test_status_when_not_installed(semtools_config):
    indexer = _make_indexer(semtools_config)
    with patch("plugins.workspace.semtools.shutil.which", return_value=None):
        status = indexer.status()

    assert status["backend"] == "semtools"
    assert status["installed"] is False


# ---------------------------------------------------------------------------
# 10. _parse_results edge cases
# ---------------------------------------------------------------------------


def test_parse_results_empty_json():
    from plugins.workspace.semtools import SemtoolsIndexer

    results = SemtoolsIndexer._parse_results(json.dumps({"results": []}))
    assert results == []


def test_parse_results_clamps_negative_score():
    from plugins.workspace.semtools import SemtoolsIndexer

    data = json.dumps(
        {
            "results": [
                {
                    "filename": "/a.py",
                    "start_line_number": 0,
                    "end_line_number": 1,
                    "match_line_number": 0,
                    "distance": 1.5,
                    "content": "x",
                }
            ]
        }
    )
    results = SemtoolsIndexer._parse_results(data)
    assert len(results) == 1
    assert results[0].score == 0.0
