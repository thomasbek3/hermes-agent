"""Tests for TypedDict shape definitions added in commit fc00f699.

Verifies that _CamofoxConfig is importable, honours total=False
(all fields optional), and nests correctly inside _BrowserConfig.
"""

from __future__ import annotations


def test_camofox_config_is_partial_typeddict():
    """_CamofoxConfig should accept zero or more keys (total=False)."""
    from hermes_cli.config import _CamofoxConfig, _BrowserConfig

    # total=False: constructing with no keys must succeed at runtime
    cfg_empty: _CamofoxConfig = {}
    cfg_with_field: _CamofoxConfig = {"managed_persistence": True}

    assert cfg_empty == {}
    assert cfg_with_field.get("managed_persistence") is True


def test_camofox_config_nested_in_browser_config():
    """_CamofoxConfig should be accepted in the camofox slot of _BrowserConfig."""
    from hermes_cli.config import _CamofoxConfig, _BrowserConfig

    browser: _BrowserConfig = {
        "inactivity_timeout": 60,
        "command_timeout": 10,
        "record_sessions": False,
        "allow_private_urls": False,
        "cdp_url": "http://localhost:9222",
        "camofox": {"managed_persistence": False},
    }

    assert browser["camofox"].get("managed_persistence") is False


def test_camofox_config_total_false_flag():
    """_CamofoxConfig.__total__ must be False (all fields optional)."""
    from hermes_cli.config import _CamofoxConfig

    assert _CamofoxConfig.__total__ is False
