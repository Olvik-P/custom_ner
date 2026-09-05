"""Тесты для privacyguard_pipeline.mcp_server.registry (MaskRegistry)."""

from __future__ import annotations

import time

import pytest
from privacyguard_pipeline.masker import Masker
from privacyguard_pipeline.mcp_server.registry import (
    HandleNotFoundError,
    MaskRegistry,
)


def test_register_then_pop_round_trip() -> None:
    registry = MaskRegistry(ttl_seconds=60)
    masker = Masker()

    handle = registry.register(masker)
    assert registry.pop(handle) is masker


def test_pop_with_unknown_handle_raises() -> None:
    registry = MaskRegistry(ttl_seconds=60)

    with pytest.raises(HandleNotFoundError):
        registry.pop('never-issued')


def test_pop_twice_with_same_handle_raises_second_time() -> None:
    registry = MaskRegistry(ttl_seconds=60)
    handle = registry.register(Masker())

    registry.pop(handle)
    with pytest.raises(HandleNotFoundError):
        registry.pop(handle)


def test_expired_handle_is_unknown_to_pop() -> None:
    registry = MaskRegistry(ttl_seconds=0)
    handle = registry.register(Masker())

    time.sleep(0.01)
    with pytest.raises(HandleNotFoundError):
        registry.pop(handle)


def test_sweep_removes_only_expired_entries() -> None:
    # register()/pop() уже подметают просроченные записи сами (см.
    # docstring sweep()), поэтому здесь нужно проверить sweep()
    # изолированно, не вызывая их между регистрацией протухшей записи и
    # самим sweep() — иначе он найдёт нечего удалять.
    registry = MaskRegistry(ttl_seconds=0.05)
    stale_handle = registry.register(Masker())
    time.sleep(0.1)

    removed = registry.sweep()

    assert removed == 1
    assert len(registry) == 0
    with pytest.raises(HandleNotFoundError):
        registry.pop(stale_handle)
