"""Тесты для privacyguard_pipeline.mcp_server.registry (MaskRegistry)."""

from __future__ import annotations

import time
from pathlib import Path

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


# ---------------------------------------------------------------------------
# masked_file_path — файл на диске, связанный с handle (для mask_file)
# ---------------------------------------------------------------------------


def test_pop_deletes_associated_masked_file(tmp_path: Path) -> None:
    registry = MaskRegistry(ttl_seconds=60)
    masked_path = tmp_path / 'note_masked.txt'
    masked_path.write_text('<PER_ABCD1234>', encoding='utf-8')

    handle = registry.register(Masker(), masked_file_path=masked_path)
    registry.pop(handle)

    assert not masked_path.exists()


def test_sweep_deletes_masked_file_of_expired_entry(tmp_path: Path) -> None:
    registry = MaskRegistry(ttl_seconds=0.05)
    masked_path = tmp_path / 'note_masked.txt'
    masked_path.write_text('<PER_ABCD1234>', encoding='utf-8')
    registry.register(Masker(), masked_file_path=masked_path)

    time.sleep(0.1)
    removed = registry.sweep()

    assert removed == 1
    assert not masked_path.exists()


def test_register_without_masked_file_path_still_works() -> None:
    # register()/pop() без masked_file_path (mask/demask для голого
    # текста) не должны требовать его и не должны пытаться удалить
    # файл, которого никогда не было.
    registry = MaskRegistry(ttl_seconds=60)
    masker = Masker()

    handle = registry.register(masker)
    assert registry.pop(handle) is masker
