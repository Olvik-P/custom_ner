"""deny_list в MCP-инструментах; allow_list через MCP недоступен."""

from __future__ import annotations

import inspect
from pathlib import Path

import fitz
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from privacyguard_pipeline.config import settings
from privacyguard_pipeline.detection import PIIDetector
from privacyguard_pipeline.mcp_server import server as mcp_server
from privacyguard_pipeline.mcp_server.registry import MaskRegistry
from privacyguard_pipeline.mcp_server.server import (
    anonymize_pdf_to_path,
    detect_file,
    mask_file_contents,
    mask_text,
)

_TEXT = 'Внутри проекта Заря и ПРОЕКТ ЗАРЯ идут работы'


def test_mask_text_masks_deny_entry_and_round_trips(
    detector: PIIDetector,
) -> None:
    registry = MaskRegistry(ttl_seconds=60)

    masked = mask_text(_TEXT, detector, registry, deny_list=['Проект Заря'])

    assert 'Заря' not in masked['masked_text']
    assert 'ЗАРЯ' not in masked['masked_text']
    assert '<CUSTOM_' in masked['masked_text']


def test_mask_text_without_deny_list_leaves_text_alone(
    detector: PIIDetector,
) -> None:
    registry = MaskRegistry(ttl_seconds=60)

    masked = mask_text(_TEXT, detector, registry)

    assert '<CUSTOM_' not in masked['masked_text']


def test_deny_list_does_not_leak_into_next_call(
    detector: PIIDetector,
) -> None:
    registry = MaskRegistry(ttl_seconds=60)
    mask_text(_TEXT, detector, registry, deny_list=['Проект Заря'])

    later = mask_text(_TEXT, detector, registry)

    assert '<CUSTOM_' not in later['masked_text']


def test_configured_deny_list_applies_to_tool_calls(
    detector: PIIDetector,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, 'deny_list', ['Проект Заря'])
    registry = MaskRegistry(ttl_seconds=60)

    masked = mask_text(_TEXT, detector, registry, deny_list=[])

    assert '<CUSTOM_' in masked['masked_text']


def test_mask_text_rejects_invalid_entry_without_creating_handle(
    detector: PIIDetector,
) -> None:
    registry = MaskRegistry(ttl_seconds=60)

    with pytest.raises(ToolError, match=r'deny_list\[0\]'):
        mask_text(_TEXT, detector, registry, deny_list=[''])

    assert len(registry) == 0


def test_error_text_does_not_contain_entry(detector: PIIDetector) -> None:
    registry = MaskRegistry(ttl_seconds=60)
    secret = 'секрет' * 40

    with pytest.raises(ToolError) as info:
        mask_text(_TEXT, detector, registry, deny_list=[secret])

    assert 'секрет' not in str(info.value)


def test_mask_file_contents_applies_deny_list(
    tmp_path: Path,
    detector: PIIDetector,
) -> None:
    path = tmp_path / 'note.txt'
    path.write_text(_TEXT, encoding='utf-8')
    registry = MaskRegistry(ttl_seconds=60)

    result = mask_file_contents(
        str(path),
        detector,
        registry,
        deny_list=['Проект Заря'],
    )

    masked = Path(result['masked_path']).read_text(encoding='utf-8')
    assert 'Заря' not in masked
    assert '<CUSTOM_' in masked


def test_mask_file_contents_rejects_invalid_entry_without_side_effects(
    tmp_path: Path,
    detector: PIIDetector,
) -> None:
    path = tmp_path / 'note.txt'
    path.write_text(_TEXT, encoding='utf-8')
    registry = MaskRegistry(ttl_seconds=60)

    with pytest.raises(ToolError):
        mask_file_contents(str(path), detector, registry, deny_list=['   '])

    assert not (tmp_path / 'note_masked.txt').exists()
    assert len(registry) == 0


def test_detect_file_counts_custom_entities(
    tmp_path: Path,
    detector: PIIDetector,
) -> None:
    path = tmp_path / 'note.txt'
    path.write_text(_TEXT, encoding='utf-8')

    result = detect_file(str(path), detector, deny_list=['Проект Заря'])

    assert result['entity_counts'].get('CUSTOM') == 2


def test_detect_file_rejects_invalid_entry(
    tmp_path: Path,
    detector: PIIDetector,
) -> None:
    path = tmp_path / 'note.txt'
    path.write_text(_TEXT, encoding='utf-8')

    with pytest.raises(ToolError):
        detect_file(str(path), detector, deny_list=['x'] * 501)


def test_anonymize_pdf_redacts_deny_entry(
    tmp_path: Path,
    cyrillic_font_path: str,
    detector: PIIDetector,
) -> None:
    source = tmp_path / 'in.pdf'
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 100),
        'Внутри проекта Заря идут работы',
        fontfile=cyrillic_font_path,
        fontname='F1',
    )
    doc.save(str(source))
    doc.close()

    result = anonymize_pdf_to_path(
        source,
        tmp_path / 'out.pdf',
        detector,
        deny_list=['Проект Заря'],
    )

    assert sum(result['redacted_by_type'].values()) >= 1
    with fitz.open(str(tmp_path / 'out.pdf')) as out:
        assert 'Заря' not in out[0].get_text()


def test_anonymize_pdf_rejects_invalid_entry_without_output(
    tmp_path: Path,
    detector: PIIDetector,
) -> None:
    out = tmp_path / 'out.pdf'

    with pytest.raises(ToolError):
        anonymize_pdf_to_path(
            tmp_path / 'in.pdf',
            out,
            detector,
            deny_list=[''],
        )

    assert not out.exists()


@pytest.mark.parametrize(
    'tool',
    [
        mcp_server.mask,
        mcp_server.demask,
        mcp_server.mask_file,
        mcp_server.close_masked_file,
        mcp_server.anonymize_pdf,
        mcp_server.detect,
    ],
)
def test_no_tool_accepts_allow_list(tool: object) -> None:
    params = inspect.signature(tool).parameters  # type: ignore[arg-type]

    assert 'allow_list' not in params


@pytest.mark.parametrize(
    'tool',
    [
        mcp_server.mask,
        mcp_server.mask_file,
        mcp_server.anonymize_pdf,
        mcp_server.detect,
    ],
)
def test_detection_tools_accept_deny_list(tool: object) -> None:
    params = inspect.signature(tool).parameters  # type: ignore[arg-type]

    assert 'deny_list' in params
