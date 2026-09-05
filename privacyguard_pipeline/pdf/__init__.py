"""Подпакет анонимизации PDF для PrivacyGuard Pipeline.

Публичная поверхность: PDFAnonymizer, PDFAnonymizationResult.
Внутренние модули (text_extractor, ocr, renderer, anonymizer) не
предназначены для прямого импорта извне этого подпакета.

Требует опциональную группу зависимостей "pdf" (PyMuPDF, pytesseract,
Pillow). Импорт этого подпакета никогда не требует этих зависимостей —
они импортируются лениво, только когда реально запрашиваются
PDFAnonymizer/PDFAnonymizationResult, поэтому остальная часть
privacyguard_pipeline продолжает работать без установленной экстры.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from privacyguard_pipeline.exceptions import PDFDependencyError

if TYPE_CHECKING:
    from privacyguard_pipeline.pdf.anonymizer import (
        PDFAnonymizationResult as PDFAnonymizationResult,
    )
    from privacyguard_pipeline.pdf.anonymizer import (
        PDFAnonymizer as PDFAnonymizer,
    )
    from privacyguard_pipeline.pdf.anonymizer import (
        detect_page_entity_counts as detect_page_entity_counts,
    )

__all__ = [
    'PDFAnonymizationResult',
    'PDFAnonymizer',
    'detect_page_entity_counts',
]

_PDF_EXTRA_HINT = (
    "PDF anonymization requires the optional 'pdf' dependency group "
    '(PyMuPDF, pytesseract, Pillow). Install it with: '
    'uv sync --extra pdf'
)


def __getattr__(name: str) -> Any:
    """Лениво импортирует реализацию pdf/ при первом обращении к атрибуту.

    Args:
        name: Атрибут, к которому обращаются в этом модуле.

    Returns:
        Запрошенный атрибут из anonymizer.py.

    Raises:
        AttributeError: Если name не входит в публичный API этого модуля.
        PDFDependencyError: Если опциональные зависимости "pdf" не
            установлены.
    """
    if name not in __all__:
        raise AttributeError(
            f'module {__name__!r} has no attribute {name!r}',
        )
    try:
        from privacyguard_pipeline.pdf import anonymizer as _anonymizer_mod
    except ImportError as exc:
        raise PDFDependencyError(_PDF_EXTRA_HINT) from exc
    return getattr(_anonymizer_mod, name)
