"""PDF anonymization subpackage for PrivacyGuard Pipeline.

Public surface: PDFAnonymizer, PDFAnonymizationResult. Internal modules
(text_extractor, ocr, renderer, anonymizer) are not meant to be imported
directly from outside this subpackage.

Requires the optional "pdf" dependency group (PyMuPDF, pytesseract,
Pillow). Importing this subpackage never requires those dependencies —
they are only imported, lazily, when PDFAnonymizer/PDFAnonymizationResult
are actually accessed, so the rest of privacyguard_pipeline keeps working
without the extra installed.
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

__all__ = ['PDFAnonymizationResult', 'PDFAnonymizer']

_PDF_EXTRA_HINT = (
    "PDF anonymization requires the optional 'pdf' dependency group "
    '(PyMuPDF, pytesseract, Pillow). Install it with: '
    'uv sync --extra pdf'
)


def __getattr__(name: str) -> Any:
    """Lazily import the pdf/ implementation on first attribute access.

    Args:
        name: Attribute being accessed on this module.

    Returns:
        The requested attribute from anonymizer.py.

    Raises:
        AttributeError: If name is not part of this module's public API.
        PDFDependencyError: If the optional "pdf" dependencies are not
            installed.
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
