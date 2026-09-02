from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = '1.2.0'

if TYPE_CHECKING:
    from privacyguard_pipeline.pdf import (
        PDFAnonymizationResult as PDFAnonymizationResult,
    )
    from privacyguard_pipeline.pdf import PDFAnonymizer as PDFAnonymizer

__all__ = ['PDFAnonymizationResult', 'PDFAnonymizer', '__version__']


def __getattr__(name: str) -> Any:
    """Lazily re-export the optional PDF anonymization API.

    Keeps `import privacyguard_pipeline` working without the optional
    "pdf" dependency group installed — only accessing PDFAnonymizer /
    PDFAnonymizationResult triggers privacyguard_pipeline.pdf's own lazy
    import (and its PDFDependencyError if the extra is missing).

    Args:
        name: Attribute being accessed on this module.

    Returns:
        The requested attribute from the pdf subpackage.

    Raises:
        AttributeError: If name is not part of this module's public API.
    """
    if name not in ('PDFAnonymizer', 'PDFAnonymizationResult'):
        raise AttributeError(
            f'module {__name__!r} has no attribute {name!r}',
        )
    from privacyguard_pipeline import pdf as _pdf_mod

    return getattr(_pdf_mod, name)
