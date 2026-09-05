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
    """Лениво реэкспортирует опциональный API анонимизации PDF.

    Позволяет `import privacyguard_pipeline` продолжать работать без
    установленной опциональной группы зависимостей "pdf" — только
    обращение к PDFAnonymizer / PDFAnonymizationResult запускает
    собственный ленивый импорт privacyguard_pipeline.pdf (и его
    PDFDependencyError, если экстра отсутствует).

    Args:
        name: Атрибут, к которому обращаются в этом модуле.

    Returns:
        Запрошенный атрибут из подпакета pdf.

    Raises:
        AttributeError: Если name не входит в публичный API этого
            модуля.
    """
    if name not in ('PDFAnonymizer', 'PDFAnonymizationResult'):
        raise AttributeError(
            f'module {__name__!r} has no attribute {name!r}',
        )
    from privacyguard_pipeline import pdf as _pdf_mod

    return getattr(_pdf_mod, name)
