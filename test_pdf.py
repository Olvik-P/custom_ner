"""Manual smoke test for PDF anonymization.

Not part of the pytest suite (see privacyguard_pipeline/tests/ for that) —
mirrors test.py's role as a manual, human-run check. Takes a real PDF from
docs/, runs it through PDFAnonymizer, and reports only counts/statuses.
Never prints extracted text or PII values — matches the project's audit
invariant (entity types and counts only, never values).

Usage:
    python test_pdf.py [input.pdf] [output.pdf]
"""

from __future__ import annotations

import sys
from pathlib import Path

from privacyguard_pipeline.pdf import PDFAnonymizer

DEFAULT_INPUT = Path('docs') / 'Решение (безбумажное).pdf'


def main() -> None:
    input_pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    output_pdf = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else input_pdf.with_name(
            f'{input_pdf.stem}_redacted{input_pdf.suffix}',
        )
    )

    print(f'Входной файл:  {input_pdf}')
    print(f'Выходной файл: {output_pdf}')
    print()

    result = PDFAnonymizer().anonymize(input_pdf, output_pdf)

    print(f'Успех:               {result.success}')
    print(f'Страниц обработано:  {result.pages_processed}')
    print(f'Всего PII зачёркнуто: {result.total_spans_redacted}')
    print(f'По типам:            {result.redacted_by_type}')
    if not result.success:
        print(f'Ошибка:              {result.error_message}')


if __name__ == '__main__':
    main()
