"""Ручной смоук-тест для анонимизации PDF.

Не часть набора pytest (см. privacyguard_pipeline/tests/ для этого) —
повторяет роль test.py как ручной, запускаемой человеком проверки.
Берёт настоящий PDF из docs/, прогоняет его через PDFAnonymizer и
сообщает только счётчики/статусы. Никогда не печатает извлечённый
текст или значения PII — в соответствии с инвариантом аудита проекта
(только типы сущностей и счётчики, никогда значения).

Использование:
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
