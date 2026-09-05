"""Ручной смоук-тест: прогоняет настоящие шаги пайплайна с реальным LLM.

Повторяет PrivacyGuardPipeline.process() (детекция -> маскирование ->
LLM -> демаскирование) шаг за шагом вместо вызова как чёрного ящика,
чтобы был виден промежуточный результат каждого этапа: что было
обнаружено, как оно замаскировано, сырой (ещё замаскированный) ответ
LLM и демаскированный результат.
"""

import asyncio
import os
from pathlib import Path

# Settings() резолвит ".env" относительно текущей директории процесса
# (см. config.py), а этот файл лежит в privacyguard_pipeline/, а не
# здесь, в корне репозитория — поэтому `python test.py` подхватывает
# ключ API только при запуске изнутри privacyguard_pipeline/. Меняем
# директорию туда до импорта чего-либо из пакета (config.py собирает
# `settings` во время импорта), чтобы скрипт работал одинаково
# независимо от того, откуда его запускают.
os.chdir(Path(__file__).resolve().parent / 'privacyguard_pipeline')

from privacyguard_pipeline.detection import PIIDetector  # noqa: E402
from privacyguard_pipeline.exceptions import (  # noqa: E402
    LLMAuthenticationError,
    LLMConnectionError,
)
from privacyguard_pipeline.llm_proxy import LLMProxy  # noqa: E402
from privacyguard_pipeline.masker import Masker  # noqa: E402

TEXT = (
    'Пациент Иванов Пётр Сергеевич, тел. +7(916)123-45-67. '
    'Повтори имя пациента и его телефон дословно.'
)


def _section(title: str) -> None:
    print(f'\n{"=" * 60}\n{title}\n{"=" * 60}')


async def main() -> None:
    detector = PIIDetector()
    masker = Masker()
    llm_proxy = LLMProxy()

    try:
        _section('1. Исходный текст')
        print(TEXT)

        _section('2. Обнаруженные PII-сущности')
        detection = detector.detect(TEXT)
        if detection.spans:
            for span in detection.spans:
                print(
                    f'  {span.entity_type:<8} '
                    f'{span.text!r:<30} '
                    f'source={span.source} confidence={span.confidence}',
                )
        else:
            print('  (ничего не найдено)')

        _section('3. Замаскированный текст (отправляется в LLM)')
        masked_text = masker.mask(TEXT, detection.spans)
        print(masked_text)

        if masker.mapping:
            print('\n  Карта маскирования (token -> оригинал):')
            for token, entry in masker.mapping.items():
                print(
                    f'    {token:<20} -> {entry.original!r} '
                    f'({entry.entity_type})',
                )

        _section('4. Ответ LLM (как пришёл, до демаскирования)')
        try:
            raw_response = await llm_proxy.send(masked_text)
            print(raw_response)
        except (LLMAuthenticationError, LLMConnectionError) as exc:
            print(f'  Ошибка запроса к LLM: {exc}')
            raw_response = None

        if raw_response is not None:
            _section('5. Ответ LLM после демаскирования')
            demasked_response = masker.demask(raw_response)
            print(demasked_response)
    finally:
        masker.clear()
        await llm_proxy.close()


if __name__ == '__main__':
    asyncio.run(main())
