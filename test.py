import asyncio
from privacyguard_pipeline.main import process


async def main():
    result = await process(
        "Пациент Иванов Пётр Сергеевич, тел. +7(916)123-45-67"
    )
    print(f'Анонимный текс:\n {result["anonymized_text"]}')
    print(f'Ответ модели:\n {result["llm_response"]}')
    print(f'Статистика:\n {result["stats"]}')

asyncio.run(main())
