"""Основная точка входа для PrivacyGuard Pipeline.

Предоставляет:
- Асинхронный метод process() для программного использования.
- Режим CLI: python main.py "текст для обработки"
- Обработку сигналов для корректного завершения (SIGINT/Ctrl+C).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from privacyguard_pipeline.config import settings
from privacyguard_pipeline.constants import PANEL_PREVIEW_CHARS
from privacyguard_pipeline.exceptions import (
    PrivacyGuardError,
    TextTooLongError,
)
from privacyguard_pipeline.pipeline import PrivacyGuardPipeline

# Настройка UTF-8 для Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[union-attr]

console = Console()
logger = logging.getLogger(__name__)

# Глобальные ссылки на пайплайн/задачу для обработчика сигналов
_pipeline: PrivacyGuardPipeline | None = None
_main_task: asyncio.Task[Any] | None = None


def _setup_logging() -> None:
    """Настраивает логирование для приложения."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                'logs/privacyguard.log',
                encoding='utf-8',
            ),
        ],
    )


def _handle_signal(sig: int, frame: object) -> None:
    """Обрабатывает SIGINT/Ctrl+C для корректного завершения.

    Отменяет выполняющуюся основную задачу вместо прямого вызова
    ``sys.exit()``: жёсткий выход изнутри обработчика сигнала рушит
    event loop, не давая ``finally: await _pipeline.close()`` внутри
    ``process()`` шанса выполниться. Отмена доставляется в следующей
    точке await задачи и нормально разворачивается через этот
    ``finally``, поэтому очистка реально завершается до выхода
    процесса.
    """
    console.print('\n[yellow]Shutting down gracefully...[/yellow]')
    if _main_task is not None:
        _main_task.get_loop().call_soon_threadsafe(_main_task.cancel)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы CLI с помощью argparse.

    Args:
        argv: Опциональный список аргументов (по умолчанию
            sys.argv[1:]).

    Returns:
        Пространство имён разобранных аргументов с полями 'text' и
        'system_prompt'.
    """
    parser = argparse.ArgumentParser(
        description=(
            'PrivacyGuard Pipeline — PII anonymization for LLM API calls'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python main.py "Text with PII"\n'
            '  python main.py --system "Be concise" -- "Text with PII"\n'
            '  echo "Text with PII" | python main.py'
        ),
    )
    parser.add_argument(
        'text',
        nargs='*',
        help='Text to process (if omitted, reads from stdin)',
    )
    parser.add_argument(
        '--system',
        type=str,
        default=None,
        help='Optional system prompt for the LLM',
    )
    args = parser.parse_args(argv)

    # Если текст не передан через аргументы, пробуем читать из stdin
    if not args.text:
        if not sys.stdin.isatty():
            stdin_text = sys.stdin.read().strip()
            if stdin_text:
                args.text = [stdin_text]

    if not args.text:
        parser.error(
            'No text provided. Pass text as argument or pipe via stdin.'
        )

    args.text = ' '.join(args.text)
    return args


def _print_result(result: dict[str, Any]) -> None:
    """Печатает результат пайплайна в форматированном виде через Rich.

    Args:
        result: Словарь результата пайплайна.
    """
    console.print(
        Panel(
            result['anonymized_text'][:PANEL_PREVIEW_CHARS],
            title='[bold blue]Anonymized Text[/bold blue]',
            border_style='blue',
        ),
    )

    llm_response = result.get('llm_response', '')
    if llm_response:
        console.print(
            Panel(
                llm_response[:PANEL_PREVIEW_CHARS],
                title='[bold green]LLM Response (Demasked)[/bold green]',
                border_style='green',
            ),
        )
    else:
        console.print(
            Panel(
                '[yellow]No LLM response (no API key or API error)[/yellow]',
                title='[bold yellow]LLM Response[/bold yellow]',
                border_style='yellow',
            ),
        )

    # Таблица статистики
    stats = result.get('stats', {})
    if isinstance(stats, dict):
        table = Table(title='Session Statistics', border_style='cyan')
        table.add_column('Metric', style='cyan')
        table.add_column('Value', style='white')

        for key, value in stats.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    table.add_row(f'  {sub_key}', str(sub_value))
            else:
                table.add_row(key.replace('_', ' ').title(), str(value))

        console.print(table)


async def process(
    text: str,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Обрабатывает текст через пайплайн PrivacyGuard.

    Это основной публичный метод API.

    Args:
        text: Входной текст, который может содержать PII.
        system_prompt: Опциональный системный промпт для LLM.

    Returns:
        Словарь с anonymized_text, llm_response и stats.

    Raises:
        TextTooLongError: Если текст превышает максимально допустимую
            длину.
        PrivacyGuardError: При других ошибках пайплайна.
    """
    global _pipeline

    _pipeline = PrivacyGuardPipeline()
    try:
        result = await _pipeline.process(text, system_prompt)
        return result
    except TextTooLongError:
        raise
    except PrivacyGuardError:
        raise
    except Exception as exc:
        raise PrivacyGuardError(f'Pipeline error: {exc}') from exc
    finally:
        await _pipeline.close()
        _pipeline = None


async def _run_with_task_tracking(
    text: str,
    system_prompt: str | None,
) -> dict[str, Any]:
    """Запускает process(), сохраняя задачу для обработчика сигналов."""
    global _main_task
    _main_task = asyncio.current_task()
    return await process(text, system_prompt)


def main() -> None:
    """Точка входа CLI."""
    _setup_logging()

    # Регистрируем обработчики сигналов
    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, _handle_signal)

    # Разбираем аргументы CLI
    args = _parse_args()

    console.print('[bold]PrivacyGuard Pipeline[/bold]')
    console.print(f'Processing text ({len(args.text)} chars)...\n')

    try:
        result = asyncio.run(
            _run_with_task_tracking(args.text, args.system),
        )
        _print_result(result)
    except TextTooLongError as exc:
        console.print(f'[bold red]Error:[/bold red] {exc}')
        sys.exit(1)
    except PrivacyGuardError as exc:
        console.print(f'[bold red]Error:[/bold red] {exc}')
        sys.exit(1)
    except (asyncio.CancelledError, KeyboardInterrupt):
        console.print('\n[yellow]Interrupted by user[/yellow]')
        sys.exit(0)


if __name__ == '__main__':
    main()
